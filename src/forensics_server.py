#!/usr/bin/env python3
"""
forensics_server.py - upgraded with RL attacker trainer, Stackelberg reward shaping,
CSV logging, robust checkpoint loading, explain endpoints, health & attacker controls.

Usage:
    python -m src.forensics_server --checkpoint results/checkpoints/defender_final.pth --device cpu

Notes:
 - Requires src.forensics_worker to exist and provide explain_all, _infer_input_dim, to_native.
 - AttackerEnv (optional) should be accessible at src.attacker_env.AttackerEnv for richer attacker states.
"""
import argparse
import json
import time
import os
import csv
import threading
import traceback
from collections import deque, namedtuple
from typing import Tuple, Optional

from flask import Flask, request, jsonify

# try to import attacker environment (optional)
try:
    from src.attacker_env import AttackerEnv
except Exception:
    AttackerEnv = None

# local explainability worker (must exist)
from src.forensics_worker import explain_all, _infer_input_dim, to_native  # noqa: E402

# lazy torch/numpy imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
except Exception:
    torch = None
    nn = None
    F = None
    np = __import__("numpy")

# ----------------------------
# Defender network & loader
# ----------------------------
class DQNNetwork(nn.Module):
    def __init__(self, input_dim=15, hidden1=256, hidden2=128, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """Wrapper for defender policy network (loads checkpoint)."""
    def __init__(self, input_dim=15, n_actions=5, device="cpu", hidden1=256, hidden2=128):
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.device = torch.device(device) if torch is not None else "cpu"
        self.policy_net = DQNNetwork(self.input_dim, hidden1, hidden2, self.n_actions)
        if torch is not None:
            self.policy_net.to(self.device)
            self.policy_net.eval()

    def predict_np(self, x_np: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch required")
        t = torch.tensor(np.asarray(x_np, dtype=np.float32), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.policy_net(t)
            if isinstance(out, tuple):
                out = out[0]
            return out.detach().cpu().numpy()

    def load_checkpoint(self, checkpoint_path: str):
        if torch is None:
            raise RuntimeError("torch required")
        ck = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ck, dict) and ("policy_state_dict" in ck or "state_dict" in ck):
            sd = ck.get("policy_state_dict", ck.get("state_dict"))
        else:
            sd = ck
        try:
            self.policy_net.load_state_dict(sd)
            return {"loaded": "ok", "mode": "direct"}
        except Exception as e:
            # try to infer dims & auto-adjust
            try:
                possible_weights = []
                for k, v in sd.items():
                    if hasattr(v, "shape") and len(v.shape) == 2:
                        possible_weights.append((k, v.shape))
                final = None
                for k, shape in possible_weights:
                    if "net.4.weight" in k or k.endswith("net.4.weight") or k.endswith(".net.4.weight"):
                        final = (k, shape)
                        break
                if final is None and possible_weights:
                    final = possible_weights[-1]
                if final is None:
                    raise RuntimeError("cannot infer network dims from checkpoint")
                out_dim, in_dim = int(final[1][0]), int(final[1][1])
                hidden1 = max(128, min(512, in_dim * 2))
                hidden2 = max(64, min(256, in_dim))
                self.policy_net = DQNNetwork(in_dim, hidden1, hidden2, out_dim).to(self.device)
                self.policy_net.load_state_dict(sd)
                self.input_dim = in_dim
                self.n_actions = out_dim
                return {"loaded": "auto_adjusted", "inferred": {"input_dim": in_dim, "hidden1": hidden1, "hidden2": hidden2, "output": out_dim}}
            except Exception as e2:
                raise RuntimeError(f"Failed to load checkpoint: {e}") from e2

# ----------------------------
# Attacker: small DQN + trainer
# ----------------------------
Transition = namedtuple('Transition', ('s', 'a', 'r', 's2', 'done'))

class SmallDQN(nn.Module):
    def __init__(self, input_dim=15, hidden=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


class AttackerTrainer:
    """Background trainer thread for attacker. Optional AttackerEnv integration."""
    def __init__(
        self,
        input_dim=15,
        n_actions=3,
        device="cpu",
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        checkpoint_path="results/checkpoints/attacker_live.pth",
        save_every=300
    ):
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.device = torch.device(device) if torch is not None else "cpu"
        self.net = SmallDQN(self.input_dim, hidden=64, output_dim=self.n_actions)
        self.target_net = SmallDQN(self.input_dim, hidden=64, output_dim=self.n_actions)
        if torch is not None:
            self.net.to(self.device)
            self.target_net.to(self.device)
            self.target_net.load_state_dict(self.net.state_dict())
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.opt = None
        self.gamma = gamma
        self.replay = ReplayBuffer(capacity=50000)
        self.batch_size = int(batch_size)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.step = 0
        self.stats = {"episode": 0, "avg_reward": 0.0, "last_loss": 0.0}
        self.checkpoint_path = checkpoint_path
        self.save_every = int(save_every)
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 1e-4
        # optional environment integration
        self.env = AttackerEnv(None, self.input_dim) if AttackerEnv is not None else None
        self.defender_agent = None

    def set_defender(self, defender_agent: Optional[DQNAgent]):
        """Link defender so env can call it for next-state/rewards if needed."""
        self.defender_agent = defender_agent
        if self.env is not None:
            self.env.defender = defender_agent

    def act(self, state_np: np.ndarray) -> int:
        if torch is None:
            return int(np.random.randint(0, self.n_actions))
        if np.random.rand() < self.eps:
            return int(np.random.randint(0, self.n_actions))
        t = torch.tensor(state_np.reshape(1, -1).astype(np.float32)).to(self.device)
        with torch.no_grad():
            q = self.net(t)
            return int(q.argmax(dim=1).item())

    def push_transition(self, s, a, r, s2, done):
        # accept None s2; store as-is
        self.replay.push(np.asarray(s, dtype=np.float32), int(a), float(r),
                         np.asarray(s2, dtype=np.float32) if s2 is not None else None, bool(done))

    def _learn_step(self):
        if torch is None or len(self.replay) < max(256, self.batch_size):
            return 0.0
        batch = self.replay.sample(self.batch_size)
        s = torch.tensor(np.stack(batch.s), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array(batch.a), dtype=torch.int64).to(self.device)
        r = torch.tensor(np.array(batch.r, dtype=np.float32)).to(self.device)
        non_final_mask = np.array([not d for d in batch.done], dtype=np.bool_)
        if non_final_mask.any():
            s2s = np.stack([x for (x, d) in zip(batch.s2, batch.done) if not d])
            s2 = torch.tensor(s2s, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                next_q = self.target_net(s2).max(dim=1)[0]
            next_q_full = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            idxs = np.where(non_final_mask)[0]
            next_q_full[idxs] = next_q
        else:
            next_q_full = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        q_values = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        target = r + self.gamma * next_q_full
        loss = F.mse_loss(q_values, target.detach())
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        if self.step % 100 == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def _training_loop(self):
        # optional episodic loop using env if available
        # This loop mainly trains from replay; external transitions (via push_transition) are also accepted.
        while self.running:
            self.eps = max(self.eps_min, self.eps - self.eps_decay)
            self.step += 1
            loss = self._learn_step()
            # update stats
            with self.lock:
                self.stats["last_loss"] = loss
            # periodic saving + rotation
            if self.step % self.save_every == 0:
                try:
                    self.save(self.checkpoint_path)
                    base = os.path.splitext(self.checkpoint_path)[0]
                    rotated = f"{base}_{self.step}.pth"
                    torch.save(self.net.state_dict(), rotated)
                    # keep last 3
                    d = os.path.dirname(self.checkpoint_path) or "."
                    base_name = os.path.basename(base)
                    ckpts = sorted([f for f in os.listdir(d) if f.startswith(base_name) and f.endswith(".pth")],
                                   key=lambda x: os.path.getmtime(os.path.join(d, x)))
                    if len(ckpts) > 3:
                        for old in ckpts[:-3]:
                            try:
                                os.remove(os.path.join(d, old))
                            except Exception:
                                pass
                except Exception:
                    print("[!] Attacker checkpoint save error:", traceback.format_exc())
            # training log CSV
            try:
                os.makedirs("results", exist_ok=True)
                log_path = "results/training_log.csv"
                write_header = not os.path.exists(log_path)
                with open(log_path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    if write_header:
                        writer.writerow(["timestamp", "episode", "step", "epsilon", "last_loss", "replay_len", "avg_reward"])
                    writer.writerow([time.time(), self.stats.get("episode", 0), self.step, round(self.eps, 5),
                                     round(self.stats.get("last_loss", 0.0), 8), len(self.replay),
                                     round(self.stats.get("avg_reward", 0.0), 6)])
                # rotate large file
                if os.path.getsize(log_path) > 5 * 1024 * 1024:
                    os.rename(log_path, f"{log_path}.{int(time.time())}.rot")
            except Exception:
                print("[!] Training log write error:", traceback.format_exc())
            time.sleep(0.01)

    def start(self):
        if self.running:
            return False
        self.running = True
        self.thread = threading.Thread(target=self._training_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        if not self.running:
            return False
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        return True

    def save(self, path=None):
        if torch is None:
            return False
        p = path or self.checkpoint_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save(self.net.state_dict(), p)
        return True

    def load(self, path):
        if torch is None:
            return False
        if os.path.exists(path):
            sd = torch.load(path, map_location=self.device)
            self.net.load_state_dict(sd)
            self.target_net.load_state_dict(self.net.state_dict())
            return True
        return False

    def status(self):
        with self.lock:
            return dict(step=self.step, eps=self.eps, stats=self.stats, replay_len=len(self.replay), running=self.running)

# ----------------------------
# Utility & Flask wiring
# ----------------------------
APP = Flask(__name__)
AGENT: Optional[DQNAgent] = None
CHECKPOINT = None
DEVICE = "cpu"
INPUT_DIM = 15
N_ACTIONS = 5
BACKGROUND_CACHE = None
CSV_LOG_PATH = "results/forensics_log.csv"

ATTACKER_TRAINER: Optional[AttackerTrainer] = None
ATTACKER_CHECKPOINT = "results/checkpoints/attacker_live.pth"
STACKELBERG_LAMBDA = 0.5

# ----------------------------
# Forensic decision helpers
# ----------------------------
def _forensic_decision(agent: Optional[DQNAgent], features: np.ndarray) -> Tuple[str, float, int]:
    try:
        if torch is None or agent is None:
            return "allow", 0.0, 0
        model_dev = next(agent.policy_net.parameters()).device
        t = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32).to(model_dev)
        agent.policy_net.eval()
        with torch.no_grad():
            q = agent.policy_net(t)
        if isinstance(q, tuple):
            q = q[0]
        if isinstance(q, torch.Tensor):
            if q.ndim == 2:
                idx = int(torch.argmax(q, dim=1).item())
                probs = torch.softmax(q, dim=1)
                severity = float(torch.max(probs).item())
            else:
                idx = int(torch.argmax(q).item())
                severity = 0.0
        else:
            arr = np.asarray(q)
            if arr.ndim == 2:
                idx = int(np.argmax(arr, axis=1)[0])
                severity = 0.0
            else:
                idx = int(np.argmax(arr))
                severity = 0.0
        actions = ["allow", "inspect", "block", "alert", "log"]
        action = actions[idx] if idx < len(actions) else "allow"
        return action, float(severity), int(idx)
    except Exception:
        return "allow", 0.0, 0

def simple_attacker_simulator(features):
    arr = np.asarray(features, dtype=np.float32).reshape(-1)
    s = float(arr.sum())
    if s < -1.0:
        return 0
    if s > 1.0:
        return 2
    return 1

def compute_stackelberg_adjusted_reward(def_action_idx: int, severity: float, attacker_eq_action_idx: int, lambda_=0.5):
    a_def = float(def_action_idx)
    a_eq = float(attacker_eq_action_idx)
    try:
        return float(severity) - float(lambda_) * ((a_def - a_eq) ** 2)
    except Exception:
        return float(severity)

def _append_csv_log(row: dict, csv_path=CSV_LOG_PATH):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    header = ["ts", "id", "action", "action_idx", "severity", "adjusted_reward", "features_json", "top_shap", "top_lime", "top_ig"]
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        # rotate large logs
        try:
            if os.path.getsize(csv_path) > 10 * 1024 * 1024:
                base, ext = os.path.splitext(csv_path)
                os.rename(csv_path, f"{base}_{int(time.time())}{ext}")
        except Exception:
            pass
    except Exception:
        print("[!] CSV log error:", traceback.format_exc())

# ----------------------------
# Flask endpoints
# ----------------------------
@APP.route("/ping", methods=["GET"])
def ping():
    meta = {
        "ok": True if AGENT is not None else False,
        "checkpoint": CHECKPOINT,
        "device": DEVICE,
        "input_dim": INPUT_DIM,
        "n_actions": N_ACTIONS,
        "time": time.time()
    }
    return jsonify(meta), (200 if AGENT is not None else 500)

@APP.route("/meta", methods=["GET"])
def meta():
    return jsonify({"input_dim": INPUT_DIM, "n_actions": N_ACTIONS, "device": DEVICE, "checkpoint": CHECKPOINT}), 200

@APP.route("/health", methods=["GET"])
def health():
    return jsonify({
        "service": "forensics_server",
        "checkpoint": CHECKPOINT,
        "device": DEVICE,
        "input_dim": INPUT_DIM,
        "n_actions": N_ACTIONS,
        "attacker_running": bool(ATTACKER_TRAINER.running) if ATTACKER_TRAINER else False,
        "attacker_replay_len": len(ATTACKER_TRAINER.replay) if ATTACKER_TRAINER else 0,
        "timestamp": time.time()
    }), 200

@APP.route("/explain", methods=["POST"])
def explain_endpoint():
    """
    POST payload: {"id": "...", "features": [...], "meta": {...}}
    Returns: forensic decision + explain results + attacker_sim + logging
    """
    try:
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({"status": "error", "error": "Empty request"}), 400

        features = payload.get("features")
        if features is None:
            return jsonify({"status": "error", "error": "Missing 'features' in payload"}), 400

        # support batch or single
        if isinstance(features, list) and len(features) > 0 and isinstance(features[0], list):
            feat = features[0]
        else:
            feat = features

        # ensure correct dim
        model_dim = _infer_input_dim(AGENT)
        feat_np = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        if feat_np.shape[1] != model_dim:
            if feat_np.shape[1] > model_dim:
                feat_np = feat_np[:, :model_dim]
            else:
                pad = np.zeros((1, model_dim - feat_np.shape[1]), dtype=np.float32)
                feat_np = np.concatenate([feat_np, pad], axis=1)

        # defender decision
        action_str, severity, action_idx = _forensic_decision(AGENT, feat_np)

        # attacker equilibrium action
        if ATTACKER_TRAINER and ATTACKER_TRAINER.running:
            attacker_idx = ATTACKER_TRAINER.act(feat_np.flatten())
            # push synthetic transition for training signal
            success = 1.0 if (action_str == "allow" and attacker_idx == 2) else -0.1
            try:
                ATTACKER_TRAINER.push_transition(feat_np.flatten(), int(attacker_idx), float(success), None, False)
            except Exception:
                pass
        else:
            attacker_idx = simple_attacker_simulator(feat_np.flatten().tolist())

        adjusted_reward = compute_stackelberg_adjusted_reward(action_idx, severity, attacker_idx, lambda_=STACKELBERG_LAMBDA)

        # run explainers (may be slow)
        explain_results = explain_all(AGENT, feat_np, background=BACKGROUND_CACHE, device=DEVICE)

        # extract top features for logging best-effort
        top_shap = None
        try:
            shap_m = explain_results.get("shap", {}) or {}
            mean_abs = shap_m.get("mean_abs")
            if mean_abs:
                arr = np.asarray(mean_abs)
                if arr.ndim == 2:
                    avg = arr.mean(axis=0)
                else:
                    avg = arr
                top_idx = list(np.argsort(-np.abs(avg))[:5])
                top_shap = [{"f": int(i), "v": float(avg[i])} for i in top_idx]
        except Exception:
            top_shap = None

        top_lime = None
        try:
            lime_m = explain_results.get("lime", {}) or {}
            lr = lime_m.get("lime_results", [])
            if lr:
                lst = lr[0].get("lime_features", [])
                top_lime = [{"f": int(f), "v": float(v)} for f, v in lst[:8]]
        except Exception:
            top_lime = None

        top_ig = None
        try:
            ig_m = explain_results.get("integrated_gradients", {}) or {}
            ig_vals = ig_m.get("ig_values")
            if ig_vals:
                arr = np.asarray(ig_vals)
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.ndim == 2:
                    col_sum = np.sum(arr, axis=0)
                else:
                    col_sum = arr[0]
                idxs = list(np.argsort(-np.abs(col_sum))[:8])
                top_ig = [{"f": int(i), "v": float(col_sum[i])} for i in idxs]
        except Exception:
            top_ig = None

        response = {
            "status": "ok",
            "forensic_decision": {
                "action": action_str,
                "severity": severity,
                "adjusted_reward": adjusted_reward,
                "record": {
                    "features": feat_np.flatten().tolist(),
                    "id": payload.get("id", "unknown"),
                    "ts": time.time(),
                    "agent_meta": {"checkpoint": CHECKPOINT}
                }
            },
            "explain": explain_results,
            "attacker_sim": {"attacker_action_idx": int(attacker_idx)}
        }

        # CSV log write
        csv_row = {
            "ts": time.time(),
            "id": payload.get("id", "unknown"),
            "action": action_str,
            "action_idx": int(action_idx),
            "severity": float(severity),
            "adjusted_reward": float(adjusted_reward),
            "features_json": json.dumps(feat_np.flatten().tolist()),
            "top_shap": json.dumps(top_shap),
            "top_lime": json.dumps(top_lime),
            "top_ig": json.dumps(top_ig)
        }
        _append_csv_log(csv_row, CSV_LOG_PATH)

        return jsonify(to_native(response)), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500

# Attacker control endpoints
@APP.route("/attacker/train/start", methods=["POST"])
def attacker_train_start():
    global ATTACKER_TRAINER
    try:
        if ATTACKER_TRAINER is None:
            ATTACKER_TRAINER = AttackerTrainer(input_dim=INPUT_DIM, n_actions=3, device=DEVICE, checkpoint_path=ATTACKER_CHECKPOINT)
            # try to load
            try:
                ATTACKER_TRAINER.load(ATTACKER_CHECKPOINT)
            except Exception:
                pass
            # set defender if agent exists
            try:
                ATTACKER_TRAINER.set_defender(AGENT)
            except Exception:
                pass
        started = ATTACKER_TRAINER.start()
        return jsonify({"started": bool(started), "status": ATTACKER_TRAINER.status()}), 200
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@APP.route("/attacker/train/stop", methods=["POST"])
def attacker_train_stop():
    global ATTACKER_TRAINER
    try:
        if ATTACKER_TRAINER is None:
            return jsonify({"stopped": True, "reason": "not running"}), 200
        stopped = ATTACKER_TRAINER.stop()
        ATTACKER_TRAINER.save(ATTACKER_CHECKPOINT)
        return jsonify({"stopped": bool(stopped), "status": ATTACKER_TRAINER.status()}), 200
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@APP.route("/attacker/train/status", methods=["GET"])
def attacker_train_status():
    if ATTACKER_TRAINER is None:
        return jsonify({"running": False}), 200
    return jsonify({"running": bool(ATTACKER_TRAINER.running), "status": ATTACKER_TRAINER.status()}), 200

@APP.route("/mode", methods=["POST"])
def switch_mode():
    global ATTACKER_TRAINER, ATTACKER_CHECKPOINT
    try:
        # Validate payload
        payload = request.get_json(force=True)
        if not payload or not payload.get("mode") or not payload.get("checkpoint"):
            return jsonify({"status": "error", "error": "Missing mode or checkpoint in payload"}), 400
        mode = payload["mode"]
        checkpoint = payload["checkpoint"]
        if mode not in ["live_marl", "co_trained"]:
            return jsonify({"status": "error", "error": f"Invalid mode '{mode}', must be 'live_marl' or 'co_trained'"}), 400

        # Validate checkpoint path
        checkpoint = os.path.abspath(checkpoint)
        if not os.path.exists(checkpoint):
            return jsonify({"status": "error", "error": f"Checkpoint file '{checkpoint}' not found"}), 400
        if not os.access(checkpoint, os.R_OK):
            return jsonify({"status": "error", "error": f"Checkpoint file '{checkpoint}' is not readable"}), 400

        # Validate torch availability
        if torch is None:
            return jsonify({"status": "error", "error": "PyTorch is not available, cannot load checkpoint"}), 500

        # Validate INPUT_DIM
        if INPUT_DIM is None or INPUT_DIM <= 0:
            return jsonify({"status": "error", "error": f"Invalid INPUT_DIM ({INPUT_DIM}), ensure defender is loaded correctly"}), 500

        print(f"[+] Switching to mode '{mode}' with checkpoint '{checkpoint}'")

        # Stop existing trainer if running
        if ATTACKER_TRAINER and ATTACKER_TRAINER.running:
            print("[+] Stopping existing attacker trainer")
            ATTACKER_TRAINER.stop()
            try:
                ATTACKER_TRAINER.save(ATTACKER_CHECKPOINT)
                print(f"[+] Saved current checkpoint to '{ATTACKER_CHECKPOINT}'")
            except Exception as e:
                print(f"[!] Error saving checkpoint '{ATTACKER_CHECKPOINT}': {str(e)}")

        # Update checkpoint path
        ATTACKER_CHECKPOINT = checkpoint

        # Initialize new trainer
        try:
            ATTACKER_TRAINER = AttackerTrainer(
                input_dim=INPUT_DIM,
                n_actions=3,
                device=DEVICE,
                checkpoint_path=ATTACKER_CHECKPOINT
            )
            print(f"[+] Initialized new AttackerTrainer with input_dim={INPUT_DIM}, device={DEVICE}")
        except Exception as e:
            return jsonify({"status": "error", "error": f"Failed to initialize AttackerTrainer: {str(e)}", "trace": traceback.format_exc()}), 500

        # Link defender
        try:
            ATTACKER_TRAINER.set_defender(AGENT)
            print("[+] Successfully linked defender to attacker trainer")
        except Exception as e:
            print(f"[!] Warning: Failed to link defender: {str(e)}")

        # Load checkpoint
        try:
            loaded = ATTACKER_TRAINER.load(ATTACKER_CHECKPOINT)
            if not loaded:
                return jsonify({"status": "error", "error": f"Failed to load checkpoint '{ATTACKER_CHECKPOINT}'"}), 500
            print(f"[+] Successfully loaded checkpoint '{ATTACKER_CHECKPOINT}'")
        except Exception as e:
            return jsonify({"status": "error", "error": f"Checkpoint load failed: {str(e)}", "trace": traceback.format_exc()}), 500

        # Configure mode
        if mode == "live_marl":
            try:
                started = ATTACKER_TRAINER.start()
                if not started:
                    return jsonify({"status": "error", "error": "Failed to start trainer for live_marl mode"}), 500
                print("[+] Started attacker trainer for live_marl mode")
            except Exception as e:
                return jsonify({"status": "error", "error": f"Failed to start live_marl trainer: {str(e)}", "trace": traceback.format_exc()}), 500
        else:  # co_trained
            try:
                ATTACKER_TRAINER.stop()
                print("[+] Ensured attacker trainer is stopped for co_trained mode")
            except Exception as e:
                print(f"[!] Warning: Failed to stop trainer for co_trained mode: {str(e)}")

        return jsonify({
            "status": "ok",
            "mode": mode,
            "checkpoint": ATTACKER_CHECKPOINT,
            "running": ATTACKER_TRAINER.running,
            "input_dim": INPUT_DIM,
            "device": DEVICE
        }), 200

    except Exception as e:
        print(f"[!] Mode switch error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "error": f"Internal server error: {str(e)}", "trace": traceback.format_exc()}), 500

# ----------------------------
# Main bootstrap
# ----------------------------
def main(checkpoint_path, host="127.0.0.1", port=5001, device="cpu", csv_path="results/forensics_log.csv", lambda_stack=0.5):
    global AGENT, CHECKPOINT, DEVICE, INPUT_DIM, N_ACTIONS, BACKGROUND_CACHE, CSV_LOG_PATH, STACKELBERG_LAMBDA, ATTACKER_CHECKPOINT

    DEVICE = "cuda" if device == "cuda" and torch is not None and torch.cuda.is_available() else "cpu"
    CHECKPOINT = checkpoint_path
    CSV_LOG_PATH = csv_path
    STACKELBERG_LAMBDA = float(lambda_stack)
    ATTACKER_CHECKPOINT = os.path.abspath(ATTACKER_CHECKPOINT)

    # instantiate agent (guesses — load_checkpoint will adjust if needed)
    InputDim_guess = 15
    NActions_guess = 5
    AGENT = DQNAgent(input_dim=InputDim_guess, n_actions=NActions_guess, device=DEVICE)

    # load checkpoint
    try:
        load_meta = AGENT.load_checkpoint(checkpoint_path)
        print("[+] Loaded checkpoint:", load_meta)
    except Exception as e:
        print("[!] Failed to load checkpoint:", e)
        raise

    # update input_dim & n_actions
    INPUT_DIM = _infer_input_dim(AGENT)
    try:
        last = AGENT.policy_net.net[-1]
        N_ACTIONS = int(last.out_features)
    except Exception:
        N_ACTIONS = NActions_guess

    # prepare background cache for SHAP/LIME
    BACKGROUND_CACHE = np.random.normal(size=(min(200, max(10, 10)), INPUT_DIM)).astype(np.float32)

    # ensure results directories
    os.makedirs(os.path.dirname(CSV_LOG_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(ATTACKER_CHECKPOINT) or ".", exist_ok=True)

    print(f"[+] Agent ready — input_dim={INPUT_DIM}, n_actions={N_ACTIONS}, device={DEVICE}")
    print(f"[+] Background cache shape: {BACKGROUND_CACHE.shape}")
    print(f"[+] Logging CSV to: {CSV_LOG_PATH}")
    print(f"[+] Attacker checkpoint: {ATTACKER_CHECKPOINT}")
    print(f"[+] Health endpoint: http://{host}:{port}/health")
    print(f"[+] Starting Forensics server on {host}:{port}")
    APP.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5001, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--csv", default="results/forensics_log.csv")
    parser.add_argument("--lambda_stack", default=0.5, type=float)
    args = parser.parse_args()
    main(args.checkpoint, host=args.host, port=args.port, device=args.device, csv_path=args.csv, lambda_stack=args.lambda_stack)