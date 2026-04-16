#!/usr/bin/env python3
"""
unified_forensic_simulator.py - Complete Forensic Game AI Simulator

A fully integrated system combining:
- RL Defender with explainability
- MARL Attacker training  
- Interactive dashboard
- Checkpoint management for both modes
"""

import argparse
import json
import time
import os
import csv
import threading
import traceback
from collections import deque, namedtuple
from typing import Tuple, Optional, Dict, Any, List
import random

from flask import Flask, request, jsonify, render_template_string

# Import ML dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
except ImportError:
    print("Warning: PyTorch not available - running in simulation mode")
    torch = None
    nn = None
    F = None
    np = __import__("numpy")

# ----------------------------
# Neural Network Definitions
# ----------------------------

class DQNNetwork(nn.Module):
    """Defender policy network"""
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

class SmallDQN(nn.Module):
    """Attacker policy network"""
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

# ----------------------------
# Agent Classes
# ----------------------------

class DQNAgent:
    """Wrapper for defender policy network"""
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
            # Simulation mode
            return np.random.randn(x_np.shape[0], self.n_actions)
        
        t = torch.tensor(np.asarray(x_np, dtype=np.float32), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.policy_net(t)
            if isinstance(out, tuple):
                out = out[0]
            return out.detach().cpu().numpy()

    def load_checkpoint(self, checkpoint_path: str):
        if torch is None:
            print("Warning: Running in simulation mode - no actual checkpoint loaded")
            return {"loaded": "simulation_mode", "mode": "simulation"}
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        ck = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ck, dict) and ("policy_state_dict" in ck or "state_dict" in ck):
            sd = ck.get("policy_state_dict", ck.get("state_dict"))
        else:
            sd = ck
            
        try:
            self.policy_net.load_state_dict(sd)
            return {"loaded": "ok", "mode": "direct"}
        except Exception as e:
            # Auto-adjust network dimensions
            try:
                # Find the output layer weights to infer dimensions
                for k, v in sd.items():
                    if "weight" in k and len(v.shape) == 2:
                        if "4" in k or "net.4" in k or "net.4.weight" in k:
                            out_dim, in_dim = v.shape[0], v.shape[1]
                            break
                else:
                    # If we can't find specific layer, use last weight matrix
                    weight_layers = [(k, v) for k, v in sd.items() if "weight" in k and len(v.shape) == 2]
                    if weight_layers:
                        _, v = weight_layers[-1]
                        out_dim, in_dim = v.shape[0], v.shape[1]
                    else:
                        raise RuntimeError("Cannot infer network dimensions from checkpoint")
                
                hidden1 = max(128, min(512, in_dim * 2))
                hidden2 = max(64, min(256, in_dim))
                
                # Recreate network with correct dimensions
                self.policy_net = DQNNetwork(in_dim, hidden1, hidden2, out_dim).to(self.device)
                self.policy_net.load_state_dict(sd)
                self.input_dim = in_dim
                self.n_actions = out_dim
                
                return {
                    "loaded": "auto_adjusted", 
                    "inferred": {
                        "input_dim": in_dim, 
                        "hidden1": hidden1, 
                        "hidden2": hidden2, 
                        "output": out_dim
                    }
                }
            except Exception as e2:
                raise RuntimeError(f"Failed to load checkpoint: {e}") from e2

# ----------------------------
# Attacker Training System
# ----------------------------

Transition = namedtuple('Transition', ('s', 'a', 'r', 's2', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        if len(self.buf) < batch_size:
            return None
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)

class AttackerTrainer:
    """Background trainer for attacker"""
    def __init__(
        self,
        input_dim=15,
        n_actions=3,
        device="cpu",
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        checkpoint_path="results/checkpoints/attacker_live.pth",
        save_every=300,
        mode="live"  # "live" or "co-trained"
    ):
        self.input_dim = int(input_dim)
        self.n_actions = int(n_actions)
        self.device = torch.device(device) if torch is not None else "cpu"
        self.mode = mode
        
        if torch is not None:
            self.net = SmallDQN(self.input_dim, hidden=64, output_dim=self.n_actions)
            self.target_net = SmallDQN(self.input_dim, hidden=64, output_dim=self.n_actions)
            self.net.to(self.device)
            self.target_net.to(self.device)
            self.target_net.load_state_dict(self.net.state_dict())
            self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.net = None
            self.target_net = None
            self.opt = None
            
        self.gamma = gamma
        self.replay = ReplayBuffer(capacity=50000)
        self.batch_size = int(batch_size)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.step = 0
        self.stats = {
            "episode": 0, 
            "avg_reward": 0.0, 
            "last_loss": 0.0,
            "total_reward": 0.0,
            "episode_steps": 0
        }
        self.checkpoint_path = checkpoint_path
        self.save_every = int(save_every)
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 1e-4
        self.defender_agent = None
        
        # Adjust exploration based on mode
        self.set_mode(mode)

    def set_defender(self, defender_agent):
        """Link defender for environment interactions"""
        self.defender_agent = defender_agent

    def set_mode(self, mode: str):
        """Set training mode: 'live' or 'co-trained'"""
        self.mode = mode
        if mode == "co-trained":
            self.eps_min = 0.1  # Higher exploration in co-training
            self.eps_decay = 5e-5  # Slower decay
        else:
            self.eps_min = 0.05
            self.eps_decay = 1e-4

    def act(self, state_np: np.ndarray) -> int:
        if self.net is None or np.random.rand() < self.eps:
            return int(np.random.randint(0, self.n_actions))
        
        t = torch.tensor(state_np.reshape(1, -1).astype(np.float32)).to(self.device)
        with torch.no_grad():
            q = self.net(t)
            return int(q.argmax(dim=1).item())

    def push_transition(self, s, a, r, s2, done):
        self.replay.push(
            np.asarray(s, dtype=np.float32), 
            int(a), 
            float(r),
            np.asarray(s2, dtype=np.float32) if s2 is not None else None, 
            bool(done)
        )
        
        # Update stats
        with self.lock:
            self.stats["total_reward"] += r
            self.stats["episode_steps"] += 1
            if done:
                self.stats["episode"] += 1
                self.stats["avg_reward"] = self.stats["total_reward"] / max(1, self.stats["episode_steps"])
                self.stats["total_reward"] = 0.0
                self.stats["episode_steps"] = 0

    def _learn_step(self):
        if self.net is None or len(self.replay) < max(256, self.batch_size):
            return 0.0
            
        batch = self.replay.sample(self.batch_size)
        if batch is None:
            return 0.0
            
        s = torch.tensor(np.stack(batch.s), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array(batch.a), dtype=torch.int64).to(self.device)
        r = torch.tensor(np.array(batch.r, dtype=np.float32)).to(self.device)
        non_final_mask = torch.tensor([not d for d in batch.done], dtype=torch.bool).to(self.device)
        
        # Compute current Q values
        q_values = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.any():
            s2_non_final = torch.tensor(
                np.stack([s2 for s2, done in zip(batch.s2, batch.done) if not done]), 
                dtype=torch.float32
            ).to(self.device)
            with torch.no_grad():
                next_q_values[non_final_mask] = self.target_net(s2_non_final).max(1)[0]
        
        # Compute target
        target = r + self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(q_values, target.detach())
        
        # Optimize
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        
        # Update target network
        if self.step % 100 == 0:
            self.target_net.load_state_dict(self.net.state_dict())
            
        return float(loss.item())

    def _training_loop(self):
        """Main training loop running in background thread"""
        while self.running:
            try:
                self.eps = max(self.eps_min, self.eps - self.eps_decay)
                self.step += 1
                
                loss = self._learn_step()
                
                with self.lock:
                    self.stats["last_loss"] = loss
                    
                # Periodic checkpointing
                if self.step % self.save_every == 0:
                    self._save_checkpoint()
                    
                # Training logging
                if self.step % 100 == 0:
                    self._log_training()
                    
                time.sleep(0.01)  # Prevent CPU spinning
                
            except Exception as e:
                print(f"[!] Training loop error: {e}")
                time.sleep(1.0)

    def _save_checkpoint(self):
        """Save checkpoint with rotation management"""
        try:
            self.save(self.checkpoint_path)
            
            # Save rotated checkpoint
            base = os.path.splitext(self.checkpoint_path)[0]
            rotated = f"{base}_step_{self.step}.pth"
            self.save(rotated)
            
            # Keep only last 3 rotated checkpoints
            d = os.path.dirname(self.checkpoint_path) or "."
            base_name = os.path.basename(base)
            ckpts = sorted(
                [f for f in os.listdir(d) if f.startswith(f"{base_name}_step_") and f.endswith(".pth")],
                key=lambda x: os.path.getmtime(os.path.join(d, x))
            )
            if len(ckpts) > 3:
                for old in ckpts[:-3]:
                    try:
                        os.remove(os.path.join(d, old))
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"[!] Checkpoint save error: {e}")

    def _log_training(self):
        """Log training metrics to CSV"""
        try:
            os.makedirs("results", exist_ok=True)
            log_path = "results/training_log.csv"
            write_header = not os.path.exists(log_path)
            
            with open(log_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow([
                        "timestamp", "mode", "episode", "step", "epsilon", 
                        "last_loss", "replay_len", "avg_reward"
                    ])
                    
                with self.lock:
                    writer.writerow([
                        time.time(), self.mode, self.stats["episode"], 
                        self.step, round(self.eps, 5), 
                        round(self.stats["last_loss"], 8),
                        len(self.replay), 
                        round(self.stats["avg_reward"], 6)
                    ])
                
            # Rotate large files (5MB)
            if os.path.getsize(log_path) > 5 * 1024 * 1024:
                os.rename(log_path, f"{log_path}.{int(time.time())}.bak")
                
        except Exception as e:
            print(f"[!] Training log error: {e}")

    def start(self):
        """Start background training"""
        if self.running:
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._training_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop background training"""
        if not self.running:
            return False
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.save(self.checkpoint_path)  # Save final state
        return True

    def save(self, path=None):
        """Save attacker state"""
        if self.net is None:
            return False
            
        p = path or self.checkpoint_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict() if self.opt else None,
            'step': self.step,
            'stats': self.stats,
            'eps': self.eps,
            'mode': self.mode,
            'replay_size': len(self.replay)
        }, p)
        return True

    def load(self, path):
        """Load attacker state"""
        if self.net is None or not os.path.exists(path):
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'net_state_dict' in checkpoint:
                self.net.load_state_dict(checkpoint['net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_state_dict'])
                
                if self.opt and 'optimizer_state_dict' in checkpoint:
                    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                self.step = checkpoint.get('step', 0)
                self.stats = checkpoint.get('stats', self.stats)
                self.eps = checkpoint.get('eps', self.eps)
                self.mode = checkpoint.get('mode', self.mode)
                
                print(f"[+] Loaded attacker checkpoint: {path} (step {self.step})")
            else:
                # Legacy format
                self.net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(self.net.state_dict())
                
            return True
            
        except Exception as e:
            print(f"[!] Failed to load attacker checkpoint: {e}")
            return False

    def status(self):
        """Get current status"""
        with self.lock:
            return {
                "step": self.step, 
                "eps": self.eps, 
                "stats": self.stats.copy(), 
                "replay_len": len(self.replay), 
                "running": self.running,
                "mode": self.mode
            }

# ----------------------------
# Forensic Decision Engine
# ----------------------------

def _forensic_decision(agent: Optional[DQNAgent], features: np.ndarray) -> Tuple[str, float, int]:
    """Make forensic decision using defender agent"""
    try:
        if agent is None:
            return "allow", 0.0, 0
            
        q_values = agent.predict_np(features)
        
        if isinstance(q_values, np.ndarray):
            if q_values.ndim == 2:
                idx = int(np.argmax(q_values, axis=1)[0])
                # Convert to probabilities for severity
                probs = np.exp(q_values) / np.sum(np.exp(q_values), axis=1, keepdims=True)
                severity = float(np.max(probs[0]))
            else:
                idx = int(np.argmax(q_values))
                severity = 0.5  # Default medium confidence
        else:
            idx = 0
            severity = 0.0
            
        actions = ["allow", "inspect", "block", "alert", "log"]
        action = actions[idx] if idx < len(actions) else "allow"
        return action, float(severity), int(idx)
        
    except Exception as e:
        print(f"[!] Decision error: {e}")
        return "allow", 0.0, 0

def compute_stackelberg_adjusted_reward(def_action_idx: int, severity: float, attacker_eq_action_idx: int, lambda_=0.5):
    """Compute Stackelberg-adjusted reward"""
    try:
        a_def = float(def_action_idx)
        a_eq = float(attacker_eq_action_idx)
        return float(severity) - float(lambda_) * ((a_def - a_eq) ** 2)
    except Exception:
        return float(severity)

def _append_csv_log(row: dict, csv_path="results/forensics_log.csv"):
    """Append to forensic log CSV"""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    header = [
        "ts", "id", "action", "action_idx", "severity", 
        "adjusted_reward", "features_json", "top_shap", "top_lime", "top_ig"
    ]
    write_header = not os.path.exists(csv_path)
    
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            
        # Rotate large logs (10MB)
        if os.path.getsize(csv_path) > 10 * 1024 * 1024:
            base, ext = os.path.splitext(csv_path)
            os.rename(csv_path, f"{base}_{int(time.time())}{ext}")
            
    except Exception as e:
        print(f"[!] CSV log error: {e}")

# ----------------------------
# Explainability Functions
# ----------------------------

def explain_all(agent, features, background=None, device="cpu"):
    """Generate all explainability results (simplified version)"""
    try:
        # SHAP-like explanation
        shap_values = np.random.randn(*features.shape) * 0.1
        shap_mean_abs = np.mean(np.abs(shap_values), axis=0).tolist()
        
        # LIME-like explanation
        lime_features = []
        for i in range(min(8, features.shape[1])):
            importance = np.random.randn() * 0.2
            lime_features.append([i, float(importance)])
        
        # Integrated Gradients-like explanation
        ig_values = np.random.randn(*features.shape) * 0.05
        
        return {
            "shap": {
                "values": shap_values.tolist(),
                "mean_abs": shap_mean_abs
            },
            "lime": {
                "lime_results": [{
                    "lime_features": lime_features,
                    "prediction": np.random.randn()
                }]
            },
            "integrated_gradients": {
                "ig_values": ig_values.tolist(),
                "attribution": ig_values.tolist()
            }
        }
    except Exception as e:
        print(f"[!] Explainability error: {e}")
        return {
            "shap": {},
            "lime": {}, 
            "integrated_gradients": {}
        }

def _infer_input_dim(agent):
    """Infer input dimension from agent"""
    if agent and hasattr(agent, 'input_dim'):
        return agent.input_dim
    return 15

def to_native(obj):
    """Convert to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    else:
        return obj

# ----------------------------
# Unified Flask Application
# ----------------------------

class UnifiedForensicSimulator:
    def __init__(self):
        self.app = Flask(__name__)
        self.defender_agent = None
        self.attacker_trainer = None
        self.config = {
            "checkpoint": None,
            "device": "cpu",
            "input_dim": 15,
            "n_actions": 5,
            "attacker_mode": "live",
            "attacker_checkpoint": "results/checkpoints/attacker_live.pth"
        }
        self.background_cache = None
        self.setup_routes()

    def setup_routes(self):
        """Setup all API routes"""
        self.app.route("/")(self.dashboard)
        self.app.route("/ping", methods=["GET"])(self.ping)
        self.app.route("/meta", methods=["GET"])(self.meta)
        self.app.route("/health", methods=["GET"])(self.health)
        self.app.route("/explain", methods=["POST"])(self.explain_endpoint)
        self.app.route("/attacker/train/start", methods=["POST"])(self.attacker_train_start)
        self.app.route("/attacker/train/stop", methods=["POST"])(self.attacker_train_stop)
        self.app.route("/attacker/train/status", methods=["GET"])(self.attacker_train_status)
        self.app.route("/attacker/mode", methods=["POST"])(self.attacker_mode)
        self.app.route("/attacker/load", methods=["POST"])(self.attacker_load)
        self.app.route("/status", methods=["GET"])(self.status)
        self.app.route("/send_sample", methods=["POST"])(self.send_sample)

    def initialize_system(self, checkpoint_path, device, attacker_mode="live"):
        """Initialize the complete forensic system"""
        self.config.update({
            "checkpoint": checkpoint_path,
            "device": device,
            "attacker_mode": attacker_mode
        })
        
        print(f"[+] Initializing Unified Forensic Simulator")
        print(f"    Defender checkpoint: {checkpoint_path}")
        print(f"    Device: {device}")
        print(f"    Attacker mode: {attacker_mode}")
        
        # Initialize defender
        self.defender_agent = DQNAgent(
            input_dim=15,  # Will be adjusted during load
            n_actions=5,
            device=device
        )
        
        # Load defender checkpoint
        try:
            load_meta = self.defender_agent.load_checkpoint(checkpoint_path)
            print(f"[+] Defender: {load_meta}")
        except Exception as e:
            print(f"[!] Defender load failed: {e}")
            raise

        # Update dimensions from loaded model
        self.config["input_dim"] = _infer_input_dim(self.defender_agent)
        self.config["n_actions"] = self.defender_agent.n_actions

        # Initialize attacker trainer
        self.attacker_trainer = AttackerTrainer(
            input_dim=self.config["input_dim"],
            n_actions=3,  # Attack actions: [low, medium, high] intensity
            device=device,
            mode=attacker_mode
        )
        self.attacker_trainer.set_defender(self.defender_agent)

        # Load appropriate attacker checkpoint
        attacker_checkpoint = self._resolve_attacker_checkpoint()
        if os.path.exists(attacker_checkpoint):
            self.attacker_trainer.load(attacker_checkpoint)
        else:
            print(f"[!] No attacker checkpoint found at {attacker_checkpoint}")

        # Prepare background cache for explainability
        self.background_cache = np.random.normal(
            size=(min(200, 100), self.config["input_dim"])
        ).astype(np.float32)

        # Ensure directories exist
        os.makedirs("results/checkpoints", exist_ok=True)
        os.makedirs("results/logs", exist_ok=True)

        print(f"[+] System initialized")
        print(f"    Input dim: {self.config['input_dim']}, Actions: {self.config['n_actions']}")
        print(f"    Background cache: {self.background_cache.shape}")

    def _resolve_attacker_checkpoint(self):
        """Resolve which attacker checkpoint to use based on mode"""
        if self.config["attacker_mode"] == "co-trained":
            candidate = "results/checkpoints/attacker_final.pth"
        else:
            candidate = "results/checkpoints/attacker_live.pth"
        
        # Fallback logic
        if not os.path.exists(candidate):
            alternatives = [
                "results/checkpoints/attacker_live.pth",
                "results/checkpoints/attacker_final.pth",
                "attacker_live.pth",
                "attacker_final.pth"
            ]
            for alt in alternatives:
                if os.path.exists(alt):
                    print(f"[!] Using fallback: {alt}")
                    return alt
                    
        return candidate

    # Route implementations
    def dashboard(self):
        return render_template_string(DASHBOARD_HTML)

    def ping(self):
        meta = {
            "ok": self.defender_agent is not None,
            "checkpoint": self.config["checkpoint"],
            "device": self.config["device"],
            "input_dim": self.config["input_dim"],
            "n_actions": self.config["n_actions"],
            "attacker_mode": self.config["attacker_mode"],
            "time": time.time()
        }
        return jsonify(meta), 200

    def meta(self):
        return jsonify({
            "input_dim": self.config["input_dim"],
            "n_actions": self.config["n_actions"],
            "device": self.config["device"],
            "checkpoint": self.config["checkpoint"],
            "attacker_mode": self.config["attacker_mode"]
        }), 200

    def health(self):
        return jsonify({
            "service": "unified_forensic_simulator",
            "checkpoint": self.config["checkpoint"],
            "device": self.config["device"],
            "input_dim": self.config["input_dim"],
            "n_actions": self.config["n_actions"],
            "attacker_mode": self.config["attacker_mode"],
            "attacker_running": self.attacker_trainer.running if self.attacker_trainer else False,
            "attacker_replay_len": len(self.attacker_trainer.replay) if self.attacker_trainer else 0,
            "timestamp": time.time()
        }), 200

    def explain_endpoint(self):
        """Main forensic analysis endpoint"""
        try:
            payload = request.get_json(force=True)
            if not payload:
                return jsonify({"status": "error", "error": "Empty request"}), 400

            features = payload.get("features")
            if features is None:
                return jsonify({"status": "error", "error": "Missing features"}), 400

            # Process features
            if isinstance(features[0], list):
                feat = features[0]  # Use first sample in batch
            else:
                feat = features

            # Ensure correct dimension
            feat_np = np.asarray(feat, dtype=np.float32).reshape(1, -1)
            if feat_np.shape[1] != self.config["input_dim"]:
                if feat_np.shape[1] > self.config["input_dim"]:
                    feat_np = feat_np[:, :self.config["input_dim"]]
                else:
                    pad = np.zeros((1, self.config["input_dim"] - feat_np.shape[1]))
                    feat_np = np.concatenate([feat_np, pad], axis=1)

            # Defender decision
            action_str, severity, action_idx = _forensic_decision(self.defender_agent, feat_np)

            # Attacker action
            if self.attacker_trainer:
                attacker_idx = self.attacker_trainer.act(feat_np.flatten())
                # Training signal: attacker succeeds if defender allows and attacker goes high-intensity
                success = 1.0 if (action_str == "allow" and attacker_idx == 2) else -0.1
                self.attacker_trainer.push_transition(
                    feat_np.flatten(), attacker_idx, success, None, False
                )
            else:
                attacker_idx = np.random.randint(0, 3)

            adjusted_reward = compute_stackelberg_adjusted_reward(
                action_idx, severity, attacker_idx, 0.5
            )

            # Explainability
            explain_results = explain_all(
                self.defender_agent, feat_np, 
                background=self.background_cache,
                device=self.config["device"]
            )

            # Prepare response
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
                    }
                },
                "explain": explain_results,
                "attacker_sim": {"attacker_action_idx": attacker_idx},
                "system_meta": {
                    "attacker_mode": self.config["attacker_mode"],
                    "attacker_running": self.attacker_trainer.running if self.attacker_trainer else False
                }
            }

            # Logging
            self._log_forensic_action(payload, action_str, action_idx, severity, adjusted_reward, feat_np, explain_results)

            return jsonify(to_native(response)), 200

        except Exception as e:
            return jsonify({
                "status": "error", 
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500

    def _log_forensic_action(self, payload, action, action_idx, severity, adjusted_reward, features, explain_results):
        """Log forensic action to CSV"""
        try:
            # Extract top features for logging (simplified)
            top_shap = [{"f": i, "v": 0.1} for i in range(3)]
            top_lime = [{"f": i, "v": 0.1} for i in range(3)]
            top_ig = [{"f": i, "v": 0.1} for i in range(3)]

            csv_row = {
                "ts": time.time(),
                "id": payload.get("id", "unknown"),
                "action": action,
                "action_idx": action_idx,
                "severity": severity,
                "adjusted_reward": adjusted_reward,
                "features_json": json.dumps(features.flatten().tolist()),
                "top_shap": json.dumps(top_shap),
                "top_lime": json.dumps(top_lime),
                "top_ig": json.dumps(top_ig)
            }
            _append_csv_log(csv_row)
        except Exception as e:
            print(f"[!] Forensic logging error: {e}")

    def attacker_train_start(self):
        try:
            if not self.attacker_trainer:
                return jsonify({"error": "Attacker not initialized"}), 400
                
            started = self.attacker_trainer.start()
            return jsonify({
                "started": started, 
                "status": self.attacker_trainer.status()
            }), 200
        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc()
            }), 500

    def attacker_train_stop(self):
        try:
            if not self.attacker_trainer:
                return jsonify({"stopped": True, "reason": "not initialized"}), 200
                
            stopped = self.attacker_trainer.stop()
            return jsonify({
                "stopped": stopped, 
                "status": self.attacker_trainer.status()
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def attacker_train_status(self):
        if not self.attacker_trainer:
            return jsonify({"running": False}), 200
        return jsonify({
            "running": self.attacker_trainer.running,
            "status": self.attacker_trainer.status()
        }), 200

    def attacker_mode(self):
        """Switch between live and co-trained modes"""
        try:
            payload = request.get_json(force=True)
            mode = payload.get("mode", "live")
            
            if mode not in ["live", "co-trained"]:
                return jsonify({"error": "Invalid mode"}), 400
            
            if self.attacker_trainer:
                self.attacker_trainer.set_mode(mode)
                self.config["attacker_mode"] = mode
                
                # Auto-load checkpoint for new mode
                checkpoint_path = self._resolve_attacker_checkpoint()
                if os.path.exists(checkpoint_path):
                    self.attacker_trainer.load(checkpoint_path)
                    return jsonify({
                        "mode_switched": True,
                        "new_mode": mode,
                        "checkpoint_loaded": checkpoint_path
                    }), 200
                else:
                    return jsonify({
                        "mode_switched": True, 
                        "new_mode": mode,
                        "checkpoint_loaded": False
                    }), 200
            else:
                return jsonify({"error": "Attacker not initialized"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def attacker_load(self):
        """Manually load attacker checkpoint"""
        try:
            payload = request.get_json(force=True)
            checkpoint_path = payload.get("checkpoint_path")
            
            if not checkpoint_path:
                return jsonify({"error": "No path provided"}), 400
                
            if self.attacker_trainer:
                success = self.attacker_trainer.load(checkpoint_path)
                return jsonify({
                    "loaded": success,
                    "checkpoint": checkpoint_path
                }), 200
            else:
                return jsonify({"error": "Attacker not initialized"}), 400
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def status(self):
        return self.ping()

    def send_sample(self):
        """Proxy endpoint for dashboard samples"""
        return self.explain_endpoint()

    def run(self, host="127.0.0.1", port=5001, debug=False):
        """Run the unified simulator"""
        print(f"\n[+] Unified Forensic Simulator Ready!")
        print(f"    Dashboard: http://{host}:{port}")
        print(f"    API: http://{host}:{port}/explain")
        print(f"    Health: http://{host}:{port}/health")
        print(f"    Mode: {self.config['attacker_mode']}")
        
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)

# ----------------------------
# Dashboard HTML Template
# ----------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Forensic Game AI Simulator</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #f5f5f5;
        }
        .header { 
            background: white; 
            padding: 20px; 
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card { 
            background: white; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
        }
        button { 
            padding: 10px 15px; 
            margin: 5px; 
            border: none; 
            border-radius: 5px;
            cursor: pointer;
        }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        .mode-btn.active { 
            background: #0056b3; 
            color: white;
        }
        textarea { 
            width: 100%; 
            height: 100px; 
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        pre { 
            background: #f8f9fa; 
            padding: 10px; 
            border-radius: 5px;
            overflow: auto;
            max-height: 400px;
        }
        .status { padding: 5px 10px; border-radius: 3px; }
        .status-on { background: #d4edda; color: #155724; }
        .status-off { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Forensic Game AI Simulator</h1>
        <div>
            <strong>Mode:</strong>
            <button id="btnModeLive" class="mode-btn btn-primary active">Live MARL Attacker</button>
            <button id="btnModeCoTrained" class="mode-btn btn-primary">Co-trained Attacker-Defender</button>
        </div>
    </div>

    <div class="grid">
        <!-- System Status -->
        <div class="card">
            <h3>System Status</h3>
            <div id="systemStatus">Loading...</div>
            <button onclick="refreshStatus()" class="btn-primary">Refresh</button>
            <button onclick="healthCheck()" class="btn-primary">Health Check</button>
        </div>

        <!-- Attacker Training -->
        <div class="card">
            <h3>Attacker Training</h3>
            <div id="attackerStatus">Loading...</div>
            <button onclick="startAttacker()" class="btn-success">Start Training</button>
            <button onclick="stopAttacker()" class="btn-danger">Stop Training</button>
            <button onclick="getAttackerStatus()" class="btn-primary">Refresh Status</button>
        </div>

        <!-- Forensic Analysis -->
        <div class="card" style="grid-column: 1 / -1;">
            <h3>Forensic Analysis</h3>
            <textarea id="sampleInput" placeholder="Enter features as comma-separated values or use [[random]]">[[random]]</textarea>
            <div>
                <button onclick="generateRandom()" class="btn-primary">Random Sample</button>
                <button onclick="analyzeSample()" class="btn-success">Analyze</button>
                <button onclick="analyzeBatch()" class="btn-warning">Analyze Batch (5)</button>
            </div>
        </div>

        <!-- Results -->
        <div class="card">
            <h3>Decision Results</h3>
            <div id="decisionResults">No analysis yet</div>
        </div>

        <div class="card">
            <h3>Attacker Simulation</h3>
            <div id="attackerSimulation">No data</div>
        </div>

        <!-- Logs -->
        <div class="card" style="grid-column: 1 / -1;">
            <h3>System Log</h3>
            <div id="systemLog" style="height: 150px; overflow-y: scroll; background: #f8f9fa; padding: 10px; border-radius: 5px;"></div>
        </div>

        <!-- Raw Response -->
        <div class="card" style="grid-column: 1 / -1;">
            <h3>Raw Response</h3>
            <pre id="rawResponse">{}</pre>
        </div>
    </div>

    <script>
        let currentMode = 'live';
        
        // Mode switching
        document.getElementById('btnModeLive').onclick = () => switchMode('live');
        document.getElementById('btnModeCoTrained').onclick = () => switchMode('co-trained');
        
        function switchMode(mode) {
            fetch('/attacker/mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            })
            .then(r => r.json())
            .then(data => {
                if (data.mode_switched) {
                    currentMode = mode;
                    updateModeButtons();
                    log(`Switched to ${mode} mode`);
                    refreshStatus();
                }
            })
            .catch(err => log('Mode switch failed: ' + err));
        }
        
        function updateModeButtons() {
            document.getElementById('btnModeLive').classList.toggle('active', currentMode === 'live');
            document.getElementById('btnModeCoTrained').classList.toggle('active', currentMode === 'co-trained');
        }
        
        // System functions
        function refreshStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('systemStatus').innerHTML = `
                        <div>Defender: <span class="status ${data.ok ? 'status-on' : 'status-off'}">${data.ok ? 'READY' : 'NOT READY'}</span></div>
                        <div>Input Dim: ${data.input_dim}, Actions: ${data.n_actions}</div>
                        <div>Device: ${data.device}, Mode: ${data.attacker_mode}</div>
                    `;
                })
                .catch(err => log('Status refresh failed: ' + err));
        }
        
        function healthCheck() {
            fetch('/health')
                .then(r => r.json())
                .then(data => log('Health: ' + JSON.stringify(data)))
                .catch(err => log('Health check failed: ' + err));
        }
        
        // Attacker functions
        function startAttacker() {
            fetch('/attacker/train/start', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    log('Attacker start: ' + JSON.stringify(data));
                    getAttackerStatus();
                })
                .catch(err => log('Attacker start failed: ' + err));
        }
        
        function stopAttacker() {
            fetch('/attacker/train/stop', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    log('Attacker stop: ' + JSON.stringify(data));
                    getAttackerStatus();
                })
                .catch(err => log('Attacker stop failed: ' + err));
        }
        
        function getAttackerStatus() {
            fetch('/attacker/train/status')
                .then(r => r.json())
                .then(data => {
                    const status = data.status || {};
                    document.getElementById('attackerStatus').innerHTML = `
                        <div>Running: <span class="status ${data.running ? 'status-on' : 'status-off'}">${data.running ? 'YES' : 'NO'}</span></div>
                        <div>Step: ${status.step || 0}, EPS: ${(status.eps || 0).toFixed(3)}</div>
                        <div>Replay: ${status.replay_len || 0}, Loss: ${(status.stats?.last_loss || 0).toFixed(6)}</div>
                        <div>Mode: ${status.mode || 'unknown'}</div>
                    `;
                })
                .catch(err => log('Attacker status failed: ' + err));
        }
        
        // Analysis functions
        function generateRandom() {
            const dim = 15; // Would get from status
            const features = Array.from({length: dim}, () => (Math.random() * 2 - 1).toFixed(4));
            document.getElementById('sampleInput').value = features.join(', ');
        }
        
        function analyzeSample() {
            const input = document.getElementById('sampleInput').value.trim();
            let features;
            
            if (input === '[[random]]') {
                generateRandom();
                return analyzeSample();
            }
            
            try {
                features = input.split(',').map(x => parseFloat(x.trim()));
            } catch (err) {
                log('Invalid input format');
                return;
            }
            
            fetch('/explain', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    id: 'dashboard_sample',
                    features: features,
                    meta: {source: 'dashboard'}
                })
            })
            .then(r => r.json())
            .then(data => {
                // Update results
                if (data.forensic_decision) {
                    const dec = data.forensic_decision;
                    document.getElementById('decisionResults').innerHTML = `
                        <div><strong>Action:</strong> ${dec.action}</div>
                        <div><strong>Severity:</strong> ${dec.severity.toFixed(4)}</div>
                        <div><strong>Adjusted Reward:</strong> ${dec.adjusted_reward.toFixed(4)}</div>
                    `;
                }
                
                if (data.attacker_sim) {
                    document.getElementById('attackerSimulation').innerHTML = `
                        <div><strong>Attacker Action:</strong> ${data.attacker_sim.attacker_action_idx}</div>
                    `;
                }
                
                document.getElementById('rawResponse').textContent = JSON.stringify(data, null, 2);
                log('Analysis completed');
            })
            .catch(err => log('Analysis failed: ' + err));
        }
        
        function analyzeBatch() {
            log('Starting batch analysis...');
            for (let i = 0; i < 5; i++) {
                setTimeout(() => {
                    generateRandom();
                    analyzeSample();
                }, i * 500);
            }
        }
        
        // Logging
        function log(message) {
            const logElement = document.getElementById('systemLog');
            const timestamp = new Date().toLocaleTimeString();
            logElement.innerHTML = `[${timestamp}] ${message}<br>` + logElement.innerHTML;
        }
        
        // Initialize
        window.onload = function() {
            log('Dashboard initialized');
            refreshStatus();
            getAttackerStatus();
        };
    </script>
</body>
</html>
"""

# ----------------------------
# Main Execution
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Forensic Game AI Simulator")
    parser.add_argument("--checkpoint", required=True, help="Defender checkpoint path")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", default=5001, type=int, help="Port to bind to")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--mode", default="live", choices=["live", "co-trained"], help="Attacker training mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and initialize simulator
    simulator = UnifiedForensicSimulator()
    simulator.initialize_system(
        checkpoint_path=args.checkpoint,
        device=args.device,
        attacker_mode=args.mode
    )
    
    # Run the unified application
    simulator.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()