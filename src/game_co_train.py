#!/usr/bin/env python3
"""
src/game_co_train.py

Phase 10 — Co-training: RL Defender vs RL Attacker (both DQN)

Usage:
    python -m src.game_co_train --epochs 500 --steps_per_epoch 200 --device cuda

Features:
 - Loads defender baseline rewards dynamically from results/metrics.csv (if available)
 - Co-trains two DQN agents (defender & attacker) adversarially
 - Saves checkpoints, plots returns & defender value evolution
"""

import os, sys, time, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ensure imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import DQNAgent
from src.game_theory import build_payoff_matrix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPTS = RESULTS / "checkpoints"
PLOTS = RESULTS / "plots"
RESULTS.mkdir(exist_ok=True)
CKPTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utility: load defender baseline rewards dynamically
# ---------------------------------------------------------------------
def load_baseline_rewards():
    metrics_csv = RESULTS / "metrics.csv"
    rewards = {}
    if metrics_csv.exists():
        sums, counts = {}, {}
        with open(metrics_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row.get("split", "").lower().strip()
                try:
                    val = float(row.get("episode_reward", 0))
                except ValueError:
                    continue
                sums[split] = sums.get(split, 0.0) + val
                counts[split] = counts.get(split, 0) + 1
        for k in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            if k in sums:
                rewards[k] = sums[k] / max(1, counts[k])
    if not rewards:
        # fallback static pattern (normalized)
        rewards = {d: 0.98 + 0.01 * (i % 5) for i, d in enumerate(["monday","tuesday","wednesday","thursday","friday"])}
    return rewards


# ---------------------------------------------------------------------
# State & environment helpers
# ---------------------------------------------------------------------
def build_state(def_rewards, last_att_idx, last_def_idx):
    """State vector: [defender_rewards (n), last_attacker_onehot (n), last_defender_onehot (n)]"""
    n = len(def_rewards)
    s = np.zeros(3 * n, dtype=np.float32)
    s[:n] = np.array([def_rewards[d] for d in def_rewards], dtype=np.float32)
    if last_att_idx is not None:
        s[n + last_att_idx] = 1.0
    if last_def_idx is not None:
        s[2 * n + last_def_idx] = 1.0
    return s


def step_dynamics(defender_rewards, def_action, att_action, domains):
    """
    Apply dynamics:
      - defender chosen domain gets slight boost
      - attacked domain gets slight penalty
      - others slowly recover
      - small gaussian noise
    Returns new defender_rewards (dict)
    """
    new = defender_rewards.copy()
    for d in new:
        if d == domains[def_action]:
            new[d] = min(new[d] * 1.015 + 0.001, 1.0)
        elif d == domains[att_action]:
            new[d] = max(new[d] * 0.96 - 0.001, 0.01)
        else:
            new[d] = min(new[d] * 1.005 + 0.0005, 1.0)
        # small noise
        new[d] += np.random.normal(0, 0.002)
        new[d] = float(np.clip(new[d], 0.01, 1.0))
    return new


# ---------------------------------------------------------------------
# Main co-training loop
# ---------------------------------------------------------------------
def co_train(epochs=300, steps_per_epoch=200, device=None,
             defender_lr=1e-4, attacker_lr=1e-4, seed=42,
             save_every=100):

    np.random.seed(seed)
    domains = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    n = len(domains)

    # --- Load initial defender rewards ---
    defender_rewards = load_baseline_rewards()
    print("[✓] Initial defender rewards:")
    for k, v in defender_rewards.items():
        print(f"   {k.capitalize():10s}: {v:.4f}")
    print()

    # --- Initialize DQN agents (defender + attacker) ---
    state_dim = 3 * n
    action_dim = n

    defender = DQNAgent(input_dim=state_dim, n_actions=action_dim, device=device,
                        lr=defender_lr, batch_size=128, buffer_size=100_000, target_sync_every=500)
    attacker = DQNAgent(input_dim=state_dim, n_actions=action_dim, device=device,
                        lr=attacker_lr, batch_size=128, buffer_size=100_000, target_sync_every=500)

    history = {"epoch": [], "def_v": [], "def_return": [], "att_return": []}
    last_att_idx = None
    last_def_idx = None

    for ep in range(1, epochs + 1):
        ep_def_total = 0.0
        ep_att_total = 0.0

        for step in range(steps_per_epoch):
            state = build_state(defender_rewards, last_att_idx, last_def_idx)
            def_action = defender.act(state)
            att_action = attacker.act(state)

            _, M = build_payoff_matrix(defender_rewards)
            def_payoff = float(M[def_action, att_action])
            att_reward = -def_payoff

            ep_def_total += def_payoff
            ep_att_total += att_reward

            next_def_rewards = step_dynamics(defender_rewards, def_action, att_action, domains)
            next_state = build_state(next_def_rewards, att_action, def_action)

            defender.push_transition(state, def_action, def_payoff, next_state, False)
            attacker.push_transition(state, att_action, att_reward, next_state, False)
            defender.update()
            attacker.update()

            defender_rewards = next_def_rewards
            last_att_idx = att_action
            last_def_idx = def_action

        # Epoch statistics
        avg_def_return = ep_def_total / steps_per_epoch
        avg_att_return = ep_att_total / steps_per_epoch
        _, M_final = build_payoff_matrix(defender_rewards)
        min_over_att = M_final.min(axis=1)
        v = float(np.max(min_over_att))

        history["epoch"].append(ep)
        history["def_v"].append(v)
        history["def_return"].append(avg_def_return)
        history["att_return"].append(avg_att_return)

        if ep % 10 == 0:
            print(f"Epoch {ep:04d} | v={v:.4f} | def_ret={avg_def_return:.4f} | att_ret={avg_att_return:.4f} | eps_def={defender.epsilon():.4f} | eps_att={attacker.epsilon():.4f}")

        if ep % save_every == 0:
            defender.save(str(CKPTS / f"defender_ep{ep}.pth"))
            attacker.save(str(CKPTS / f"attacker_ep{ep}.pth"))

    # --- Final saves + plots ---
    defender.save(str(CKPTS / "defender_final.pth"))
    attacker.save(str(CKPTS / "attacker_final.pth"))

    plt.figure(figsize=(6,3))
    plt.plot(history["epoch"], history["def_return"], label="Defender Return")
    plt.plot(history["epoch"], history["att_return"], label="Attacker Return")
    plt.legend(); plt.title("Co-training: Average Returns per Epoch"); plt.tight_layout()
    plt.savefig(PLOTS / "co_train_returns.png"); plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(history["epoch"], history["def_v"])
    plt.title("Defender Conservative Guaranteed Value (max-min)")
    plt.tight_layout()
    plt.savefig(PLOTS / "co_train_def_v.png"); plt.close()

    print("[✓] Co-training complete. Models and plots saved under results/")
    return history


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_every", type=int, default=100)
    args = ap.parse_args()

    co_train(epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
             device=args.device, save_every=args.save_every)
