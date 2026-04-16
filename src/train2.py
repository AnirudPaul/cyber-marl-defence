#!/usr/bin/env python3
"""
Multi-day DQN Training Script (Monday–Friday)

Usage:
    python -m src.train2 --episodes 10 --episode_len 2000 --device auto

Trains sequentially on all weekday splits:
    monday, tuesday, wednesday, thursday, friday

Each day's results are logged to results/metrics.csv
and checkpoints saved under results/checkpoints/
"""

import argparse
import numpy as np
import time
import csv
from pathlib import Path
from src.environment import NetworkFlowEnv
from src.agent import DQNAgent

# --------------------------
# Default directories
# --------------------------
ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = RESULTS_DIR / "metrics.csv"

# --------------------------
# Utilities
# --------------------------
def load_split(split_name: str):
    path = SPLIT_DIR / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    d = np.load(path, allow_pickle=True)
    return d['X'], d['y']

def append_metrics(row: dict):
    header = [
        'timestamp', 'split', 'episode', 'steps',
        'episode_reward', 'avg_reward_per_step',
        'tp', 'tn', 'fp', 'fn', 'epsilon'
    ]
    file_exists = METRICS_CSV.exists()
    with open(METRICS_CSV, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# --------------------------
# Main training routine
# --------------------------
def train_all_days(device=None,
                   episodes=10,
                   episode_len=2000,
                   lr=1e-4,
                   batch_size=128,
                   gamma=0.99,
                   replay_capacity=1_000_000,   # upgraded replay buffer size
                   target_update_steps=1000):

    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]

    # Auto device selection
    import torch
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[✓] Using device: {device}")

    agent = None

    for split in days:
        print(f"\n{'='*70}")
        print(f"[🚀] Training on split: {split.upper()}")
        print(f"{'='*70}\n")

        X, y = load_split(split)
        print(f"[+] Loaded {split}: X.shape={X.shape}, y.shape={y.shape}")

        env = NetworkFlowEnv(X, y, rng_seed=42, episode_length=episode_len)
        input_dim = env.feature_size
        n_actions = env.n_actions

        # Initialize agent once (keeps learning across all days)
        if agent is None:
            agent = DQNAgent(
                input_dim=input_dim,
                n_actions=n_actions,
                device=device,
                lr=lr,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=replay_capacity,   # consistent with train.py naming
                eps_decay_steps=500_000,
                target_sync_every=target_update_steps
            )

        # Train for given number of episodes
        for ep in range(1, episodes + 1):
            start_time = time.time()
            state = env.reset(start_index=0)
            ep_reward = 0.0
            steps = 0

            while True:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.push_transition(
                    state, action, reward,
                    next_state if next_state is not None else None,
                    done
                )
                loss = agent.update()
                state = next_state if next_state is not None else state
                ep_reward += reward
                steps += 1

                if done:
                    break

            # Log metrics
            metrics = env.render_metrics()
            avg_step_reward = ep_reward / max(1, steps)
            row = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'split': split,
                'episode': ep,
                'steps': steps,
                'episode_reward': f"{ep_reward:.6f}",
                'avg_reward_per_step': f"{avg_step_reward:.6f}",
                'tp': metrics.get('true_positives', 0),
                'tn': metrics.get('true_negatives', 0),
                'fp': metrics.get('false_positives', 0),
                'fn': metrics.get('false_negatives', 0),
                'epsilon': f"{agent.epsilon():.4f}"
            }
            append_metrics(row)

            # Save checkpoint periodically
            if ep % 5 == 0:
                ckpt_path = CKPT_DIR / f"dqn_{split}_ep{ep}.pth"
                agent.save(str(ckpt_path))
                print(f"[💾] Checkpoint saved: {ckpt_path}")

            elapsed = time.time() - start_time
            print(
                f"Ep {ep:03d} | {split} | steps {steps:6d} | "
                f"reward {ep_reward:8.4f} | eps {agent.epsilon():.4f} | time {elapsed:.1f}s"
            )

        # End of day — save final model for that split
        final_path = CKPT_DIR / f"dqn_{split}_final.pth"
        agent.save(str(final_path))
        print(f"[✓] Completed training on {split}. Model saved to {final_path}\n")

    print("\n✅ All days trained successfully!")

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--episode_len", type=int, default=2000)
    p.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--target_update_steps", type=int, default=1000)
    args = p.parse_args()

    train_all_days(
        device=args.device,
        episodes=args.episodes,
        episode_len=args.episode_len,
        lr=args.lr,
        batch_size=args.batch_size,
        target_update_steps=args.target_update_steps
    )
