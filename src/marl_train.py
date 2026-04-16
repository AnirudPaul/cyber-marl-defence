"""
src/marl_train.py

Phase 7 – Cooperative Multi-Agent DQN Training
"""

import os, sys, time
from pathlib import Path
import numpy as np
from src.multi_agent import MultiAgentCoordinator
from src.evaluate import load_split

ROOT = Path(__file__).resolve().parents[1]
CKPTS = ROOT / "results" / "checkpoints"
CKPTS.mkdir(parents=True, exist_ok=True)

def main(device=None, epochs=10, steps_per_agent=1000):
    print("\n======================================================================")
    print(" PHASE 7: Multi-Agent Reinforcement Learning")
    print("======================================================================")

    # Load per-domain splits
    splits = {}
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        X, y = load_split(day)
        splits[day] = (X, y)  # use small subset for demo

    marl = MultiAgentCoordinator(splits, checkpoint_dir=CKPTS, device=device)

    for ep in range(1, epochs + 1):
        rewards = marl.train_one_epoch(steps_per_agent=steps_per_agent)
        print(f"Epoch {ep:03d} | Rewards per agent:")
        for k, v in rewards.items():
            print(f"  {k:10s}: {v:.3f}")
        if ep % 5 == 0:
            marl.save_all(ep)
    print("\n[✓] Multi-Agent training complete.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    main(device=args.device, epochs=args.epochs, steps_per_agent=args.steps)
