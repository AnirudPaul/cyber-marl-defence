#!/usr/bin/env python3
"""
src/game_train.py

Phase 8 – Stackelberg Attacker–Defender Simulation (LP formulation)

Now dynamically loads defender rewards from results/metrics.csv (MARL outputs).
Usage:
    python -m src.game_train --epochs 20
"""

import os, sys, time, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.game_theory import compute_stackelberg_lp

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 🔹 Utility: Load mean defender rewards per weekday from metrics.csv
# ---------------------------------------------------------------------
def load_defender_rewards_from_csv(metrics_csv: Path):
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_csv}")

    rewards = defaultdict(list)
    with open(metrics_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "").lower()
            if split in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
                try:
                    r = float(row["episode_reward"])
                    rewards[split].append(r)
                except ValueError:
                    continue

    # compute mean per split
    mean_rewards = {k: np.mean(v) if v else 0.0 for k, v in rewards.items()}

    print("\n[✓] Loaded defender rewards from metrics.csv:")
    for k, v in mean_rewards.items():
        print(f"   {k.capitalize():10s}: {v:.4f}")
    return mean_rewards


# ---------------------------------------------------------------------
# 🔹 Stackelberg Simulation Loop
# ---------------------------------------------------------------------
def simulate_game_lp(epochs=20):
    print("\n======================================================================")
    print(" PHASE 8: Stackelberg Attacker–Defender (LP) Simulation")
    print("======================================================================")

    metrics_csv = RESULTS / "metrics.csv"
    defender_rewards = load_defender_rewards_from_csv(metrics_csv)

    # fallback if any day missing
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        defender_rewards.setdefault(day, 0.98)

    attack_costs = {day: 1.0 for day in defender_rewards}

    history = []
    for ep in range(1, epochs + 1):
        domains, p, best_js, v = compute_stackelberg_lp(defender_rewards, attack_costs)
        attacker_choices = [domains[j] for j in best_js]

        print(f"\nEpoch {ep:03d} | Defender guaranteed value v = {v:.4f}")
        for d, prob in zip(domains, p):
            print(f"   Defender[{d}] = {prob:.3f}")
        print(f"   Attacker best-response(s): {attacker_choices}")
        print("-" * 60)

        history.append({"ep": ep, "v": v, "p": p.copy(), "attacker": attacker_choices})

        # attacker adaptation: reduce defender strength on chosen domain
        chosen = np.random.choice(attacker_choices)
        defender_rewards[chosen] = max(0.7 * defender_rewards[chosen], 0.01)

    # plot payoff curve
    vs = [h["v"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(vs) + 1), vs, marker="o")
    plt.title("Stackelberg Leader Value (Defender guaranteed payoff)")
    plt.xlabel("Epoch")
    plt.ylabel("Defender guaranteed value v")
    plt.grid(True)
    out = RESULTS / "stackelberg_payoff_curve_lp.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"[📊] Saved payoff curve → {out}")

    return history


# ---------------------------------------------------------------------
# 🔹 CLI Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    simulate_game_lp(epochs=args.epochs)
