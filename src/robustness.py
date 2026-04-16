"""
src/robustness.py

Phase 5 – Robustness Evaluation
--------------------------------
Adds Gaussian noise to the evaluation dataset and computes the robustness index:

    R_robust = 1 - |A_perf - A_pert| / A_perf

Outputs:
    results/robustness.csv
    results/plots/confmat_<split>_perturbed.png
"""

import os, sys, argparse, time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import DQNAgent
from src.environment import NetworkFlowEnv
from src.evaluate import load_split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)


def evaluate_model(agent, X, y):
    """Run model once over dataset and return accuracy + predictions."""
    env = NetworkFlowEnv(X, y, rng_seed=42)
    s = env.reset(0)
    y_true, y_pred = [], []

    with torch.no_grad():
        for _ in range(len(X)):
            a = agent.act(s, eval_mode=True)
            ns, r, done, info = env.step(a)
            y_true.append(int(info["label"]))
            y_pred.append(1 if a in [1, 2] else 0)
            if done:
                break
            s = ns

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm


def add_noise(X, noise_level=0.1, seed=42):
    """Add Gaussian noise (mean=0, std=noise_level * std(feature))."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level * np.std(X, axis=0), X.shape)
    return X + noise


def save_confmat(cm, split, tag):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {split} ({tag})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha="center", va="center", color="red")
    path = PLOTS / f"confmat_{split}_{tag}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[📊] Saved {tag} confusion matrix → {path}")


def robustness_test(split, checkpoint, device=None, noise_level=0.1, limit=None):
    print(f"\n======================================================================")
    print(f" PHASE 5: ROBUSTNESS TESTING - Split: {split}")
    print(f"======================================================================")

    X, y = load_split(split)
    if limit:
        X, y = X[:limit], y[:limit]
        print(f"[i] Using first {limit} samples for quick test")

    agent = DQNAgent(input_dim=X.shape[1], n_actions=3, device=device)
    agent.load(checkpoint)
    agent.policy_net.eval()

    print("[+] Evaluating on clean data...")
    acc_perf, cm_perf = evaluate_model(agent, X, y)
    save_confmat(cm_perf, split, "clean")

    print("[+] Evaluating on perturbed (noisy) data...")
    X_noisy = add_noise(X, noise_level=noise_level)
    acc_pert, cm_pert = evaluate_model(agent, X_noisy, y)
    save_confmat(cm_pert, split, "perturbed")

    R_robust = 1 - abs(acc_perf - acc_pert) / acc_perf
    print(f"\n[✓] Results:")
    print(f"  Clean Accuracy     = {acc_perf:.4f}")
    print(f"  Perturbed Accuracy = {acc_pert:.4f}")
    print(f"  Robustness Index   = {R_robust:.4f}")

    # write CSV
    csv_path = RESULTS / "robustness.csv"
    header = "timestamp,split,noise_level,acc_perf,acc_pert,robustness\n"
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')},{split},{noise_level},{acc_perf:.6f},{acc_pert:.6f},{R_robust:.6f}\n"
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write(header)
    with open(csv_path, "a") as f:
        f.write(line)
    print(f"[🧮] Logged to {csv_path}")

    return acc_perf, acc_pert, R_robust


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="wednesday")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    robustness_test(args.split, args.checkpoint, args.device, args.noise, args.limit)
