#!/usr/bin/env python3
"""
src/explain.py

Use SHAP to interpret a trained DQN's predictions for specific flows.

Usage:
    python -m src.explain --split wednesday --checkpoint results/checkpoints/dqn_wednesday_final.pth --samples 100
"""

import os, sys, argparse
from pathlib import Path
import numpy as np
import shap
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import DQNAgent

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_split(split):
    p = SPLIT_DIR / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset split: {p}")
    d = np.load(p, allow_pickle=True)
    return d['X'], d['y']

def explain(split, checkpoint, samples=100, device=None):
    print(f"[+] SHAP explainability on split={split}, checkpoint={checkpoint}")

    X, y = load_split(split)
    X_small = X[:samples]
    input_dim, n_actions = X.shape[1], 3

    # Initialize agent
    agent = DQNAgent(input_dim, n_actions, device=device)
    agent.load(checkpoint)
    agent.policy_net.eval()

    # Prepare data
    background = torch.tensor(X_small[:50], dtype=torch.float32, device=agent.device)
    test_data = torch.tensor(X_small, dtype=torch.float32, device=agent.device)

    # SHAP explanation
    explainer = shap.DeepExplainer(agent.policy_net, background)
    shap_values = explainer.shap_values(test_data)

    # --- Handle SHAP output shape ---
    # SHAP returns list of arrays (one per action) or a single 3D array
    if isinstance(shap_values, list):
        # Average across all action outputs
        shap_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        # SHAP single array (samples, features, actions)
        shap_combined = np.mean(np.abs(shap_values), axis=-1)

    # Summarize average absolute SHAP values per feature
    mean_abs = np.mean(shap_combined, axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(range(10), mean_abs[top_idx])
    plt.xticks(range(10), [f"F{i}" for i in top_idx], rotation=45)
    plt.title(f"Top 10 Feature Importances (SHAP) - {split}")
    plt.tight_layout()

    path = RESULTS_DIR / f"shap_top10_{split}.png"
    plt.savefig(path)
    plt.close()
    print(f"[📈] SHAP feature importance saved → {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="wednesday")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    explain(args.split, args.checkpoint, args.samples, args.device)
