"""
src/evaluate.py

Evaluate a trained DQN model on any data split (binary or multi-class).
Computes accuracy, precision, recall, F1, confusion matrix, and saves plots.

Usage Example:
    python -m src.evaluate --split wednesday --checkpoint results/checkpoints/dqn_wednesday_final.pth --device cuda
"""

import os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

# Ensure src imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import DQNAgent
from src.environment import NetworkFlowEnv

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split):
    path = SPLIT_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def evaluate(split, checkpoint, device=None, limit=None, plot=True):
    print(f"\n======================================================================")
    print(f" PHASE 4: EVALUATION - Split: {split}")
    print(f"======================================================================")
    print(f"[+] Evaluating checkpoint: {checkpoint}")

    X, y = load_split(split)
    if limit:
        X, y = X[:limit], y[:limit]
        print(f"[i] Using first {limit} samples for quick evaluation")

    env = NetworkFlowEnv(X, y, rng_seed=42)
    input_dim, n_actions = env.feature_size, env.n_actions

    agent = DQNAgent(input_dim=input_dim, n_actions=n_actions, device=device)
    agent.load(checkpoint)
    agent.policy_net.eval()

    y_true, y_pred = [], []
    total_reward = 0.0
    state = env.reset(0)

    with torch.no_grad():
        for i in range(len(X)):
            action = agent.act(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            y_true.append(int(info["label"]))

            # Prediction heuristic: action {1,2}=attack, 0=benign
            y_pred.append(1 if action in [1, 2] else 0)

            if done:
                break
            state = next_state

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Detect number of unique labels
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    print(f"[i] Detected {n_classes} unique labels: {unique_classes}")

    # Compute metrics safely
    if n_classes > 2:
        avg = "weighted"
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print(f"[✓] Multiclass metrics computed (weighted average).")
    else:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        print(f"[✓] Binary metrics computed.")

    # Extract confusion matrix values safely
    tn = fp = fn = tp = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

    # Print metrics summary
    print("\n==================== Evaluation Summary ====================")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Total Reward: {total_reward:.2f}")
    print("============================================================")

    # Save confusion matrix plot
    if plot:
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix - {split}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, f"{val}", ha="center", va="center", color="red")
        path = RESULTS_DIR / f"confmat_{split}.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[📊] Confusion matrix saved → {path}\n")

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="wednesday")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    evaluate(args.split, args.checkpoint, args.device, args.limit)
