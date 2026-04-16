"""
PHASE 11 — Final Stable SHAP Explainability for DQN Cyber Defence
Compatible with: multi-output models, variable feature counts, GPU/CPU.
Author: GPT-5 (for Anirudh’s cyber_marl_defence project)
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import pickle

# project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import DQNAgent
from src.evaluate import load_split

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


# ============================== HELPERS ==============================

def align_features(X, expected_dim):
    """Trim or pad feature columns to match checkpoint model input size."""
    current_dim = X.shape[1]
    if current_dim > expected_dim:
        print(f"[i] Trimming dataset features from {current_dim} → {expected_dim}")
        X = X[:, :expected_dim]
    elif current_dim < expected_dim:
        print(f"[i] Padding dataset features from {current_dim} → {expected_dim}")
        X = np.pad(X, ((0, 0), (0, expected_dim - current_dim)), mode="constant")
    return X


def normalize_shap_values(shap_values, n_features):
    """Normalize SHAP outputs to shape (n_actions, n_samples, n_features)."""
    if isinstance(shap_values, list):
        clean = []
        for sv in shap_values:
            arr = np.array(sv, dtype=float)
            if arr.ndim == 1:  # (features,)
                arr = arr[np.newaxis, :]
            elif arr.ndim == 2 and arr.shape[1] != n_features:
                arr = np.resize(arr, (arr.shape[0], n_features))
            clean.append(arr)
        max_len = max(c.shape[0] for c in clean)
        clean = [
            np.pad(c, ((0, max_len - c.shape[0]), (0, 0)), mode='edge') for c in clean
        ]
        shap_arr = np.stack(clean, axis=0)
    else:
        arr = np.array(shap_values, dtype=float)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        shap_arr = arr
    return shap_arr


def load_feature_names(expected_dim):
    """Load real feature names if available, else use generic ones."""
    feat_path = ROOT / "data" / "processed" / "feature_cols.pkl"
    if feat_path.exists():
        with open(feat_path, "rb") as f:
            cols = pickle.load(f)
        if len(cols) >= expected_dim:
            print(f"[+] Loaded {len(cols)} feature names from feature_cols.pkl")
            return cols[:expected_dim]
    print(f"[i] Using generic F0..F{expected_dim - 1}")
    return [f"F{i}" for i in range(expected_dim)]


# ============================== CORE ==============================

def explain_checkpoint(checkpoint, split="wednesday", samples=100, background=50, topk=10, device="cpu"):
    print("======================================================================")
    print(f" PHASE 11: SHAP Explainability - {Path(checkpoint).stem}")
    print("======================================================================")

    # Load data
    X, _ = load_split(split)
    X_small = X[:samples]
    X_background = X[:background] if background <= len(X) else X[:samples]

    # Load checkpoint metadata
    ckpt = torch.load(checkpoint, map_location=device)
    policy_state = ckpt["policy_state_dict"]
    input_dim = list(policy_state.values())[0].shape[1]
    n_actions = list(policy_state.values())[-1].shape[0]
    print(f"[i] Checkpoint expects input_dim={input_dim}, n_actions={n_actions}")

    # Align data
    X_small = align_features(X_small, input_dim)
    X_background = align_features(X_background, input_dim)

    # Load agent
    agent = DQNAgent(input_dim=input_dim, n_actions=n_actions, device=device)
    agent.load(checkpoint)
    agent.policy_net.eval()

    feat_names = load_feature_names(input_dim)
    print(f"[+] Loaded agent on {agent.device}")
    print(f"[*] Using {samples} samples ({background} background) from split '{split}'")

    # Prediction function
    def predict(x):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=agent.device)
            return agent.policy_net(xt).cpu().numpy()

    # SHAP explainability
    try:
        background_t = torch.tensor(X_background, dtype=torch.float32, device=agent.device)
        explainer = shap.GradientExplainer(agent.policy_net, background_t)
        shap_values = explainer.shap_values(torch.tensor(X_small, dtype=torch.float32, device=agent.device))
        print("[✓] Used GradientExplainer (GPU-accelerated).")
    except Exception as e:
        print(f"[!] GradientExplainer failed ({e}), switching to KernelExplainer.")
        explainer = shap.KernelExplainer(predict, X_background)
        shap_values = explainer.shap_values(X_small, nsamples=100)

    # Normalize SHAP output
    shap_arr = normalize_shap_values(shap_values, input_dim)
    mean_abs = np.mean(np.abs(shap_arr), axis=(0, 1))
    top_idx = np.argsort(mean_abs)[::-1][:topk]

    # ========================= AGGREGATE PLOT =========================
    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(top_idx))
    widths = mean_abs[top_idx]
    plt.barh(y_pos, widths, align='center')
    plt.yticks(y_pos, [feat_names[i] for i in top_idx])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top-{topk} Important Features — {Path(checkpoint).stem}")
    out_agg = PLOTS / f"shap_top{topk}_{Path(checkpoint).stem}_{split}.png"
    plt.tight_layout()
    plt.savefig(out_agg)
    plt.close()
    print(f"[+] Saved aggregate SHAP plot → {out_agg}")

    # ========================= PER-SAMPLE PLOTS =========================
    for i in range(min(10, len(X_small))):
        sv = np.mean(shap_arr[:, i, :], axis=0)
        idx = np.argsort(np.abs(sv))[::-1][:topk]
        y_pos = np.arange(len(idx))
        plt.figure(figsize=(7, 3))
        plt.barh(y_pos, sv[idx], align='center')
        plt.yticks(y_pos, [feat_names[j] for j in idx])
        plt.gca().invert_yaxis()
        plt.title(f"Sample {i} | Mean SHAP (Top {topk})")
        out = PLOTS / f"shap_sample{i}_{Path(checkpoint).stem}_{split}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    print(f"[✓] Saved per-sample SHAP plots → {PLOTS}")
    print("[✓] Phase 11 completed successfully.")
    print("======================================================================\n")

    return {"agg_plot": str(out_agg), "samples_dir": str(PLOTS)}


# ============================== MAIN ==============================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="wednesday")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--background", type=int, default=50)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    explain_checkpoint(
        args.checkpoint,
        split=args.split,
        samples=args.samples,
        background=args.background,
        topk=args.topk,
        device=args.device,
    )
