"""
src/dashboard_explain.py

Streamlit dashboard for interactive explainability of Defender/Attacker agents.

Launch:
    streamlit run src/dashboard_explain.py
"""

import os, sys
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import load_split
from src.agent import DQNAgent

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

st.set_page_config(page_title="Explainable Cyber-MARL Dashboard", layout="wide")
st.title("Explainable Cyber-MARL — Agent Decision Explorer")

# Sidebar controls
st.sidebar.header("Agent / Data selection")
agent_type = st.sidebar.selectbox("Agent type", ["defender", "attacker", "marl_agent"])
checkpoint = st.sidebar.text_input("Checkpoint path", "results/checkpoints/defender_final.pth")
split = st.sidebar.selectbox("Data split (for example states)", ["monday","tuesday","wednesday","thursday","friday"])
sample_index = st.sidebar.number_input("Sample index", min_value=0, value=0, step=1)
samples_preview = st.sidebar.slider("Preview samples", 5, 100, 20)

# Load split sample states
try:
    X, y = load_split(split)
    st.sidebar.success(f"Loaded split {split}: {X.shape[0]} samples")
except Exception as e:
    st.sidebar.error(f"Could not load split: {e}")
    X = np.zeros((1,80))
    y = np.zeros(1)

# Load agent
load_ok = False
if os.path.exists(checkpoint):
    try:
        # infer input_dim and n_actions
        input_dim = X.shape[1]
        # default n_actions to 5 if not known
        n_actions = 5
        agent = DQNAgent(input_dim=input_dim, n_actions=n_actions, device=None)
        agent.load(checkpoint)
        load_ok = True
        st.sidebar.success(f"Loaded agent from {checkpoint}")
    except Exception as e:
        st.sidebar.error(f"Failed to load agent: {e}")

else:
    st.sidebar.warning("Checkpoint not found on disk (enter a valid path).")

# Show sample state and agent policy
st.header("Agent policy & sample state")
if load_ok:
    idx = int(np.clip(sample_index, 0, max(0, X.shape[0]-1)))
    state = X[idx]
    st.subheader(f"Sample #{idx}")
    st.write("True label (original):", int(y[idx]))
    st.write("State vector (first 40 dims):")
    st.write(state[:40].tolist())

    q_vals = agent.policy_net(torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)).cpu().detach().numpy()[0]
    action = int(np.argmax(q_vals))
    st.metric("Chosen action (argmax Q)", action)
    st.write("Q-values:", q_vals.tolist())

    # Show aggregate SHAP plot if exists for this checkpoint
    agg_plot = PLOTS / f"shap_top10_{Path(checkpoint).stem}_{split}.png"
    if agg_plot.exists():
        st.image(Image.open(str(agg_plot)), caption="Aggregate SHAP feature importances (top10)")
    else:
        st.info("Run src.explain_agent to generate SHAP plots (aggregate & per-sample).")

    # Show per-sample image if exists
    sample_img = PLOTS / f"shap_sample{idx}_{Path(checkpoint).stem}_{split}.png"
    if sample_img.exists():
        st.image(Image.open(str(sample_img)), caption=f"SHAP for sample {idx}")
    else:
        st.info("Per-sample SHAP image not found. Run explain_agent with small sample count.")
else:
    st.info("Provide a valid checkpoint to view agent policy and explanations.")

# quick preview of first N samples
st.header("State preview (first N samples)")
n_preview = int(min(samples_preview, X.shape[0]))
df_preview = pd.DataFrame(X[:n_preview, :20], columns=[f"F{i}" for i in range(20)])
st.dataframe(df_preview)

st.markdown("---")
st.caption("Note: Kernel SHAP is slow. For quick interactive use, precompute SHAP plots via src.explain_agent "
           "with small sample/background sizes and load images here.")
