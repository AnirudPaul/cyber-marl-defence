# Cyber-MARL Defence — AI vs AI for Network Intrusion Detection

> **Live demo →** [yourusername.github.io/cyber-marl-defence](https://yourusername.github.io/cyber-marl-defence)

Three research papers on adversarial robustness in ML-based Network Intrusion Detection Systems (NIDS), evaluated on CIC-IDS-2017, UNSW-NB15, and TON-IoT.

---

## Papers

### 1 · CPBA — Constraint-Preserving Backdoor Attacks
Backdoor triggers that satisfy protocol validity, socket-level realizability, and inter-feature statistical constraints — making them indistinguishable from legitimate traffic.

- **Attack Success Rate** > 96% across all models
- **Clean Accuracy** > 99% (no utility loss)
- Datasets: CIC-IDS-2017, UNSW-NB15, TON-IoT

### 2 · PAN — Perturbation-Aware Normalization
A learnable gate that blends LayerNorm with raw activations — tightens normalization under perturbations, relaxes it for clean inputs.

- **+39% Clean Macro-F1** vs PGD adversarial training
- Near-zero parameter overhead
- Datasets: CIC-IDS-2017, UNSW-NB15, TON-IoT

### 3 · ASTRA — Asymmetric RL Adversarial Training
A DDPG attacker learns adaptive perturbations; a Dueling-DQN defender trains on those examples via supervised cross-entropy.

- **Robust Macro-F1: 0.761** on CIC-IDS-2017
- **F1 drop < 0.03%** under black-box attack
- LayerNorm reduces Jacobian norms by **42x-870x**
- Datasets: CIC-IDS-2017, UNSW-NB15

---

## Repository Structure

```
cyber-marl-defence/
├── docs/                    # GitHub Pages demo site (pure HTML/JS)
│   ├── index.html
│   └── assets/
│       ├── css/style.css
│       └── js/
│           ├── battle.js    # ASTRA battle arena simulation
│           └── charts.js    # Results charts (Chart.js)
│
├── src/
│   ├── agent.py             # Base RL agent
│   ├── attacker.py          # DDPG attacker agent
│   ├── attacker_env.py      # Attacker environment
│   ├── environment.py       # NIDS gym environment
│   ├── multi_agent.py       # MARL training loop
│   ├── marl_train.py        # Main training entry point
│   ├── train.py             # Supervised training
│   ├── evaluate.py          # Evaluation scripts
│   ├── robustness.py        # Robustness testing
│   ├── reward.py            # Reward functions
│   ├── game_theory.py       # Stackelberg game utilities
│   ├── explain.py           # SHAP / LIME explainability
│   └── data_preprocess.py   # Dataset preprocessing
│
├── results/
│   └── plots/               # Generated figures
│
├── requirements.txt
└── Readme.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/cyber-marl-defence.git
cd cyber-marl-defence
pip install -r requirements.txt
```

> **Datasets not included** (too large for git).
> Download CIC-IDS-2017 from [unb.ca/cic/datasets](https://www.unb.ca/cic/datasets/ids-2017.html)
> and place CSVs in `data/GeneratedLabelledFlows/`.

---

## Quickstart

```bash
# 1. Preprocess data
python src/data_preprocess.py

# 2. Train ASTRA (DDPG attacker + DQN defender)
python src/marl_train.py

# 3. Evaluate robustness
python src/evaluate.py

# 4. Run robustness sweep
python src/robustness.py
```

---

## GitHub Pages Demo

The `docs/` folder is a self-contained static site. No server, no build step.

To enable:
**GitHub -> Settings -> Pages -> Source: Deploy from branch -> main -> /docs**

---

## Results Summary

| Model | Clean Macro-F1 | Under Attack | Drop |
|---|---|---|---|
| XGB-Standard | 0.9571 | 0.6861 | -27.1% |
| Random Forest | 0.8976 | 0.6015 | -29.6% |
| PGD-AT (DNN)  | 0.4998 | 0.4999 | ~0% |
| **ASTRA (Ours)** | **0.7608** | **0.7610** | **<0.03%** |

*CIC-IDS-2017, perturbation scale sigma=0.1*

---

*Papers under review. Code released for reproducibility.*
