#!/usr/bin/env python3
"""
src/game_theory.py

Stackelberg helper utilities for the Cyber-MARL project.

Provides:
 - estimate_equilibrium_from_history: estimate attacker's equilibrium action (float/index)
 - compute_stackelberg_penalty: penalty term λ * (a_def - a_eq)^2
 - shape_defender_reward: combine MARL reward and penalty.
 - estimate_eq_from_attacker_model: attacker-model-based equilibrium estimation
 - compute_stackelberg_lp: LP-based Stackelberg defender optimization
 - build_payoff_matrix: payoff matrix generator for attacker-defender domains
 - export_stackelberg_history: export LP training evolution (for plotting/reporting)

Design notes:
 - a_def may be discrete (int) or continuous (float). a_eq is returned as float.
 - The LP solver computes the defender’s optimal mixed strategy ensuring a guaranteed payoff.
 - Uses scipy.optimize.linprog for Stackelberg linear program.
"""

from typing import Sequence, Callable, Optional
import numpy as np
import time
import logging
from pathlib import Path
import csv
from scipy.optimize import linprog

# ---------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------
logger = logging.getLogger("game_theory")
logger.setLevel(logging.INFO)

# ===========================================================
# 1️⃣ Basic Equilibrium and Reward Shaping Utilities
# ===========================================================
def estimate_equilibrium_from_history(
    attacker_action_history: Optional[Sequence[Sequence[float]]] = None,
    attacker_probs: Optional[Sequence[float]] = None,
    fallback_n_actions: int = 3,
) -> float:
    """
    Estimate an equilibrium (expected attacker action index) from history or model probabilities.
    """
    if attacker_probs is not None:
        p = np.asarray(attacker_probs, dtype=float)
        if p.ndim != 1:
            raise ValueError("attacker_probs must be a 1-D vector")
        idx = np.arange(len(p))
        a_eq = float(np.dot(idx, p))
        logger.debug("Estimated a_eq from attacker_probs: %s", a_eq)
        return a_eq

    if attacker_action_history:
        arr = np.asarray(attacker_action_history)
        if arr.ndim == 1:
            a_eq = float(arr.mean())
            logger.debug("Estimated a_eq from index history: %s", a_eq)
            return a_eq
        elif arr.ndim == 2:
            avg = arr.mean(axis=0)
            idx = np.arange(avg.shape[0])
            a_eq = float(np.dot(idx, avg))
            logger.debug("Estimated a_eq from prob history: %s", a_eq)
            return a_eq

    a_eq = (fallback_n_actions - 1) / 2.0
    logger.debug("Fallback a_eq: %s", a_eq)
    return float(a_eq)


def compute_stackelberg_penalty(a_def: float, a_eq: float, lambda_: float = 1.0) -> float:
    """Compute quadratic penalty term: lambda_ * (a_def - a_eq)^2"""
    return float(lambda_ * (float(a_def) - float(a_eq)) ** 2.0)


def shape_defender_reward(
    reward_marl: float,
    a_def: float,
    attacker_action_history: Optional[Sequence[Sequence[float]]] = None,
    attacker_probs: Optional[Sequence[float]] = None,
    lambda_: float = 1.0,
    fallback_n_actions: int = 3,
) -> (float, dict):
    """
    Shape defender reward by including equilibrium penalty.
    Returns shaped reward and metadata.
    """
    a_eq = estimate_equilibrium_from_history(attacker_action_history, attacker_probs, fallback_n_actions)
    penalty = compute_stackelberg_penalty(a_def, a_eq, lambda_)
    shaped = float(reward_marl - penalty)
    meta = {"a_def": float(a_def), "a_eq": float(a_eq), "penalty": penalty, "lambda": float(lambda_)}
    return shaped, meta


def estimate_eq_from_attacker_model(
    attacker_model_fn: Callable[[np.ndarray], Sequence[float]],
    state: np.ndarray,
) -> float:
    """Estimate equilibrium from attacker model: attacker_model_fn(state) -> prob vector"""
    probs = attacker_model_fn(state)
    return estimate_equilibrium_from_history(attacker_probs=probs)


# ===========================================================
# 2️⃣ LP-Based Stackelberg Game Solver
# ===========================================================
def build_payoff_matrix(defender_rewards: dict):
    """
    Build payoff matrix M for defender–attacker Stackelberg game.
    """
    domains = list(defender_rewards.keys())
    n = len(domains)
    rewards = np.array(list(defender_rewards.values()), dtype=float)

    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0.5 * rewards[i]  # attack success reduces payoff
            else:
                M[i, j] = rewards[i]        # defender normal payoff
    return domains, M


def compute_stackelberg_lp(defender_rewards: dict, attack_costs: dict = None):
    """
    Solve defender's Stackelberg LP (leader optimization problem).

    LP Formulation:
        maximize v
        s.t. M^T p >= v * 1
             sum(p) = 1, p >= 0

    Returns:
        domains: list[str]
        p: np.ndarray - defender mixed strategy
        best_js: list[int] - attacker best pure response indices
        v: float - defender guaranteed value
    """
    domains, M = build_payoff_matrix(defender_rewards)
    n = len(domains)

    # Objective: maximize v → minimize -v
    c = np.zeros(n + 1)
    c[-1] = -1  # -v term

    # Constraints: M^T p - v*1 >= 0 → -M^T p + v*1 <= 0
    A_ub = np.hstack([-M.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    # Equality constraint: sum(p) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]

    # Bounds: 0 <= p_i <= 1, v free
    bounds = [(0, 1) for _ in range(n)] + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError(f"Stackelberg LP failed: {res.message}")

    p = res.x[:n]
    v = res.x[-1]

    # Attacker best responses (minimizing defender payoff)
    attacker_payoffs = M.T @ p
    best_val = attacker_payoffs.min()
    best_js = np.where(np.isclose(attacker_payoffs, best_val))[0].tolist()

    return domains, p, best_js, v


# ===========================================================
# 3️⃣ Optional Logging / Export Utilities
# ===========================================================
def export_stackelberg_history(history: list, out_dir: Path):
    """
    Export Stackelberg LP training history to CSV for visualization.
    Each row: epoch, value(v), defender probabilities...
    """
    if not history:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stackelberg_history.csv"

    domains = list(history[0]["p"].keys()) if isinstance(history[0]["p"], dict) else None
    if domains is None:
        # fallback if p is a NumPy array
        n = len(history[0]["p"])
        domains = [f"domain_{i}" for i in range(n)]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "v"] + domains)
        for entry in history:
            p_vals = entry["p"] if isinstance(entry["p"], (list, np.ndarray)) else list(entry["p"].values())
            writer.writerow([entry["ep"], entry["v"]] + list(map(float, p_vals)))

    print(f"[💾] Exported Stackelberg history → {out_path}")
    return out_path
