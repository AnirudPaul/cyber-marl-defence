"""
reward.py

Implements the reward function used by the NetworkFlowEnv.

R = alpha * D_acc - beta * FPR - delta * O_comp

Where:
 - D_acc : 1 if correct decision (malicious blocked / benign allowed), else 0
 - FPR   : 1 if false positive (benign wrongly blocked/inspected->blocked), else 0
 - O_comp: computational/latency overhead cost (float), e.g., cost of inspection

This module also exposes a helper to convert integer action -> human label.
"""

import numpy as np

# action mapping
ACTION_ALLOW = 0
ACTION_BLOCK = 1
ACTION_INSPECT = 2

def action_name(a: int) -> str:
    return {0: "allow", 1: "block", 2: "inspect"}.get(int(a), "unknown")

def compute_reward(is_attack: bool,
                   action: int,
                   detect_if_inspect_prob: float = 0.85,
                   false_positive_if_inspect_prob: float = 0.02,
                   alpha: float = 1.0,
                   beta: float = 1.0,
                   delta: float = 0.05,
                   inspect_cost: float = 0.02,
                   rng: np.random.RandomState = None) -> (float, dict):
    """
    Compute reward for a single flow decision.

    Parameters
    ----------
    is_attack : bool
        True if the flow label is malicious.
    action : int
        0=allow, 1=block, 2=inspect
    detect_if_inspect_prob : float
        Probability that 'inspect' correctly detects an attack.
    false_positive_if_inspect_prob : float
        Probability that 'inspect' incorrectly flags benign as malicious.
    alpha, beta, delta : floats
        Reward weights for detection accuracy, false-positive penalty, and overhead.
    inspect_cost : float
        Overhead penalty for 'inspect' action (applied as O_comp).
    rng : np.random.RandomState or None
        Random generator for deterministic testing (optional).

    Returns
    -------
    reward : float
        Scalar reward.
    info : dict
        info fields: {'detected':bool, 'D_acc':0/1, 'FPR':0/1, 'O_comp':float}
    """
    if rng is None:
        rng = np.random.RandomState()

    detected = False
    if action == ACTION_BLOCK:
        # Block assumes detection is immediate and correct if it's actually attack.
        detected = bool(is_attack)
    elif action == ACTION_ALLOW:
        detected = False
    elif action == ACTION_INSPECT:
        # Inspect may detect attack probabilistically, and may produce FP on benign.
        if is_attack:
            detected = rng.rand() < float(detect_if_inspect_prob)
        else:
            # false positive chance when inspecting a benign flow
            detected = rng.rand() < float(false_positive_if_inspect_prob)
    else:
        raise ValueError("Unknown action: %r" % (action,))

    # Detection accuracy (1 if correct decision: detected & attack OR not detected & benign)
    D_acc = 1.0 if ((detected and is_attack) or (not detected and not is_attack)) else 0.0
    # False positive (1 if benign and detected)
    FPR = 1.0 if (detected and not is_attack) else 0.0
    # Overhead cost
    O_comp = float(inspect_cost) if action == ACTION_INSPECT else 0.0

    reward = alpha * D_acc - beta * FPR - delta * O_comp

    info = {
        "detected": bool(detected),
        "D_acc": float(D_acc),
        "FPR": float(FPR),
        "O_comp": float(O_comp)
    }
    return float(reward), info
