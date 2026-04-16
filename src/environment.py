"""
environment.py

Gym-like environment wrapping preprocessed CIC-IDS-2017 flows.

API (minimal):
 - env = NetworkFlowEnv(X, y, config...)
 - state = env.reset(start_index=0)
 - next_state, reward, done, info = env.step(action)
 - env.n_actions == 3
 - env.feature_size gives input dimension

State: scaled numeric vector (1D numpy array)
Actions: 0=allow, 1=block, 2=inspect

This environment is deterministic in sequencing but 'inspect' detection is probabilistic.
It is designed to be simple and fast for RL training proof-of-concept.
"""

import numpy as np
from typing import Optional, Tuple
from src.reward import compute_reward, ACTION_ALLOW, ACTION_BLOCK, ACTION_INSPECT

class NetworkFlowEnv:
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 delta: float = 0.05,
                 inspect_detect_prob: float = 0.85,
                 inspect_fp_prob: float = 0.02,
                 inspect_cost: float = 0.02,
                 rng_seed: Optional[int] = None,
                 episode_length: Optional[int] = None):
        """
        Parameters:
          X, y: numpy arrays (rows aligned). X is scaled numeric features.
          alpha, beta, delta: reward hyperparameters (see reward.py).
          inspect_detect_prob: P(detect attack | inspect)
          inspect_fp_prob: P(flag benign | inspect)
          inspect_cost: overhead cost for inspect action
          rng_seed: optional seed for reproducibility
          episode_length: if set, each episode has fixed number of steps; else runs until data exhausted
        """
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
        self.X = X
        self.y = y.astype(int)
        self.N = X.shape[0]
        self.feature_size = X.shape[1]
        self.n_actions = 3

        # reward params
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.inspect_detect_prob = float(inspect_detect_prob)
        self.inspect_fp_prob = float(inspect_fp_prob)
        self.inspect_cost = float(inspect_cost)

        self.rng = np.random.RandomState(rng_seed)

        # internal indices
        self.idx = 0
        self.episode_length = episode_length
        self.steps_in_episode = 0

        # metrics accumulators
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {
            "total_reward": 0.0,
            "steps": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0
        }

    def reset(self, start_index: int = 0):
        """
        Reset environment to a given start_index (default 0).
        Returns first state.
        """
        if start_index < 0 or start_index >= self.N:
            raise IndexError("start_index out of range")
        self.idx = int(start_index)
        self.steps_in_episode = 0
        self.reset_metrics()
        return self.X[self.idx].copy()

    def step(self, action: int) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        """
        Apply action to current flow and advance index by 1.
        Returns: (next_state or None if done), reward, done, info
        """
        if self.idx >= self.N:
            return None, 0.0, True, {"msg": "no more data"}

        state = self.X[self.idx]
        label = int(self.y[self.idx])
        is_attack = (label != 0)  # convention: 0 == benign (depends on label encoder; check scalers.pkl metadata)

        reward, info = compute_reward(
            is_attack=is_attack,
            action=action,
            detect_if_inspect_prob=self.inspect_detect_prob,
            false_positive_if_inspect_prob=self.inspect_fp_prob,
            alpha=self.alpha,
            beta=self.beta,
            delta=self.delta,
            inspect_cost=self.inspect_cost,
            rng=self.rng
        )

        # update metrics counters using info
        detected = info["detected"]
        if is_attack and detected:
            self.metrics["true_positives"] += 1
        if (not is_attack) and detected:
            self.metrics["false_positives"] += 1
        if (not is_attack) and (not detected):
            self.metrics["true_negatives"] += 1
        if is_attack and (not detected):
            self.metrics["false_negatives"] += 1

        self.metrics["total_reward"] += reward
        self.metrics["steps"] += 1
        self.steps_in_episode += 1

        # advance index
        self.idx += 1
        done = False
        if self.idx >= self.N:
            done = True
            next_state = None
        else:
            next_state = self.X[self.idx].copy()

        # episode length termination (if configured)
        if (self.episode_length is not None) and (self.steps_in_episode >= self.episode_length):
            done = True

        info_out = {
            "idx": self.idx - 1,
            "action": int(action),
            "action_name": {0: "allow", 1: "block", 2: "inspect"}.get(int(action)),
            "label": int(label),
            "is_attack": bool(is_attack),
            "detected": info["detected"],
            "reward_components": {"D_acc": info["D_acc"], "FPR": info["FPR"], "O_comp": info["O_comp"]},
            "cumulative_reward": float(self.metrics["total_reward"])
        }
        return next_state, float(reward), bool(done), info_out

    def seed(self, s: int):
        self.rng = np.random.RandomState(s)

    def sample_random_action(self):
        return int(self.rng.randint(0, self.n_actions))

    def render_metrics(self):
        return self.metrics.copy()
