# src/attacker_env.py
"""
Lightweight AttackerEnv for real-time DQN attacker training.
Simulates an adversarial interaction loop between:
 - Attacker (RL agent)
 - Defender (DQNAgent loaded in forensics_server)

The environment generates synthetic feature vectors representing
"system state". The defender decides actions ("allow", "block", ...),
and the attacker is rewarded for finding states that slip past detection.

This is a simplified adversarial reinforcement environment designed to
produce s → a → r → s' → done transitions for DQN training.
"""

import numpy as np
import random
import time


class AttackerEnv:
    """
    Synthetic adversarial environment for the attacker DQN.

    Parameters
    ----------
    defender : object or None
        Reference to the defender DQNAgent (forensics_server.AGENT).
    input_dim : int
        Dimensionality of feature vector.
    episode_len : int
        Number of steps per episode before reset.

    State
    -----
    - Continuous feature vector in R^input_dim ∈ [-1, 1]
    - Represents "system features" observable by attacker

    Action
    ------
    - 0 = low-intensity probe (stealthy)
    - 1 = medium perturbation
    - 2 = aggressive attack

    Reward
    -------
    - +1.0 if defender allows risky state (attacker succeeds)
    - -0.2 if defender blocks or detects correctly
    - small random noise added for stochasticity
    """

    def __init__(self, defender=None, input_dim=15, episode_len=20, verbose=False):
        self.defender = defender
        self.input_dim = input_dim
        self.episode_len = episode_len
        self.verbose = verbose

        self.t = 0
        self.state = self._random_state()

    def _random_state(self):
        """Generate random normalized state vector."""
        return np.random.uniform(-1.0, 1.0, self.input_dim).astype(np.float32)

    def reset(self):
        """Reset environment to initial state."""
        self.state = self._random_state()
        self.t = 0
        return self.state

    def _safe_defender_decision(self, features):
        """
        Safe call to defender._forensic_decision (imported lazily)
        to avoid circular imports.
        """
        try:
            from src.forensics_server import _forensic_decision
            if self.defender is None:
                # fallback dummy (simulate weak defender)
                action_str = random.choice(["allow", "inspect", "block"])
                severity = random.random() * 0.5
                idx = ["allow", "inspect", "block", "alert", "log"].index(action_str)
                return action_str, severity, idx
            return _forensic_decision(self.defender, np.asarray(features).reshape(1, -1))
        except Exception as e:
            if self.verbose:
                print("[AttackerEnv] Defender decision failed:", e)
            # fallback random
            action_str = random.choice(["allow", "block"])
            return action_str, 0.0, 0 if action_str == "allow" else 2

    def step(self, action_idx):
        """
        One environment step:
          1. Attacker selects an action index (0..2)
          2. Environment perturbs features
          3. Defender reacts, producing a reward signal
        """
        # 1️⃣ current state
        s = self.state.copy()

        # 2️⃣ attacker perturbation
        delta = (action_idx - 1) * 0.5  # mild scaling: -0.5, 0, +0.5
        noise = np.random.normal(0, 0.08, self.input_dim).astype(np.float32)
        s2 = np.clip(s + delta + noise, -1.0, 1.0)

        # 3️⃣ defender response
        action_str, severity, def_idx = self._safe_defender_decision(s)

        # 4️⃣ reward shaping (attacker success if defender fails)
        # attacker rewarded if defender "allows" risky states (sum>threshold)
        risky = float(np.mean(s) > 0.25 or np.std(s) > 0.6)
        success = (action_str == "allow" and risky > 0)
        reward = 1.0 if success else -0.2
        reward += np.random.normal(0, 0.02)  # slight noise

        # 5️⃣ update state and episode counter
        self.state = s2
        self.t += 1
        done = self.t >= self.episode_len

        # optional debug
        if self.verbose:
            print(f"[step {self.t}] act={action_idx}, def={action_str}, sev={severity:.3f}, r={reward:+.3f}")

        return s2, float(reward), bool(done), {
            "def_action": action_str,
            "def_severity": severity,
            "risky_state": risky,
            "t": self.t
        }
