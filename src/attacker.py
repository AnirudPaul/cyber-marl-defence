"""
src/attacker.py

Wrapper for RL-based attacker. Reuses DQNAgent in src.agent.
State: defender mixed-probability vector p (len = number of domains).
Action: choose target domain index to attack.
Reward: negative defender expected payoff for chosen attack (attacker tries to minimize defender payoff).
"""

import os, sys
import numpy as np
from pathlib import Path

# allow package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import DQNAgent

class RLAttacker:
    def __init__(self, n_domains: int, device: str = None,
                 hidden=(128,64), lr=1e-4, gamma=0.99, batch_size=64, buffer_size=50_000):
        """
        n_domains: number of possible attack targets (same as defender domains)
        device: 'cuda' or 'cpu'
        """
        self.n = n_domains
        self.device = device
        # The attacker observes the defender mixed vector p as state (dimension = n)
        self.agent = DQNAgent(input_dim=n_domains,
                              n_actions=n_domains,
                              device=device,
                              lr=lr,
                              gamma=gamma,
                              batch_size=batch_size,
                              buffer_size=buffer_size,
                              hidden=hidden,
                              eps_start=1.0,
                              eps_end=0.05,
                              eps_decay_steps=200_000,
                              target_sync_every=500)
    def select_action(self, state_p, eval_mode=False):
        # state_p: numpy array shape (n,)
        return self.agent.act(state_p, eval_mode=eval_mode)

    def observe(self, state_p, action, reward, next_state_p, done):
        # push to replay; next_state_p may be None
        self.agent.push_transition(state_p, action, reward, next_state_p, done)

    def train_step(self):
        return self.agent.update()

    def save(self, path):
        self.agent.save(path)

    def load(self, path):
        self.agent.load(path)
