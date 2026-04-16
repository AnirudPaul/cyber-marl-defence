"""
src/multi_agent.py

Implements a cooperative Multi-Agent DQN system.
Each agent is trained on its own domain split but can share replay experiences
with others to improve robustness.
"""

import random
import numpy as np
import torch
from collections import deque
from src.agent import DQNAgent
from src.environment import NetworkFlowEnv


class MultiAgentCoordinator:
    def __init__(self, splits, checkpoint_dir, device=None, share_rate=0.2):
        """
        splits: dict {agent_name: (X, y)}
        share_rate: fraction of replay buffer exchanged periodically
        """
        self.device = device
        self.agents = {}
        self.envs = {}
        self.buffers = {}
        self.share_rate = share_rate
        self.checkpoint_dir = checkpoint_dir

        for name, (X, y) in splits.items():
            print(f"[+] Initializing agent for {name} with {len(X)} samples")
            env = NetworkFlowEnv(X, y, rng_seed=42)
            agent = DQNAgent(input_dim=env.feature_size, n_actions=env.n_actions, device=device)
            self.agents[name] = agent
            self.envs[name] = env
            self.buffers[name] = deque(maxlen=5000)

    # ------------------------------------------------------------------
    def share_experiences(self):
        """Share random subset of replay memory across all agents."""
        all_exps = []
        for buf in self.buffers.values():
            n_share = int(len(buf) * self.share_rate)
            all_exps.extend(random.sample(buf, n_share) if n_share > 0 else [])
        for name, buf in self.buffers.items():
            buf.extend(random.sample(all_exps, min(len(all_exps), 100)))

    # ------------------------------------------------------------------
    def train_one_epoch(self, steps_per_agent=1000):
        """Each agent runs for given steps and learns; then experiences shared."""
        epoch_reward = {}
        for name, agent in self.agents.items():
            env = self.envs[name]
            s = env.reset(0)
            ep_reward = 0
            for _ in range(steps_per_agent):
                a = agent.act(s)
                ns, r, done, info = env.step(a)
                agent.push(s, a, r, ns, done)
                agent.learn()
                ep_reward += r
                s = ns if not done else env.reset(0)
            epoch_reward[name] = ep_reward / steps_per_agent
        # Cooperative exchange
        self.share_experiences()
        return epoch_reward

    # ------------------------------------------------------------------
    def save_all(self, epoch):
        for name, agent in self.agents.items():
            path = self.checkpoint_dir / f"marl_{name}_ep{epoch}.pth"
            torch.save(agent.policy_net.state_dict(), path)
            print(f"[sav] {name} checkpoint → {path}")
