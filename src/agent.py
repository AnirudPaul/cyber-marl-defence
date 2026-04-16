"""
src/agent.py

Robust DQNAgent used by both single-agent training and multi-agent coordinator.

API / methods:
 - DQNAgent(input_dim, n_actions, device=None, **hyperparams)
 - act(state, eval_mode=False) -> int
 - push(state, action, reward, next_state, done)   # used by MARL coordinator
 - push_transition(state, action, reward, next_state, done)  # compatibility with train.py
 - update() -> loss or None   # performs one learning step from replay buffer
 - learn() -> alias for update()
 - sync_target()              # copy policy -> target
 - save(path) / load(path)
 - len(replay) via len(agent.replay)
"""

import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# --------------------------
# Simple DQN network (MLP)
# --------------------------
class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden=(256, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --------------------------
# Replay Buffer
# --------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# --------------------------
# DQN Agent
# --------------------------
class DQNAgent:
    def __init__(self,
                 input_dim: int,
                 n_actions: int,
                 device: str = None,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 batch_size: int = 128,
                 buffer_size: int = 200_000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_decay_steps: int = 500_000,
                 target_sync_every: int = 1000,
                 hidden=(256,128),
                 grad_clip: float = 10.0):
        """
        Unified DQNAgent for single-agent and multi-agent use.

        Parameters:
            input_dim, n_actions: shapes
            device: 'cuda' or 'cpu' or None (auto-detect)
            lr, gamma: optimizer and discount
            batch_size, buffer_size: replay buffer params
            epsilon schedule: eps_start -> eps_end over eps_decay_steps
            target_sync_every: steps between target updates (in update())
        """
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        # networks
        self.policy_net = DQNNetwork(input_dim, n_actions, hidden=hidden).to(self.device)
        self.target_net = DQNNetwork(input_dim, n_actions, hidden=hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # replay
        self.replay = ReplayBuffer(buffer_size)

        # epsilon
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = max(1, eps_decay_steps)
        self.eps_step = 0

        # bookkeeping
        self.learn_steps = 0
        self.target_sync_every = target_sync_every
        self.loss_fn = nn.MSELoss()

    # --------------------------
    # Action selection
    # --------------------------
    def epsilon(self) -> float:
        # linear decay
        frac = min(1.0, self.eps_step / float(self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        state: 1D numpy array (feature vector)
        eval_mode: greedy (no exploration)
        """
        if (not eval_mode) and (random.random() < self.epsilon()):
            return random.randrange(self.n_actions)
        self.policy_net.eval()
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(s)
        return int(q.argmax(dim=1).item())

    # --------------------------
    # Buffer interface (MARL uses push)
    # --------------------------
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition. Accepts numpy arrays or lists or None for next_state.
        """
        # store raw; conversion to tensors happens on sampling
        self.replay.push(state, int(action), float(reward),
                         None if next_state is None else np.array(next_state, dtype=np.float32),
                         bool(done))

    # compatibility wrapper expected by train.py
    def push_transition(self, state, action, reward, next_state, done):
        self.push(state, action, reward, next_state, done)

    # --------------------------
    # Learning step
    # --------------------------
    def update(self):
        """
        Sample a batch and perform a gradient step. Returns loss (float) or None if not enough samples.
        """
        if len(self.replay) < max(1000, self.batch_size):
            # wait until buffer has some data
            self.eps_step += 1
            return None

        batch = self.replay.sample(self.batch_size)
        states = np.stack(batch.state)
        actions = np.array(batch.action, dtype=np.int64)
        rewards = np.array(batch.reward, dtype=np.float32)
        # next_state may include None if terminal; handle using mask
        non_final_mask = np.array([s is not None for s in batch.next_state], dtype=bool)
        if non_final_mask.any():
            non_final_next = np.stack([s for s in batch.next_state if s is not None])
        else:
            non_final_next = None
        dones = np.array(batch.done, dtype=np.float32)

        # to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # next Q
        next_q = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
        if non_final_next is not None:
            next_states_t = torch.tensor(non_final_next, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_q_vals = self.target_net(next_states_t).max(dim=1)[0].unsqueeze(1)
            # scatter next_q_vals into positions where non_final_mask True
            it = 0
            for i, valid in enumerate(non_final_mask):
                if valid:
                    next_q[i] = next_q_vals[it]
                    it += 1

        expected_q = rewards_t + (1.0 - dones_t) * (self.gamma * next_q)

        loss = self.loss_fn(q_values, expected_q)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # bookkeeping
        self.learn_steps += 1
        self.eps_step += 1

        # target sync
        if self.target_sync_every and (self.learn_steps % self.target_sync_every == 0):
            self.sync_target()

        return float(loss.item())

    # alias
    learn = update

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # --------------------------
    # Persistence
    # --------------------------
    def save(self, path: str):
        payload = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eps_step': self.eps_step,
            'learn_steps': self.learn_steps,
        }
        torch.save(payload, path)

    def load(self, path: str):
        d = torch.load(path, map_location=self.device)
        if 'policy_state_dict' in d:
            self.policy_net.load_state_dict(d['policy_state_dict'])
            self.target_net.load_state_dict(d.get('target_state_dict', d['policy_state_dict']))
            if 'optimizer_state_dict' in d:
                try:
                    self.optimizer.load_state_dict(d['optimizer_state_dict'])
                except Exception:
                    # optimizer state may be incompatible across devices; ignore safely
                    pass
            self.eps_step = d.get('eps_step', 0)
            self.learn_steps = d.get('learn_steps', 0)
        else:
            # legacy: assume entire state dict saved
            self.policy_net.load_state_dict(d)
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # convenience
    def __len__(self):
        return len(self.replay)
