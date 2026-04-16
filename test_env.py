# test_env.py (one-off)
import numpy as np
from pathlib import Path
from src.environment import NetworkFlowEnv

p = Path("data/processed/splits/monday.npz")
d = np.load(p, allow_pickle=True)
X = d['X']
y = d['y']
print("Loaded split:", p, "X.shape:", X.shape, "y.shape:", y.shape)

env = NetworkFlowEnv(X, y,
                     alpha=1.0, beta=1.0, delta=0.05,
                     inspect_detect_prob=0.9, inspect_fp_prob=0.01,
                     inspect_cost=0.02, rng_seed=42,
                     episode_length=10)

s = env.reset(0)
print("Initial state vector len:", len(s))

for i in range(10):
    a = env.sample_random_action()
    ns, r, done, info = env.step(a)
    print(f"step {i:02d} idx={info['idx']:6d} action={info['action_name']:7s} label={info['label']:2d} reward={r:.4f} detected={info['detected']}")
    if done:
        break

print("Metrics:", env.render_metrics())
