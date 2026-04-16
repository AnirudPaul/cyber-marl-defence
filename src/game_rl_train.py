"""
src/game_rl_train.py

Train an RL-based attacker against the LP-based defender Stackelberg leader.

Flow:
 - Use compute_stackelberg_lp(defender_rewards) to get defender mixed p and matrix M.
 - Attacker sees state = p, picks action a.
 - Attacker reward r_att = - defender_expected_payoff_for_a = - (p^T M[:, a])
 - Update attacker DQN from transitions.
 - Defender_rewards are adapted smoothly after each attack (decay + recovery).
 - Log attacker episodic returns and defender v-values over time.
"""

import os, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.game_theory import build_payoff_matrix, compute_stackelberg_lp
from src.attacker import RLAttacker

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPTS = RESULTS / "checkpoints"
PLOTS = RESULTS / "plots"
RESULTS.mkdir(parents=True, exist_ok=True)
CKPTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

def simulate_rl_attacker(epochs=1000, attacker_steps_per_epoch=10, device=None,
                         lr=1e-4, eval_every=100, seed=42):
    np.random.seed(seed)
    # initialize defender_rewards (use your MARL results as baseline)
    defender_rewards = {
        "monday": 0.982,
        "tuesday": 0.986,
        "wednesday": 0.988,
        "thursday": 0.984,
        "friday": 0.990,
    }
    domains = list(defender_rewards.keys())
    n = len(domains)

    attacker = RLAttacker(n_domains=n, device=device, lr=lr)

    history = {"epoch": [], "v": [], "attacker_return": []}

    for ep in range(1, epochs + 1):
        # compute defender leader strategy via LP (p, attacker best response indices, v)
        domains_lp, p, best_js, v = compute_stackelberg_lp(defender_rewards)
        # build payoff matrix M (rows defender, cols attacker)
        _, M = build_payoff_matrix(defender_rewards)

        # attacker performs multiple attack steps in this epoch (to collect transitions)
        total_attacker_reward = 0.0
        for step in range(attacker_steps_per_epoch):
            state = p.copy()  # attacker observes current defender mixed strategy
            action = attacker.select_action(state, eval_mode=False)
            # defender expected payoff for this pure attacker action:
            # vector of payoff when defender picks each row; attacker cost is column a
            def_payoff_for_action = M[:, action]  # shape (n,)
            # defender expected payoff under defender mixing p: p^T M[:, a]
            defender_expected = float(np.dot(p, M[:, action]))
            # attacker reward: negative defender expected payoff (attacker wants to minimize defender payoff)
            att_reward = -defender_expected + 0.1 * np.random.randn()
            total_attacker_reward += att_reward

            # Next state: simulate small defender update and recompute p' (we approximate by slight change)
            # For efficiency: we create a perturbed defender_rewards' to simulate adaptation
            # Smooth adaptation: attacked domain receives slight penalty, others recover
            attacked_domain = domains[action]
            next_defender_rewards = defender_rewards.copy()
            for d in next_defender_rewards:
                if d == attacked_domain:
                    next_defender_rewards[d] = max(0.95 * next_defender_rewards[d], 0.01)
                else:
                    next_defender_rewards[d] = min(next_defender_rewards[d] * 1.005, 1.0)
            # add tiny gaussian noise
            for d in next_defender_rewards:
                next_defender_rewards[d] += np.random.normal(0, 0.001)
                next_defender_rewards[d] = float(np.clip(next_defender_rewards[d], 0.01, 1.0))

            # compute new p' after this single adaptation (for next state)
            _, p_next, _, _ = compute_stackelberg_lp(next_defender_rewards)

            # push transition and train
            attacker.observe(state, action, att_reward, p_next, False)
            loss = attacker.train_step()

            # update the live defender_rewards to reflect adaptation for the next step in epoch
            defender_rewards = next_defender_rewards

            # update p for next iteration
            p = p_next

        # epoch bookkeeping
        history["epoch"].append(ep)
        history["v"].append(v)
        history["attacker_return"].append(total_attacker_reward)

        # periodic logging and evaluation
        if ep % 10 == 0:
            print(f"Epoch {ep:04d} | v={v:.4f} | attacker_return={total_attacker_reward:.4f} | eps={attacker.agent.epsilon():.4f}")
        if ep % eval_every == 0:
            # save attacker checkpoint and plot curves
            attacker.save(str(CKPTS / f"attacker_dqn_ep{ep}.pth"))
            # plot attacker returns
            plt.figure(figsize=(6,3))
            plt.plot(history["epoch"], history["attacker_return"])
            plt.title("Attacker episodic returns")
            plt.xlabel("epoch"); plt.ylabel("return")
            plt.tight_layout(); plt.savefig(PLOTS / f"attacker_returns_ep{ep}.png"); plt.close()

    # final save
    attacker.save(str(CKPTS / "attacker_dqn_final.pth"))
    # final plots
    plt.figure(figsize=(6,3))
    plt.plot(history["epoch"], history["attacker_return"])
    plt.title("Attacker episodic returns (final)")
    plt.xlabel("epoch"); plt.ylabel("return")
    plt.tight_layout(); plt.savefig(PLOTS / "attacker_returns_final.png"); plt.close()

    # payoff curve
    plt.figure(figsize=(6,3))
    plt.plot(history["epoch"], history["v"])
    plt.title("Defender guaranteed value (v) over time")
    plt.xlabel("epoch"); plt.ylabel("v")
    plt.tight_layout(); plt.savefig(PLOTS / "stackelberg_v_curve_rl.png"); plt.close()

    print("[✓] RL-attacker training complete. Checkpoints & plots saved in results/")
    return history

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--eval_every", type=int, default=100)
    args = ap.parse_args()
    simulate_rl_attacker(epochs=args.epochs, attacker_steps_per_epoch=args.steps, device=args.device, eval_every=args.eval_every)
