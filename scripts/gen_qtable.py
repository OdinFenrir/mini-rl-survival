from __future__ import annotations

import argparse
import random

from core.qlearning import QLearningAgent, QLearningConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a small random Q-table for demo/testing.")
    ap.add_argument("--out", type=str, default="data/qtable_random.pkl")
    ap.add_argument("--states", type=int, default=5000, help="how many random states to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--energy-min", type=int, default=1)
    ap.add_argument("--energy-max", type=int, default=40)
    ap.add_argument("--level-id", type=int, default=0)
    ap.add_argument("--goal-x", type=int, default=-1)
    ap.add_argument("--goal-y", type=int, default=-1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = QLearningConfig()
    agent = QLearningAgent(n_actions=4, cfg=cfg, seed=args.seed)

    gx = args.goal_x if args.goal_x >= 0 else args.w - 1
    gy = args.goal_y if args.goal_y >= 0 else args.h - 1
    for _ in range(args.states):
        ax = rng.randrange(args.w)
        ay = rng.randrange(args.h)
        fx = rng.randrange(args.w)
        fy = rng.randrange(args.h)
        energy = rng.randrange(args.energy_min, args.energy_max + 1)
        s = (args.level_id, ax, ay, fx, fy, gx, gy, energy)
        agent.Q[s] = [rng.uniform(-1.0, 1.0) for _ in range(agent.n_actions)]

    agent.save(args.out)
    print(f"Saved random Q-table with {len(agent.Q)} states to {args.out}")


if __name__ == "__main__":
    main()
