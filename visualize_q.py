#!/usr/bin/env python3
"""
Visualize Q-table produced by QLearningAgent.save(...).

Usage:
    python visualize_q.py --qtable qtable.pkl --out q_heatmap.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend so saving works on Windows/servers
from qlearn import QLearningAgent
from env import GridSurvivalEnv
import viz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qtable", required=True, help="path to saved Q-table (qtable.pkl)")
    ap.add_argument("--out", default="q_heatmap.png", help="output PNG filename")
    ap.add_argument("--w", type=int, default=10, help="grid width (must match trained env)")
    ap.add_argument("--h", type=int, default=10, help="grid height (must match trained env)")
    ap.add_argument("--seed", type=int, default=0, help="seed used when saving (if relevant)")
    args = ap.parse_args()

    # load agent (QLearningAgent.load should return a QLearningAgent instance)
    agent = QLearningAgent.load(args.qtable, seed=args.seed)

    env = GridSurvivalEnv(width=args.w, height=args.h, seed=args.seed)
    print(f"Saving heatmap to {args.out} ...")
    viz.q_heatmap(agent, env, out=args.out)
    print("Done.")


if __name__ == "__main__":
    main()