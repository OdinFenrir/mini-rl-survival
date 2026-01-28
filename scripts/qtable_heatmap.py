from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from core.qlearning import QLearningAgent


def compute_heatmap(agent: QLearningAgent, w: int, h: int, level_id: int | None = None) -> np.ndarray:
    heat = np.zeros((h, w), dtype=float)
    cnt = np.zeros((h, w), dtype=int)
    for s, qvals in agent.Q.items():
        try:
            if len(s) >= 8:
                lvl = int(s[0])
                ax, ay = int(s[1]), int(s[2])
                if level_id is not None and lvl != level_id:
                    continue
            else:
                ax, ay = int(s[0]), int(s[1])
        except Exception:
            continue
        if 0 <= ax < w and 0 <= ay < h and qvals:
            heat[ay, ax] += float(max(qvals))
            cnt[ay, ax] += 1
    out = np.zeros_like(heat)
    mask = cnt > 0
    out[mask] = heat[mask] / cnt[mask]
    mn = float(out.min()) if out.size else 0.0
    mx = float(out.max()) if out.size else 0.0
    if mx > mn:
        out = (out - mn) / (mx - mn)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a Q-table heatmap image (by agent position).")
    ap.add_argument("--qtable", type=str, required=True, help="path to Q-table pickle")
    ap.add_argument("--out", type=str, default="data/q_heatmap.png")
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--level-id", type=int, default=-1, help="level id to filter (>=0), else all")
    args = ap.parse_args()

    agent = QLearningAgent.load(args.qtable)
    lvl = args.level_id if args.level_id >= 0 else None
    hm = compute_heatmap(agent, args.w, args.h, level_id=lvl)

    plt.figure()
    plt.imshow(hm, interpolation="nearest")
    plt.title("Q-table heatmap (avg max Q by agent cell)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"Saved heatmap to {args.out}")


if __name__ == "__main__":
    main()
