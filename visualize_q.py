#!/usr/bin/env python3
"""
Quick heatmap exporter for Q-tables (compatible with viewer's qtable format).
Usage: python visualize_q.py --qtable qtable.pkl --out q_heatmap.png
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--qtable", required=True)
ap.add_argument("--out", default="q_heatmap.png")
ap.add_argument("--w", type=int, default=10)
ap.add_argument("--h", type=int, default=10)
args = ap.parse_args()

with open(args.qtable, "rb") as f:
    q = pickle.load(f)

heat = np.zeros((args.h, args.w), dtype=float)
counts = np.zeros_like(heat, dtype=int)
for s, qvals in q.items():
    try:
        ax, ay = int(s[0]), int(s[1])
    except Exception:
        continue
    if 0 <= ay < args.h and 0 <= ax < args.w:
        heat[ay, ax] += max(qvals) if qvals else 0.0
        counts[ay, ax] += 1
avg = np.zeros_like(heat)
mask = counts > 0
avg[mask] = heat[mask] / counts[mask]

plt.figure(figsize=(6, 6))
plt.imshow(avg, origin='upper', cmap='viridis')
plt.colorbar(label='avg max Q')
plt.title('Q heatmap (avg max-Q by position)')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(args.out, dpi=150)
print('Saved', args.out)
