from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from core.qlearning import QLearningAgent


ACTION_LABELS = ["UP", "RIGHT", "DOWN", "LEFT"]


def _parse_state(state: Tuple[int, ...]) -> tuple[int, int, int, int, int, int, int, int]:
	if len(state) >= 8:
		level_id = int(state[0])
		ax, ay, fx, fy, gx, gy, energy = (int(x) for x in state[1:8])
		return level_id, ax, ay, fx, fy, gx, gy, energy
	if len(state) >= 2:
		ax = int(state[0])
		ay = int(state[1])
		energy = int(state[2]) if len(state) >= 3 else 0
		return 0, ax, ay, -1, -1, -1, -1, energy
	return 0, 0, 0, -1, -1, -1, -1, 0


def _state_features(state: Tuple[int, ...]) -> list[float]:
	level_id, ax, ay, fx, fy, gx, gy, energy = _parse_state(state)
	d_food = -1.0
	if fx >= 0 and fy >= 0:
		d_food = abs(ax - fx) + abs(ay - fy)
	d_goal = abs(ax - gx) + abs(ay - gy) if gx >= 0 and gy >= 0 else -1.0
	return [
		float(level_id),
		float(ax),
		float(ay),
		float(fx),
		float(fy),
		float(gx),
		float(gy),
		float(energy),
		float(d_food),
		float(d_goal),
	]


def _q_features(qvals: Iterable[float]) -> list[float]:
	return [float(v) for v in qvals]


def _standardize(X: np.ndarray) -> np.ndarray:
	if X.size == 0:
		return X
	mu = X.mean(axis=0, keepdims=True)
	sigma = X.std(axis=0, keepdims=True)
	sigma[sigma == 0] = 1.0
	return (X - mu) / sigma


def _pca3(X: np.ndarray) -> np.ndarray:
	if X.size == 0:
		return X
	if X.shape[1] < 3:
		pad = np.zeros((X.shape[0], 3 - X.shape[1]), dtype=float)
		X = np.hstack([X, pad])
	Xn = _standardize(X)
	_, _, vt = np.linalg.svd(Xn, full_matrices=False)
	components = vt[:3].T
	return Xn @ components


def _downsample(rng: random.Random, n: int, max_points: int) -> list[int]:
	if n <= max_points:
		return list(range(n))
	return rng.sample(range(n), max_points)


def main() -> int:
	parser = argparse.ArgumentParser(description="Plot 3D state/Q-space embedding from a Q-table.")
	parser.add_argument("--load", required=True, help="Path to Q-table (.pkl)")
	parser.add_argument("--mode", choices=["state", "qvec"], default="state", help="Embedding mode")
	parser.add_argument("--color", choices=["value", "action", "confidence", "energy", "level", "d_goal", "d_food", "visits"], default="value", help="Color scheme")
	parser.add_argument("--size", choices=["count", "value", "confidence", "energy", "none"], default="count", help="Point size")
	parser.add_argument("--level", default="all", help="Filter by level_id or 'all'")
	parser.add_argument("--level-feature", action="store_true", help="Include level_id as a PCA feature (default: off)")
	parser.add_argument("--min-visits", type=int, default=0, help="Only include states with N[s] >= this")
	parser.add_argument("--max-points", type=int, default=20000, help="Downsample to at most N points")
	parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
	parser.add_argument("--export", default="", help="Optional CSV export path")
	parser.add_argument("--highlight", type=int, default=50, help="Highlight top-N by value (0 disables)")
	parser.add_argument("--view", choices=["3d", "3d+2d"], default="3d+2d", help="Layout style")
	parser.add_argument("--action-figure", action="store_true", help="Open a second figure colored by best action")
	args = parser.parse_args()

	if not os.path.exists(args.load):
		raise SystemExit(f"File not found: {args.load}")

	agent = QLearningAgent.load(args.load)
	points = []
	meta = []
	features = []
	states = []
	total_states = 0
	level_filter = None
	if str(args.level).lower() not in ("all", "*", "-1"):
		level_filter = int(args.level)
	min_visits = max(0, int(args.min_visits))
	include_level_feature = bool(args.level_feature)
	for state, qvals in agent.Q.items():
		level_id, ax, ay, fx, fy, gx, gy, energy = _parse_state(state)
		visits = agent.N.get(state, 0)
		if level_filter is not None and level_id != level_filter:
			continue
		if visits < min_visits:
			continue
		if args.mode == "state":
			feat = _state_features(state) if include_level_feature else _state_features(state)[1:]
		else:
			feat = _q_features(qvals)
		points.append(feat)
		best = int(max(range(len(qvals)), key=lambda i: qvals[i])) if qvals else 0
		sorted_q = sorted(qvals) if qvals else [0.0]
		max_q = float(sorted_q[-1]) if sorted_q else 0.0
		second_q = float(sorted_q[-2]) if len(sorted_q) > 1 else float(sorted_q[-1])
		conf = max_q - second_q
		meta.append((max_q, best, conf, visits))
		states.append(state)
		if args.mode == "state":
			features.append((level_id, ax, ay, fx, fy, gx, gy, energy))
	total_states = len(points)

	if not points:
		raise SystemExit("No states after filters. Try --level all or --min-visits 0.")

	top_visit_lines = []
	if meta:
		vis_full = np.array([m[3] for m in meta], dtype=float)
		if vis_full.size and vis_full.max() > 0:
			top_idx = np.argsort(vis_full)[-5:][::-1]
			for idx in top_idx:
				level_id, ax, ay, fx, fy, gx, gy, energy = _parse_state(states[idx])
				label = f"L{level_id} ({ax},{ay}) E{energy} N{int(vis_full[idx])}"
				top_visit_lines.append(label)

	X = np.array(points, dtype=float)
	values = np.array([m[0] for m in meta], dtype=float)
	actions = np.array([m[1] for m in meta], dtype=int)
	conf = np.array([m[2] for m in meta], dtype=float)
	visits = np.array([m[3] for m in meta], dtype=float)
	if features:
		raw = np.array(features, dtype=float)
	else:
		raw = np.array([_parse_state(s) for s in states], dtype=float) if states else np.zeros((len(values), 8), dtype=float)
	if raw.size:
		level_ids = raw[:, 0]
		ax = raw[:, 1]
		ay = raw[:, 2]
		fx = raw[:, 3]
		fy = raw[:, 4]
		gx = raw[:, 5]
		gy = raw[:, 6]
		energy = raw[:, 7]
		has_food = (fx >= 0) & (fy >= 0)
		has_goal = (gx >= 0) & (gy >= 0)
		d_food = np.where(has_food, np.abs(ax - fx) + np.abs(ay - fy), -1.0)
		d_goal = np.where(has_goal, np.abs(ax - gx) + np.abs(ay - gy), -1.0)
	else:
		level_ids = np.zeros_like(values)
		energy = np.zeros_like(values)
		d_food = np.zeros_like(values)
		d_goal = np.zeros_like(values)

	rng = random.Random(args.seed)
	idx = _downsample(rng, len(points), int(args.max_points))
	X = X[idx]
	values = values[idx]
	actions = actions[idx]
	conf = conf[idx]
	visits = visits[idx]
	level_ids = level_ids[idx]
	energy = energy[idx]
	d_food = d_food[idx]
	d_goal = d_goal[idx]
	sampled = len(idx)

	X3 = _pca3(X)

	if args.color == "action":
		colors = actions
	elif args.color == "confidence":
		colors = conf
	elif args.color == "energy":
		colors = energy
	elif args.color == "level":
		colors = level_ids
	elif args.color == "d_goal":
		colors = d_goal
	elif args.color == "d_food":
		colors = d_food
	elif args.color == "visits":
		colors = visits
	else:
		colors = values

	if args.size == "none":
		sizes = np.full(X3.shape[0], 12.0)
	else:
		if args.size == "value":
			base = values
		elif args.size == "confidence":
			base = conf
		elif args.size == "energy":
			base = energy
		else:
			base = visits
		if base.size and base.max() > 0:
			base = base / base.max()
		sizes = 12.0 + 80.0 * base

	if args.export:
		os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
		header = [
			"pc1", "pc2", "pc3",
			"value", "action", "confidence", "visits",
		]
		data_parts = [X3, values, actions, conf, visits]
		if args.mode == "state":
			header += ["level", "energy", "d_goal", "d_food"]
			data_parts += [level_ids, energy, d_goal, d_food]
		data = np.column_stack(data_parts)
		np.savetxt(args.export, data, delimiter=",", header=",".join(header), comments="")

	def _fmt(x: float, digits: int = 2) -> str:
		if math.isnan(x):
			return "n/a"
		return f"{x:.{digits}f}"

	def _mean(arr: np.ndarray) -> float:
		return float(arr.mean()) if arr.size else float("nan")

	def _p90(arr: np.ndarray) -> float:
		return float(np.percentile(arr, 90)) if arr.size else float("nan")

	def _max(arr: np.ndarray) -> float:
		return float(arr.max()) if arr.size else float("nan")

	act_counts = np.bincount(actions, minlength=4)
	act_total = max(1, int(act_counts.sum()))
	act_pct = act_counts / act_total * 100.0
	goal_mask = d_goal >= 0
	food_mask = d_food >= 0
	energy_mask = energy >= 0

	unique_levels = len(np.unique(level_ids)) if level_ids.size else 0
	level_note = "all" if level_filter is None else str(level_filter)
	level_feature_note = "on" if include_level_feature else "off"
	summary_lines = [
		f"Each dot = 1 state; axes = PCA of {'state features' if args.mode == 'state' else 'Q-vectors'}",
		f"Filter: level {level_note} | min visits {min_visits} | max points {args.max_points} | level feature {level_feature_note}",
		f"Sample: {sampled}/{total_states} (seed {args.seed}) | levels in sample: {unique_levels}",
		f"Value: mean {_fmt(_mean(values))} | p90 {_fmt(_p90(values))} | max {_fmt(_max(values))}",
		f"Conf:  mean {_fmt(_mean(conf))} | p90 {_fmt(_p90(conf))} | max {_fmt(_max(conf))}",
		f"Visits mean {_fmt(_mean(visits), 1)} | max {_fmt(_max(visits), 0)}",
		f"Best action %: U {act_pct[0]:.0f} R {act_pct[1]:.0f} D {act_pct[2]:.0f} L {act_pct[3]:.0f}",
		"Terminal/timeout: n/a (not stored in Q-table)",
	]
	if args.mode == "state":
		if include_level_feature:
			summary_lines.append("Features: level, ax, ay, fx, fy, gx, gy, energy, d_food, d_goal")
		else:
			summary_lines.append("Features: ax, ay, fx, fy, gx, gy, energy, d_food, d_goal")
		if energy_mask.any():
			summary_lines.append(f"Energy: min {energy[energy_mask].min():.0f} | med {np.median(energy[energy_mask]):.0f} | max {energy[energy_mask].max():.0f}")
		if goal_mask.any():
			summary_lines.append(f"Dist→Goal: min {d_goal[goal_mask].min():.0f} | med {np.median(d_goal[goal_mask]):.0f}")
		if food_mask.any():
			summary_lines.append(f"Dist→Food: min {d_food[food_mask].min():.0f} | med {np.median(d_food[food_mask]):.0f}")
	if top_visit_lines:
		summary_lines.append("Top visits:")
		for line in top_visit_lines[:5]:
			summary_lines.append(f"  {line}")

	if args.view == "3d+2d":
		fig = plt.figure(figsize=(12, 8))
		gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0])
		ax = fig.add_subplot(gs[:, 0], projection="3d")
		ax12 = fig.add_subplot(gs[0, 1])
		ax13 = fig.add_subplot(gs[1, 1])
	else:
		fig = plt.figure(figsize=(9, 7))
		ax = fig.add_subplot(111, projection="3d")
		ax12 = None
		ax13 = None

	cmap = "tab10" if args.color == "action" else ("tab20" if args.color == "level" else "viridis")
	sc = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=colors, s=sizes, cmap=cmap, alpha=0.85, linewidths=0.2)
	if args.color == "action":
		handles = []
		for i, label in enumerate(ACTION_LABELS):
			handles.append(plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=plt.cm.tab10(i), markersize=8))
		ax.legend(handles=handles, title="Best action", loc="upper right")
	else:
		fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)

	if args.highlight > 0 and values.size:
		hi = np.argsort(values)[-min(args.highlight, len(values)) :]
		ax.scatter(X3[hi, 0], X3[hi, 1], X3[hi, 2], s=70, c="none", edgecolors="white", linewidths=0.6)

	if ax12 is not None and ax13 is not None:
		ax12.scatter(X3[:, 0], X3[:, 1], c=colors, s=10, cmap=cmap, alpha=0.8, linewidths=0.0)
		ax12.set_xlabel("PC1")
		ax12.set_ylabel("PC2")
		ax12.set_title("PC1 vs PC2")
		ax12.grid(True, linestyle=":", alpha=0.3)

		ax13.scatter(X3[:, 0], X3[:, 2], c=colors, s=10, cmap=cmap, alpha=0.8, linewidths=0.0)
		ax13.set_xlabel("PC1")
		ax13.set_ylabel("PC3")
		ax13.set_title("PC1 vs PC3")
		ax13.grid(True, linestyle=":", alpha=0.3)

	mode_name = "State features" if args.mode == "state" else "Q-vector"
	title = f"3D {mode_name} embedding ({X3.shape[0]} points)"
	ax.set_title(title)
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	ax.grid(True, linestyle=":", alpha=0.3)
	info = f"Color: {args.color} | Size: {args.size} | Points: {X3.shape[0]}"
	text_block = info + "\n" + "\n".join(summary_lines)
	fig.text(
		0.02,
		0.02,
		text_block,
		fontsize=9,
		color="#555555",
		family="monospace",
		bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#dddddd", alpha=0.9),
	)
	plt.tight_layout()

	if args.action_figure and actions.size:
		fig2 = plt.figure(figsize=(8, 6))
		ax2 = fig2.add_subplot(111, projection="3d")
		sc2 = ax2.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=actions, s=sizes, cmap="tab10", alpha=0.85, linewidths=0.2)
		handles = []
		for i, label in enumerate(ACTION_LABELS):
			handles.append(plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=plt.cm.tab10(i), markersize=8))
		ax2.legend(handles=handles, title="Best action", loc="upper right")
		ax2.set_title("Best-action clusters (categorical)")
		ax2.set_xlabel("PC1")
		ax2.set_ylabel("PC2")
		ax2.set_zlabel("PC3")
		ax2.grid(True, linestyle=":", alpha=0.3)

	plt.show()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
