from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from typing import Tuple

from core.env import GridSurvivalEnv
from core.qlearning import QLearningAgent, QLearningConfig
from core import viz

def run_episode(env: GridSurvivalEnv, agent: QLearningAgent, max_steps: int, train: bool) -> Tuple[int, int, float, str]:
	"""Returns (steps_survived, foods_eaten, total_reward, terminal_reason)."""
	obs = env.reset()
	total_reward = 0.0
	foods = 0
	terminal = ""
	done = False

	for _ in range(max_steps):
		a = agent.act(obs, greedy=not train)
		res = env.step(a)

		if train:
			agent.learn(obs, a, res.reward, res.obs, res.done)

		total_reward += res.reward
		if res.info.get("got_food"):
			foods += 1

		obs = res.obs
		if res.done:
			terminal = res.info.get("terminal","")
			done = True
			break

	if not done:
		terminal = "timeout"
	return env.steps, foods, total_reward, terminal

def eval_stats(env: GridSurvivalEnv, agent: QLearningAgent, episodes: int, max_steps: int) -> str:
	steps_list = []
	foods_list = []
	rewards_list = []
	terminals = {} 

	for _ in range(episodes):
		steps, foods, total, term = run_episode(env, agent, max_steps=max_steps, train=False)
		steps_list.append(steps)
		foods_list.append(foods)
		rewards_list.append(total)
		terminals[term] = terminals.get(term, 0) + 1

	return (
		f"eval episodes={episodes} "
		f"avg_steps={statistics.mean(steps_list):.1f} "
		f"avg_foods={statistics.mean(foods_list):.2f} "
		f"avg_reward={statistics.mean(rewards_list):.2f} "
		f"terminals={terminals}"
	)

def eval_all_maps(agent: QLearningAgent, base_env: GridSurvivalEnv, episodes: int, max_steps: int, seed_base: int) -> list[dict]:
	rows = []
	total = GridSurvivalEnv.preset_level_count()
	for level_id in range(total):
		env = GridSurvivalEnv(
			width=base_env.base_width,
			height=base_env.base_height,
			n_hazards=base_env.n_hazards,
			energy_start=base_env.energy_start,
			energy_food_gain=base_env.energy_food_gain,
			energy_step_cost=base_env.energy_step_cost,
			energy_max=int(base_env.energy_max or 0),
			seed=int(seed_base + level_id * 10007),
			level_mode="preset",
			level_index=level_id,
			level_cycle=False,
			n_walls=base_env.n_walls,
			n_traps=base_env.n_traps,
			food_enabled=base_env.food_enabled,
		)
		steps_list = []
		foods_list = []
		rewards_list = []
		goals = 0
		for _ in range(episodes):
			steps, foods, total_reward, term = run_episode(env, agent, max_steps=max_steps, train=False)
			steps_list.append(steps)
			foods_list.append(foods)
			rewards_list.append(total_reward)
			if term == "goal":
				goals += 1
		template = GridSurvivalEnv.get_level_template(level_id) or {}
		rows.append(
			{
				"level_id": level_id,
				"level_index": level_id + 1,
				"name": template.get("name", f"Level {level_id + 1}"),
				"episodes": episodes,
				"plays": episodes,
				"goals": goals,
				"avg_steps": statistics.mean(steps_list) if steps_list else 0.0,
				"avg_foods": statistics.mean(foods_list) if foods_list else 0.0,
				"avg_reward": statistics.mean(rewards_list) if rewards_list else 0.0,
				"success_rate": goals / max(1, episodes),
			}
		)
	return rows

def _write_eval_all(rows: list[dict], path: str) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(rows, f, indent=2)
	csv_path = os.path.splitext(path)[0] + ".csv"
	if rows:
		keys = list(rows[0].keys())
		with open(csv_path, "w", encoding="utf-8", newline="") as f:
			w = csv.DictWriter(f, fieldnames=keys)
			w.writeheader()
			for r in rows:
				w.writerow(r)

def play(env: GridSurvivalEnv, agent: QLearningAgent, max_steps: int, sleep_s: float) -> None:
	obs = env.reset()
	total = 0.0
	foods = 0
	for t in range(max_steps):
		a = agent.act(obs, greedy=True)
		res = env.step(a)
		total += res.reward
		if res.info.get("got_food"):
			foods += 1

		obs = res.obs
		# use colored renderer from viz.py for nicer terminal output
		print(viz.render_color(env))
		print(f"t={t:03d} energy={obs[-1]:02d} foods={foods:02d} total_reward={total:.2f}")
		print("-" * (2 * env.width - 1))
		time.sleep(sleep_s)

		if res.done:
			print("DONE:", res.info.get("terminal", ""))
			break

def main() -> None:
	ap = argparse.ArgumentParser(description="Mini RL Survival (Tabular Q-learning, kid-friendly).")
	ap.add_argument("--episodes", type=int, default=2000, help="training episodes")
	ap.add_argument("--max-steps", type=int, default=400, help="max steps per episode")
	ap.add_argument("--eval-every", type=int, default=200, help="run evaluation every N episodes (0 disables)")
	ap.add_argument("--eval-episodes", type=int, default=50, help="how many eval episodes to average")
	ap.add_argument("--seed", type=int, default=0, help="random seed")
	ap.add_argument("--save", type=str, default="qtable.pkl", help="where to save the learned Q-table")
	ap.add_argument("--load", type=str, default="", help="load an existing Q-table to continue training or to play")
	ap.add_argument("--play", action="store_true", help="watch the greedy agent play (no training)")
	ap.add_argument("--sleep", type=float, default=0.03, help="sleep between frames in --play")
	ap.add_argument("--checkpoint-every", type=int, default=0, help="save Q-table every N episodes (0 disables)")

	# env knobs
	ap.add_argument("--w", type=int, default=12, help="grid width")
	ap.add_argument("--h", type=int, default=12, help="grid height")
	ap.add_argument("--hazards", type=int, default=0, help="number of traps")
	ap.add_argument("--energy-start", type=int, default=25, help="starting energy")
	ap.add_argument("--energy-food", type=int, default=18, help="energy gained on food")
	ap.add_argument("--energy-step", type=int, default=1, help="energy lost per step")
	ap.add_argument("--energy-max", type=int, default=0, help="maximum energy (0 = unlimited)")
	ap.add_argument("--level-mode", type=str, default="preset", help="preset or random")
	ap.add_argument("--level-index", type=int, default=0, help="level index for preset mode")
	ap.add_argument("--no-level-cycle", action="store_true", help="disable preset level cycling")
	ap.add_argument("--level-limit", type=int, default=0, help="limit preset levels to first N")
	ap.add_argument("--walls", type=int, default=18, help="number of walls for random levels")
	ap.add_argument("--traps", type=int, default=0, help="number of traps for random levels")
	ap.add_argument("--no-food", action="store_true", help="disable bonus food spawns")
	ap.add_argument("--curriculum", action="store_true", help="enable curriculum over preset levels")
	ap.add_argument("--curriculum-start", type=int, default=5, help="starting number of levels")
	ap.add_argument("--curriculum-step", type=int, default=5, help="levels added when threshold is met")
	ap.add_argument("--curriculum-window", type=int, default=50, help="episodes window for success rate")
	ap.add_argument("--curriculum-threshold", type=float, default=0.8, help="success rate to unlock more levels")
	ap.add_argument("--curriculum-eps-rewind", type=float, default=0.5, help="rewind epsilon progress on unlock (0-1)")
	ap.add_argument("--eval-all-maps", action="store_true", help="run per-map evaluation after training")
	ap.add_argument("--eval-all-episodes", type=int, default=20, help="episodes per map for eval-all")
	ap.add_argument("--eval-all-out", type=str, default="", help="write eval-all stats to JSON (CSV alongside)")

	# q-learning knobs
	ap.add_argument("--alpha", type=float, default=0.25, help="learning rate")
	ap.add_argument("--gamma", type=float, default=0.98, help="discount")
	ap.add_argument("--eps-start", type=float, default=1.0, help="starting epsilon")
	ap.add_argument("--eps-end", type=float, default=0.05, help="final epsilon")
	ap.add_argument("--eps-decay", type=int, default=30_000, help="epsilon decay steps")

	args = ap.parse_args()

	total_levels = GridSurvivalEnv.preset_level_count()
	level_limit = int(args.level_limit) if int(args.level_limit) > 0 else None
	curriculum = bool(args.curriculum) and args.level_mode == "preset" and total_levels > 0
	if curriculum:
		level_limit = max(1, min(int(args.curriculum_start), total_levels))

	env = GridSurvivalEnv(
		width=args.w,
		height=args.h,
		n_hazards=args.hazards,
		energy_start=args.energy_start,
		energy_food_gain=args.energy_food,
		energy_step_cost=args.energy_step,
		energy_max=args.energy_max,
		seed=args.seed,
		level_mode=args.level_mode,
		level_index=args.level_index,
		level_cycle=not args.no_level_cycle,
		level_limit=level_limit,
		n_walls=args.walls,
		n_traps=args.traps,
		food_enabled=not args.no_food,
	)

	cfg = QLearningConfig(
		alpha=args.alpha,
		gamma=args.gamma,
		eps_start=args.eps_start,
		eps_end=args.eps_end,
		eps_decay_steps=args.eps_decay,
	)

	if args.load:
		agent = QLearningAgent.load(args.load, seed=args.seed)
	else:
		agent = QLearningAgent(n_actions=4, cfg=cfg, seed=args.seed)

	if args.play:
		play(env, agent, max_steps=args.max_steps, sleep_s=args.sleep)
		return

	recent_success = []
	window = max(1, int(args.curriculum_window))
	for ep in range(1, args.episodes + 1):
		steps, foods, total, terminal = run_episode(env, agent, max_steps=args.max_steps, train=True)
		if args.checkpoint_every and ep % int(args.checkpoint_every) == 0:
			if args.save:
				agent.save(args.save)
				print(f"[ep {ep}] checkpoint -> {args.save}")
		if curriculum:
			recent_success.append(1 if terminal == "goal" else 0)
			if len(recent_success) > window:
				recent_success = recent_success[-window:]
			if len(recent_success) >= window:
				rate = sum(recent_success) / len(recent_success)
				if rate >= float(args.curriculum_threshold) and level_limit is not None and level_limit < total_levels:
					level_limit = min(total_levels, level_limit + max(1, int(args.curriculum_step)))
					env.level_limit = level_limit
					recent_success.clear()
					rewind = max(0.0, min(1.0, float(args.curriculum_eps_rewind)))
					if rewind < 1.0:
						agent.total_steps = int(agent.total_steps * rewind)
					print(f"[ep {ep}] curriculum unlocked -> levels {level_limit}/{total_levels}")

		if args.eval_every and ep % args.eval_every == 0:
			print(f"[ep {ep}] {eval_stats(env, agent, args.eval_episodes, args.max_steps)}")

	if args.save:
		agent.save(args.save)
		print(f"Saved Q-table to {args.save}")

	if args.eval_all_maps:
		rows = eval_all_maps(agent, env, episodes=int(args.eval_all_episodes), max_steps=args.max_steps, seed_base=int(args.seed))
		avg_success = sum(r["success_rate"] for r in rows) / max(1, len(rows))
		print(f"eval-all avg_success={avg_success * 100:.1f}% over {len(rows)} maps")
		if args.eval_all_out:
			_write_eval_all(rows, args.eval_all_out)
			print(f"Wrote eval-all stats -> {args.eval_all_out}")

if __name__ == "__main__":
	main()
