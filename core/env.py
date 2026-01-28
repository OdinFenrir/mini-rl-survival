from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
from typing import Dict, Tuple, List, Any

Action = int  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT


@dataclass
class StepResult:
	obs: Tuple[int, ...]  # (level_id, ax, ay, fx, fy, gx, gy, energy)
	reward: float
	done: bool
	info: Dict


FALLBACK_LEVEL_TEMPLATES: list[Dict[str, Any]] = [
	{
		"name": "AIMA 4x3 Maze",
		"desc": "Classic 4x3 maze (goal +, pit -).",
		"source": "Russell & Norvig, _Artificial Intelligence: A Modern Approach_ (Ch. 17)",
		"layout": [
			"####",
			"#  +",
			"# # ",
			"#S F",
		],
	},
	{
		"name": "MiniGrid FourRooms",
		"desc": "Partitioned rooms with doorways (MiniGrid FourRooms).",
		"source": "MiniGrid FourRooms (MiniGrid OpenAI)",
		"layout": [
			"############",
			"#S..#......G#",
			"#...#......#",
			"#...#.###..#",
			"##########.#",
			"#F.....#...#",
			"#.######.###",
			"#...#.....#",
			"#.#.#.###..#",
			"#.#.#......#",
			"#...#.#....#",
			"############",
		],
	},
	{
		"name": "DoorKey Hall",
		"desc": "DoorKey-style corridor with bonus food.",
		"source": "MiniGrid DoorKey (MiniGrid OpenAI)",
		"layout": [
			"############",
			"#S..#...#..#",
			"#..##...#..#",
			"#..##...#..#",
			"#....####..#",
			"#..F#D#...G#",
			"#..##D###..#",
			"#....#......#",
			"#.######...#",
			"#........#.#",
			"#...####...#",
			"############",
		],
	},
]


def _load_level_pack() -> list[Dict[str, Any]]:
	path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "levels", "maze_pack.json"))
	try:
		with open(path, "r", encoding="utf-8") as f:
			payload = json.load(f)
		levels = payload.get("levels", [])
		if not isinstance(levels, list) or not levels:
			return FALLBACK_LEVEL_TEMPLATES
		return levels
	except Exception:
		return FALLBACK_LEVEL_TEMPLATES


LEVEL_TEMPLATES: list[Dict[str, Any]] = _load_level_pack()


class GridSurvivalEnv:
	"""
	Small grid survival environment (kid-friendly).

	Observation (state):
	  (level_id, agent_x, agent_y, food_x, food_y, goal_x, goal_y, energy)

	Actions:
	  0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

	Rewards:
	  +25  when you reach the goal (after food is collected)
	  +6   when you reach food (and gain energy)
	  -25  if you hit a trap (optional mode)
	  -0.02 per step (living reward)
	  -0.25 if you bump a wall
	  -1   if you starve

	Episodes end when the agent hits the goal (after food), a trap (optional), or runs out of energy.
	"""

	def __init__(
		self,
		width: int = 12,
		height: int = 12,
		n_hazards: int = 8,
		energy_start: int = 25,
		energy_food_gain: int = 18,
		energy_step_cost: int = 1,
		energy_max: int | None = None,
		reward_shaping: str = "distance",
		shaping_strength: float = 0.05,
		seed: int | None = None,
		level_mode: str = "preset",
		level_index: int = 0,
		level_cycle: bool = True,
		level_limit: int | None = None,
		n_walls: int = 18,
		n_traps: int | None = None,
		food_enabled: bool = True,
	) -> None:
		self.base_width = width
		self.base_height = height
		self.width = width
		self.height = height
		self.n_traps = int(n_traps) if n_traps is not None else int(n_hazards)
		self.n_hazards = self.n_traps
		self.n_walls = int(n_walls)
		self.energy_start = energy_start
		self.energy_food_gain = energy_food_gain
		self.energy_step_cost = energy_step_cost
		if energy_max is None or int(energy_max) <= 0:
			self.energy_max = None
		else:
			self.energy_max = max(int(energy_max), int(energy_start))
		self.level_mode = str(level_mode or "preset").lower()
		self.level_index = int(level_index)
		self.level_cycle = bool(level_cycle)
		if level_limit is None or int(level_limit) <= 0:
			self.level_limit = None
		else:
			self.level_limit = int(level_limit)
		self.food_enabled = bool(food_enabled)
		self.food_collected = False
		self.reward_shaping = str(reward_shaping)
		self.shaping_strength = float(shaping_strength)
		self._dist_to_food: Dict[Tuple[int, int], int] | None = None
		self._dist_to_goal: Dict[Tuple[int, int], int] | None = None

		self._rng = random.Random(seed)

		self.agent = (0, 0)
		self.food = (0, 0)
		self.hazards: List[Tuple[int, int]] = []
		self.walls: List[Tuple[int, int]] = []
		self.goal = (0, 0)
		self.level_id = 0
		self.level_name = ""
		self.level_desc = ""
		self.level_source = ""
		self.level_style = ""
		self._food_ok = True
		self.episode = 0
		self.energy = self.energy_start
		if self.energy_max is not None:
			self.energy = min(self.energy, self.energy_max)
		self.steps = 0
		self._last_step_snapshot: Dict[str, Any] | None = None

		self.goal_reward = 25.0
		self.food_reward = 6.0
		self.trap_penalty = -25.0
		self.wall_penalty = -0.25
		self.step_penalty = -0.02
		self.starve_penalty = -1.0

	def seed(self, seed: int) -> None:
		self._rng.seed(seed)

	def _reachable_distances(self, start: Tuple[int, int], blocked: set) -> Dict[Tuple[int, int], int]:
		queue = [start]
		dist = {start: 0}
		while queue:
			x, y = queue.pop(0)
			for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
				nx, ny = x + dx, y + dy
				if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
					continue
				if (nx, ny) in blocked or (nx, ny) in dist:
					continue
				dist[(nx, ny)] = dist[(x, y)] + 1
				queue.append((nx, ny))
		return dist

	def _style_from_walls(self) -> str:
		total = max(1, self.width * self.height)
		walls = len(self.walls)
		ratio = walls / total
		if ratio < 0.28:
			return "Open paths"
		if ratio < 0.42:
			return "Balanced maze"
		return "Tight corridors"

	def _place_food(self, start: Tuple[int, int], goal: Tuple[int, int], blocked: set, deterministic: bool) -> bool:
		if not self.food_enabled:
			self.food = (-1, -1)
			self.food_collected = True
			return True
		dist_start = self._reachable_distances(start, blocked)
		dist_goal = self._reachable_distances(goal, blocked)
		candidates = []
		for pos, d1 in dist_start.items():
			if pos in {start, goal}:
				continue
			d2 = dist_goal.get(pos)
			if d2 is None:
				continue
			if d1 > self.energy_start:
				continue
			energy_after_food = self.energy_start - d1 + self.energy_food_gain
			if self.energy_max is not None:
				energy_after_food = min(energy_after_food, self.energy_max)
			if d2 > energy_after_food:
				continue
			candidates.append((pos, d1, d2))
		if not candidates:
			# No valid food cell: treat as already collected to avoid soft-lock.
			self.food = (-1, -1)
			self.food_collected = False
			return False
		if deterministic:
			dist_goal = dist_start.get(goal)
			if dist_goal is None:
				min_dist = 4
			else:
				min_dist = max(4, min(10, dist_goal + 2))
			filtered = [c for c in candidates if c[1] >= min_dist]
			if not filtered:
				filtered = candidates
			choice = min(filtered, key=lambda item: (item[1], item[2], item[0]))[0]
		else:
			choice = candidates[self._rng.randrange(len(candidates))][0]
		self.food = choice
		self.food_collected = False
		return True

	@staticmethod
	def preset_level_count() -> int:
		return len(LEVEL_TEMPLATES)

	@staticmethod
	def get_level_template(index: int) -> Dict[str, Any] | None:
		if not LEVEL_TEMPLATES:
			return None
		return LEVEL_TEMPLATES[index % len(LEVEL_TEMPLATES)]

	def _preset_levels(self) -> list[Dict[str, Any]]:
		return LEVEL_TEMPLATES

	def _parse_layout(self, layout: list[str]) -> Dict[str, Any]:
		if not layout:
			return {}
		width = max(len(line) for line in layout)
		height = len(layout)
		walls: list[Tuple[int, int]] = []
		traps: list[Tuple[int, int]] = []
		start: Tuple[int, int] | None = None
		goal: Tuple[int, int] | None = None
		food: Tuple[int, int] | None = None
		for y, raw in enumerate(layout):
			# Pad with walls to avoid unintended open cells from ragged lines.
			line = raw.ljust(width, "#")
			for x, ch in enumerate(line):
				if ch in ("#", "D"):
					walls.append((x, y))
				elif ch in ("-", "T"):
					traps.append((x, y))
				elif ch in ("G", "+"):
					goal = (x, y)
				elif ch == "S":
					start = (x, y)
				elif ch == "F":
					food = (x, y)
		return {
			"width": width,
			"height": height,
			"walls": walls,
			"traps": traps,
			"start": start,
			"goal": goal,
			"food": food,
		}

	def _apply_level(self, level: Dict[str, Any]) -> None:
		layout = level.get("layout", [])
		if layout:
			parsed = self._parse_layout(layout)
			self.width = parsed.get("width", self.base_width)
			self.height = parsed.get("height", self.base_height)
			self.walls = parsed.get("walls", [])
			self.hazards = parsed.get("traps", [])
			start = parsed.get("start")
			self.goal = parsed.get("goal") or (self.width - 1, self.height - 1)
			if not start:
				start = self._random_empty_cell(set(self.walls) | set(self.hazards) | {self.goal}, self.width, self.height)
			self.agent = start
			blocked = set(self.walls) | set(self.hazards)
			self._food_ok = self._place_food(self.agent, self.goal, blocked, deterministic=True)
		else:
			self.walls = [tuple(p) for p in level.get("walls", [])]
			self.hazards = [tuple(p) for p in level.get("traps", level.get("hazards", []))]
			self.goal = tuple(level.get("goal", (self.width - 1, self.height - 1)))
			start = level.get("start")
			occupied = set(self.walls) | set(self.hazards) | {self.goal}
			if start is None or start in occupied:
				start = self._random_empty_cell(occupied, self.width, self.height)
			self.agent = tuple(start)
			blocked = set(self.walls) | set(self.hazards)
			self._food_ok = self._place_food(self.agent, self.goal, blocked, deterministic=True)

		self.level_name = level.get("name", "Preset level")
		self.level_desc = level.get("desc", "")
		self.level_source = level.get("source", "")
		self.level_style = level.get("style") or self._style_from_walls()

	def _random_empty_cell(self, occupied: set, width: int, height: int) -> Tuple[int, int]:
		while True:
			x = self._rng.randrange(width)
			y = self._rng.randrange(height)
			if (x, y) not in occupied:
				return (x, y)

	def _path_exists(self, start: Tuple[int, int], goal: Tuple[int, int], blocked: set) -> bool:
		if start == goal:
			return True
		queue = [start]
		seen = {start}
		while queue:
			x, y = queue.pop(0)
			for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
				nx, ny = x + dx, y + dy
				if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
					continue
				if (nx, ny) in blocked or (nx, ny) in seen:
					continue
				if (nx, ny) == goal:
					return True
				seen.add((nx, ny))
				queue.append((nx, ny))
		return False

	def _generate_random_level(self) -> None:
		self.width = self.base_width
		self.height = self.base_height
		total_cells = max(1, self.width * self.height)
		target_walls = max(0, min(self.n_walls, total_cells - 3))
		target_traps = max(0, min(self.n_traps, total_cells - target_walls - 3))

		self.level_name = "Random level"
		self.level_desc = "Generated layout"
		self.level_source = ""

		for _ in range(80):
			occupied: set = set()
			goal = self._random_empty_cell(occupied, self.width, self.height)
			occupied.add(goal)

			walls: set = set()
			while len(walls) < target_walls:
				p = self._random_empty_cell(occupied | walls, self.width, self.height)
				walls.add(p)

			traps: set = set()
			while len(traps) < target_traps:
				p = self._random_empty_cell(occupied | walls | traps, self.width, self.height)
				traps.add(p)

			start = self._random_empty_cell(occupied | walls | traps, self.width, self.height)
			blocked = set(walls) | set(traps)
			if self._path_exists(start, goal, blocked):
				self.walls = list(walls)
				self.hazards = list(traps)
				self.goal = goal
				self.agent = start
				self.level_id = int(self._rng.randrange(1_000_000))
				self._food_ok = self._place_food(self.agent, self.goal, blocked, deterministic=False)
				if self._food_ok:
					break
		else:
			self.walls = []
			self.hazards = []
			self.goal = (self.width - 1, self.height - 1)
			self.agent = (0, 0)
			self.level_id = int(self._rng.randrange(1_000_000))
			self._food_ok = self._place_food(self.agent, self.goal, set(), deterministic=False)
		self.level_style = self._style_from_walls()

	def reset(self) -> Tuple[int, ...]:
		self.steps = 0
		self.energy = self.energy_start
		if self.energy_max is not None:
			self.energy = min(self.energy, self.energy_max)
		self.episode += 1

		self._last_step_snapshot = None
		self.food_collected = False

		levels = self._preset_levels()
		if self.level_limit:
			levels = levels[:self.level_limit]
		use_preset = self.level_mode == "preset" and bool(levels)
		if use_preset:
			if self.level_cycle:
				idx = (self.episode - 1) % len(levels)
			else:
				idx = max(0, min(self.level_index, len(levels) - 1))
			applied = False
			for attempt in range(len(levels) if self.level_cycle else 1):
				self.level_id = (idx + attempt) % len(levels) if self.level_cycle else idx
				self._apply_level(levels[self.level_id])
				blocked = set(self.walls) | set(self.hazards)
				ok_path = self._path_exists(self.agent, self.goal, blocked)
				ok_food = (not self.food_enabled) or bool(self._food_ok)
				if ok_path and ok_food:
					applied = True
					break
			if not applied:
				self._generate_random_level()
		else:
			self._generate_random_level()

		self.n_hazards = len(self.hazards)
		blocked = set(self.walls) | set(self.hazards)
		self._dist_to_goal = self._reachable_distances(self.goal, blocked)
		if self.food_enabled and (not self.food_collected) and self.food[0] >= 0:
			self._dist_to_food = self._reachable_distances(self.food, blocked)
		else:
			self._dist_to_food = None
		return self._obs()

	def _capture_pre_step_snapshot(self) -> None:
		self._last_step_snapshot = {
			'agent': self.agent,
			'food': self.food,
			'food_collected': self.food_collected,
			'hazards': list(self.hazards),
			'walls': list(self.walls),
			'goal': self.goal,
			'energy': self.energy,
			'steps': self.steps,
			'level_id': self.level_id,
			'episode': self.episode,
			'rng_state': self._rng.getstate(),
		}

	def _spawn_food(self, occupied: set) -> Tuple[int, int]:
		while True:
			fx = self._rng.randrange(self.width)
			fy = self._rng.randrange(self.height)
			if (fx, fy) not in occupied:
				return (fx, fy)

	def _distance_to_target(self, pos: Tuple[int, int]) -> int | None:
		if self.reward_shaping != "distance":
			return None
		dist_map = None
		if self.food_enabled and not self.food_collected and self._dist_to_food:
			dist_map = self._dist_to_food
		elif self._dist_to_goal:
			dist_map = self._dist_to_goal
		if not dist_map:
			return None
		return dist_map.get(pos)

	def _obs(self) -> Tuple[int, ...]:
		ax, ay = self.agent
		fx, fy = self.food
		gx, gy = self.goal
		return (int(self.level_id), ax, ay, fx, fy, gx, gy, self.energy)

	def step(self, action: Action) -> StepResult:
		self._capture_pre_step_snapshot()
		prev_dist = self._distance_to_target(self.agent)
		self.steps += 1

		ax, ay = self.agent

		if action == 0:      # UP
			ay -= 1
		elif action == 1:    # RIGHT
			ax += 1
		elif action == 2:    # DOWN
			ay += 1
		elif action == 3:    # LEFT
			ax -= 1
		else:
			raise ValueError(f"Unknown action: {action}")

		ax = max(0, min(self.width - 1, ax))
		ay = max(0, min(self.height - 1, ay))
		new_pos = (ax, ay)
		bumped = False
		if new_pos in self.walls:
			bumped = True
			new_pos = self.agent

		self.agent = new_pos

		self.energy -= self.energy_step_cost

		reward = float(self.step_penalty)
		new_dist = self._distance_to_target(self.agent)
		if prev_dist is not None and new_dist is not None and prev_dist != new_dist:
			reward += (prev_dist - new_dist) * self.shaping_strength
		done = False
		info: Dict = {"steps": self.steps, "level_id": self.level_id}

		if bumped:
			reward += self.wall_penalty
			info["bumped_wall"] = True

		if self.agent in self.hazards:
			reward = self.trap_penalty
			done = True
			info["terminal"] = "trap"

		if not done and self.agent == self.goal and (not self.food_enabled or self.food_collected):
			reward = self.goal_reward
			done = True
			info["terminal"] = "goal"

		if not done and self.food_enabled and not self.food_collected and self.agent == self.food:
			reward += self.food_reward
			self.energy += self.energy_food_gain
			if self.energy_max is not None:
				self.energy = min(self.energy, self.energy_max)
			self.food_collected = True
			self.food = (-1, -1)
			self._dist_to_food = None
			info["got_food"] = True

		if not done and self.energy <= 0:
			reward = self.starve_penalty
			done = True
			info["terminal"] = "starved"

		return StepResult(obs=self._obs(), reward=float(reward), done=done, info=info)

	def render(self) -> str:
		grid = [["." for _ in range(self.width)] for __ in range(self.height)]

		for (wx, wy) in self.walls:
			grid[wy][wx] = "#"
		for (hx, hy) in self.hazards:
			grid[hy][hx] = "T"
		if (not self.food_enabled) or self.food_collected:
			gx, gy = self.goal
			grid[gy][gx] = "G"
		fx, fy = self.food
		if fx >= 0 and fy >= 0:
			grid[fy][fx] = "F"
		ax, ay = self.agent
		grid[ay][ax] = "A"

		lines = ["".join(row) for row in grid]
		return "\n".join(lines)
