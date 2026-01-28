from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Tuple, List

Action = int  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

@dataclass
class StepResult:
	obs: Tuple[int, int, int, int, int]  # (ax, ay, fx, fy, energy)
	reward: float
	done: bool
	info: Dict

class GridSurvivalEnv:
	"""
	Small grid survival environment (kid-friendly).

	Observation (state):
	  (agent_x, agent_y, food_x, food_y, energy)

	Actions:
	  0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

	Rewards (simple on purpose):
	  +10.0  when you reach food (and gain energy)
	  -10.0  if you hit a hazard (episode ends)
	  -0.01  per step (tiny "don't wander forever" cost)
	  -1.0   if you run out of energy (episode ends)

	Episode ends:
	  - hazard
	  - energy <= 0
	"""

	def __init__(
		self,
		width: int = 10,
		height: int = 10,
		n_hazards: int = 8,
		energy_start: int = 25,
		energy_food_gain: int = 18,
		energy_step_cost: int = 1,
		seed: int | None = None,
	) -> None:
		self.width = width
		self.height = height
		self.n_hazards = n_hazards
		self.energy_start = energy_start
		self.energy_food_gain = energy_food_gain
		self.energy_step_cost = energy_step_cost

		self._rng = random.Random(seed)

		self.agent = (0, 0)
		self.food = (0, 0)
		self.hazards: List[Tuple[int, int]] = []
		self.energy = self.energy_start
		self.steps = 0

	def seed(self, seed: int) -> None:
		self._rng.seed(seed)

	def reset(self) -> Tuple[int, int, int, int, int]:
		self.steps = 0
		self.energy = self.energy_start

		self.hazards = []
		occupied = set()

		# place hazards
		while len(self.hazards) < self.n_hazards:
			x = self._rng.randrange(self.width)
			y = self._rng.randrange(self.height)
			if (x, y) in occupied:
				continue
			self.hazards.append((x, y))
			occupied.add((x, y))

		# place agent not on hazard
		while True:
			ax = self._rng.randrange(self.width)
			ay = self._rng.randrange(self.height)
			if (ax, ay) not in occupied:
				self.agent = (ax, ay)
				occupied.add((ax, ay))
				break

		# place food not on hazard or agent
		self.food = self._spawn_food(occupied)

		return self._obs()

	def _spawn_food(self, occupied: set) -> Tuple[int, int]:
		while True:
			fx = self._rng.randrange(self.width)
			fy = self._rng.randrange(self.height)
			if (fx, fy) not in occupied:
				return (fx, fy)

	def _obs(self) -> Tuple[int, int, int, int, int]:
		ax, ay = self.agent
		fx, fy = self.food
		return (ax, ay, fx, fy, self.energy)

	def step(self, action: Action) -> StepResult:
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

		# keep inside the grid (bump into walls = stay put)
		ax = max(0, min(self.width - 1, ax))
		ay = max(0, min(self.height - 1, ay))

		self.agent = (ax, ay)

		# energy cost per step
		self.energy -= self.energy_step_cost

		reward = -0.01  # tiny living cost
		done = False
		info: Dict = {"steps": self.steps}

		# hazard?
		if self.agent in self.hazards:
			reward = -10.0
			done = True
			info["terminal"] = "hazard"

		# food?
		if not done and self.agent == self.food:
			reward = 10.0
			self.energy += self.energy_food_gain
			# respawn food (cannot spawn on hazards or agent)
			occupied = set(self.hazards)
			occupied.add(self.agent)
			self.food = self._spawn_food(occupied)
			info["got_food"] = True

		# starvation?
		if not done and self.energy <= 0:
			reward = -1.0
			done = True
			info["terminal"] = "starved"

		return StepResult(obs=self._obs(), reward=reward, done=done, info=info)

	def render(self) -> str:
		grid = [["." for _ in range(self.width)] for __ in range(self.height)]

		for (hx, hy) in self.hazards:
			grid[hy][hx] = "X"
		fx, fy = self.food
		grid[fy][fx] = "F"
		ax, ay = self.agent
		grid[ay][ax] = "A"

		lines = ["".join(row) for row in grid]
		return "\n".join(lines)
