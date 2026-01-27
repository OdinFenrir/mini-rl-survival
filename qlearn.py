from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Tuple

State = Tuple[int, int, int, int, int]  # (ax, ay, fx, fy, energy)
Action = int  # 0..3


@dataclass
class QLearningConfig:
    alpha: float = 0.2     # learning rate
    gamma: float = 0.97    # discount factor
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000  # larger = slower decay


class QLearningAgent:
    """
    Classic Q-learning (tabular).

    Q[s][a] updated by:
      Q(s,a) <- Q(s,a) + alpha * (r + gamma*max_a' Q(s',a') - Q(s,a))
    """

    def __init__(self, n_actions: int, cfg: QLearningConfig, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.cfg = cfg
        self._rng = random.Random(seed)
        self.Q: Dict[State, list[float]] = {}
        self.total_steps = 0

    def _ensure(self, s: State) -> None:
        if s not in self.Q:
            self.Q[s] = [0.0 for _ in range(self.n_actions)]

    def epsilon(self) -> float:
        # linear decay to keep it very simple
        t = min(self.total_steps, self.cfg.eps_decay_steps)
        frac = t / self.cfg.eps_decay_steps
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, s: State, greedy: bool = False) -> Action:
        self._ensure(s)
        if greedy:
            return int(max(range(self.n_actions), key=lambda a: self.Q[s][a]))

        eps = self.epsilon()
        if self._rng.random() < eps:
            return self._rng.randrange(self.n_actions)
        return int(max(range(self.n_actions), key=lambda a: self.Q[s][a]))

    def learn(self, s: State, a: Action, r: float, sp: State, done: bool) -> None:
        self._ensure(s)
        self._ensure(sp)

        q_sa = self.Q[s][a]
        target = r
        if not done:
            target = r + self.cfg.gamma * max(self.Q[sp])

        self.Q[s][a] = q_sa + self.cfg.alpha * (target - q_sa)
        self.total_steps += 1
