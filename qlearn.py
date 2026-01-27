from __future__ import annotations

from dataclasses import dataclass
import pickle
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
    Classic tabular Q-learning.

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
        # linear decay to keep it simple
        t = min(self.total_steps, self.cfg.eps_decay_steps)
        frac = t / self.cfg.eps_decay_steps if self.cfg.eps_decay_steps > 0 else 1.0
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
        target = r if done else (r + self.cfg.gamma * max(self.Q[sp]))

        self.Q[s][a] = q_sa + self.cfg.alpha * (target - q_sa)
        self.total_steps += 1

    def save(self, path: str) -> None:
        payload = {
            "n_actions": self.n_actions,
            "cfg": self.cfg,
            "total_steps": self.total_steps,
            "Q": self.Q,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str, seed: int | None = None) -> "QLearningAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = QLearningAgent(n_actions=payload["n_actions"], cfg=payload["cfg"], seed=seed)
        agent.total_steps = payload.get("total_steps", 0)
        agent.Q = payload.get("Q", {})
        return agent
