from __future__ import annotations

import json
import os
from typing import Any, Dict

from core.env import GridSurvivalEnv
from core.qlearning import QLearningAgent

SAVE_VERSION = 1


def save_qtable(agent: QLearningAgent, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    agent.save(path)


def load_qtable(path: str, seed: int | None = None) -> QLearningAgent:
    return QLearningAgent.load(path, seed=seed)


def save_env_snapshot(env: GridSurvivalEnv, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload: Dict[str, Any] = {
        'version': SAVE_VERSION,
        'width': env.width,
        'height': env.height,
        'n_hazards': env.n_hazards,
        'energy_start': env.energy_start,
        'energy_food_gain': env.energy_food_gain,
        'energy_step_cost': env.energy_step_cost,
        'agent': list(env.agent),
        'food': list(env.food),
        'hazards': [list(h) for h in env.hazards],
        'energy': env.energy,
        'steps': env.steps,
    }
    rng = getattr(env, '_rng', None)
    if rng is not None and hasattr(rng, 'getstate'):
        payload['rng_state'] = rng.getstate()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)


def load_env_snapshot(path: str) -> GridSurvivalEnv:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if int(payload.get('version', 0)) != SAVE_VERSION:
        raise ValueError(f"Unsupported snapshot version: {payload.get('version')}")
    env = GridSurvivalEnv(
        width=int(payload['width']),
        height=int(payload['height']),
        n_hazards=int(payload['n_hazards']),
        energy_start=int(payload['energy_start']),
        energy_food_gain=int(payload['energy_food_gain']),
        energy_step_cost=int(payload['energy_step_cost']),
        seed=None,
    )
    env.agent = (int(payload['agent'][0]), int(payload['agent'][1]))
    env.food = (int(payload['food'][0]), int(payload['food'][1]))
    env.hazards = [(int(x), int(y)) for x, y in payload['hazards']]
    env.energy = int(payload['energy'])
    env.steps = int(payload['steps'])
    rng = getattr(env, '_rng', None)
    if rng is not None and 'rng_state' in payload and hasattr(rng, 'setstate'):
        rng.setstate(payload['rng_state'])
    return env
