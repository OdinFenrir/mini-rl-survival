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
        'n_traps': getattr(env, "n_traps", env.n_hazards),
        'n_walls': getattr(env, "n_walls", 0),
        'energy_start': env.energy_start,
        'energy_food_gain': env.energy_food_gain,
        'energy_step_cost': env.energy_step_cost,
        'energy_max': int(getattr(env, "energy_max", 0) or 0),
        'level_mode': getattr(env, "level_mode", "preset"),
        'level_index': getattr(env, "level_index", 0),
        'level_cycle': getattr(env, "level_cycle", False),
        'food_enabled': getattr(env, "food_enabled", True),
        'agent': list(env.agent),
        'food': list(env.food),
        'goal': list(getattr(env, "goal", (0, 0))),
        'hazards': [list(h) for h in env.hazards],
        'walls': [list(w) for w in getattr(env, "walls", [])],
        'food_collected': bool(getattr(env, "food_collected", False)),
        'energy': env.energy,
        'steps': env.steps,
        'level_id': getattr(env, "level_id", 0),
        'episode': getattr(env, "episode", 0),
	}
	rng = getattr(env, '_rng', None)
	if rng is not None and hasattr(rng, 'getstate'):
		payload['rng_state'] = rng.getstate()
	prev_snapshot = getattr(env, '_last_step_snapshot', None)
	if prev_snapshot is not None:
		payload['prev_state'] = _snapshot_to_payload(prev_snapshot)
		payload['state_kind'] = 'pre_step'
		env._last_step_snapshot = None
	else:
		payload['state_kind'] = 'current'
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(payload, f)


def _tupleize_state(obj: Any) -> Any:
	if isinstance(obj, list):
		return tuple(_tupleize_state(item) for item in obj)
	return obj


def _snapshot_to_payload(snapshot: Dict[str, Any]) -> Dict[str, Any]:
	return {
		'agent': [int(snapshot['agent'][0]), int(snapshot['agent'][1])],
		'food': [int(snapshot['food'][0]), int(snapshot['food'][1])],
		'goal': [int(snapshot.get('goal', (0, 0))[0]), int(snapshot.get('goal', (0, 0))[1])],
		'hazards': [[int(x), int(y)] for x, y in snapshot['hazards']],
		'walls': [[int(x), int(y)] for x, y in snapshot.get('walls', [])],
		'food_collected': bool(snapshot.get('food_collected', False)),
		'energy': int(snapshot['energy']),
		'steps': int(snapshot['steps']),
		'level_id': int(snapshot.get('level_id', 0)),
		'episode': int(snapshot.get('episode', 0)),
		'rng_state': snapshot['rng_state'],
	}


def _restore_state(env: GridSurvivalEnv, state_payload: Dict[str, Any]) -> None:
	env.agent = (int(state_payload['agent'][0]), int(state_payload['agent'][1]))
	env.food = (int(state_payload['food'][0]), int(state_payload['food'][1]))
	env.goal = (int(state_payload.get('goal', (0, 0))[0]), int(state_payload.get('goal', (0, 0))[1]))
	env.hazards = [(int(x), int(y)) for x, y in state_payload['hazards']]
	env.walls = [(int(x), int(y)) for x, y in state_payload.get('walls', [])]
	env.n_hazards = len(env.hazards)
	if hasattr(env, "n_traps"):
		env.n_traps = len(env.hazards)
	env.food_collected = bool(state_payload.get('food_collected', False))
	env.energy = int(state_payload['energy'])
	env.steps = int(state_payload['steps'])
	env.level_id = int(state_payload.get('level_id', 0))
	env.episode = int(state_payload.get('episode', 0))
	rng = getattr(env, '_rng', None)
	if rng is not None and 'rng_state' in state_payload and hasattr(rng, 'setstate'):
		rng_state = _tupleize_state(state_payload['rng_state'])
		rng.setstate(rng_state)


def load_env_snapshot(path: str) -> GridSurvivalEnv:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if int(payload.get('version', 0)) != SAVE_VERSION:
        raise ValueError(f"Unsupported snapshot version: {payload.get('version')}")
    env = GridSurvivalEnv(
        width=int(payload['width']),
        height=int(payload['height']),
        n_hazards=int(payload.get('n_hazards', payload.get('n_traps', 0))),
        energy_start=int(payload['energy_start']),
        energy_food_gain=int(payload['energy_food_gain']),
        energy_step_cost=int(payload['energy_step_cost']),
        energy_max=int(payload.get('energy_max', 0)),
        seed=None,
        level_mode=str(payload.get('level_mode', 'preset')),
        level_index=int(payload.get('level_index', 0)),
        level_cycle=bool(payload.get('level_cycle', False)),
        n_walls=int(payload.get('n_walls', 0)),
        n_traps=int(payload.get('n_traps', payload.get('n_hazards', 0))),
        food_enabled=bool(payload.get('food_enabled', True)),
    )
    state_payload = payload
    if payload.get('state_kind') == 'pre_step' and 'prev_state' in payload:
        state_payload = payload['prev_state']
    _restore_state(env, state_payload)
    setattr(env, '_last_step_snapshot', None)
    return env
