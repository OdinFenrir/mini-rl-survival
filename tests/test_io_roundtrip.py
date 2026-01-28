import os
import tempfile

from core.env import GridSurvivalEnv
from core.qlearning import QLearningAgent, QLearningConfig
from viewers.io.save_load import load_env_snapshot, load_qtable, save_env_snapshot, save_qtable


def test_qtable_roundtrip():
    cfg = QLearningConfig()
    agent = QLearningAgent(n_actions=4, cfg=cfg, seed=0)
    agent.Q[(0, 0, 1, 1, 10)] = [1.0, 2.0, 3.0, 4.0]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'q.pkl')
        save_qtable(agent, path)
        a2 = load_qtable(path, seed=0)
        assert a2.Q[(0, 0, 1, 1, 10)] == [1.0, 2.0, 3.0, 4.0]


def test_env_snapshot_roundtrip():
    env = GridSurvivalEnv(width=8, height=7, n_hazards=5, seed=123)
    env.reset()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'env.json')
        save_env_snapshot(env, path)
        e2 = load_env_snapshot(path)
        assert (e2.width, e2.height, e2.n_hazards) == (env.width, env.height, env.n_hazards)
        assert e2.agent == env.agent
        assert e2.food == env.food
        assert e2.hazards == env.hazards
        assert e2.energy == env.energy
