import os
import tempfile

from core.env import GridSurvivalEnv
from viewers.io.save_load import load_env_snapshot, save_env_snapshot


def test_snapshot_determinism_next_step():
    env = GridSurvivalEnv(width=10, height=10, n_hazards=8, seed=999)
    env.reset()
    res1 = env.step(1)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'snap.json')
        save_env_snapshot(env, p)
        e2 = load_env_snapshot(p)
        res2 = e2.step(1)
        assert res1.obs == res2.obs
        assert res1.reward == res2.reward
        assert res1.done == res2.done
        assert res1.info.get('terminal', '') == res2.info.get('terminal', '')
