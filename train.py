from __future__ import annotations

import time
from env import GridSurvivalEnv
from qlearn import QLearningAgent, QLearningConfig

# ----- Training settings (edit these!) -----
EPISODES = 800
MAX_STEPS_PER_EP = 400

ALPHA = 0.25
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 30_000

EVAL_EVERY = 100
EVAL_STEPS = 120
SLEEP_BETWEEN_RENDER = 0.03  # slow down evaluation rendering

# ------------------------------------------


def evaluate(env: GridSurvivalEnv, agent: QLearningAgent, steps: int = 120) -> None:
    obs = env.reset()
    total = 0.0
    for t in range(steps):
        a = agent.act(obs, greedy=True)
        res = env.step(a)
        total += res.reward
        obs = res.obs
        print(env.render())
        print(f"t={t:03d} energy={obs[4]:02d} total_reward={total:.2f}")
        print("-" * env.width)
        time.sleep(SLEEP_BETWEEN_RENDER)
        if res.done:
            print("DONE:", res.info.get("terminal", ""))
            break


def main() -> None:
    env = GridSurvivalEnv(
        width=10,
        height=10,
        n_hazards=10,
        energy_start=25,
        energy_food_gain=18,
        energy_step_cost=1,
        seed=0,
    )

    cfg = QLearningConfig(
        alpha=ALPHA,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay_steps=EPS_DECAY_STEPS,
    )
    agent = QLearningAgent(n_actions=4, cfg=cfg, seed=0)

    best_len = 0

    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        foods = 0

        for _ in range(MAX_STEPS_PER_EP):
            a = agent.act(obs, greedy=False)
            res = env.step(a)
            agent.learn(obs, a, res.reward, res.obs, res.done)

            total_reward += res.reward
            steps += 1
            if res.info.get("got_food"):
                foods += 1

            obs = res.obs
            if res.done:
                break

        if steps > best_len:
            best_len = steps

        if ep % 20 == 0:
            print(
                f"ep={ep:04d} steps={steps:03d} foods={foods:02d} "
                f"reward={total_reward:7.2f} eps={agent.epsilon():.3f} best_steps={best_len}"
            )

        if ep % EVAL_EVERY == 0:
            print("\n=== EVAL (greedy policy) ===")
            evaluate(env, agent, steps=EVAL_STEPS)
            print("=== back to training ===\n")


if __name__ == "__main__":
    main()
