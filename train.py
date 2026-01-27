from __future__ import annotations

import argparse
import statistics
import time
from typing import Tuple

from env import GridSurvivalEnv
from qlearn import QLearningAgent, QLearningConfig
import viz


def run_episode(env: GridSurvivalEnv, agent: QLearningAgent, max_steps: int, train: bool) -> Tuple[int, int, float, str]:
    """Returns (steps_survived, foods_eaten, total_reward, terminal_reason)."""
    obs = env.reset()
    total_reward = 0.0
    foods = 0
    terminal = ""

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
            break

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
        print(f"t={{t:03d}} energy={{obs[4]:02d}} foods={{foods:02d}} total_reward={{total:.2f}}")
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

    # env knobs
    ap.add_argument("--w", type=int, default=10, help="grid width")
    ap.add_argument("--h", type=int, default=10, help="grid height")
    ap.add_argument("--hazards", type=int, default=10, help="number of hazards")
    ap.add_argument("--energy-start", type=int, default=25, help="starting energy")
    ap.add_argument("--energy-food", type=int, default=18, help="energy gained on food")
    ap.add_argument("--energy-step", type=int, default=1, help="energy lost per step")

    # q-learning knobs
    ap.add_argument("--alpha", type=float, default=0.25, help="learning rate")
    ap.add_argument("--gamma", type=float, default=0.98, help="discount")
    ap.add_argument("--eps-start", type=float, default=1.0, help="starting epsilon")
    ap.add_argument("--eps-end", type=float, default=0.05, help="final epsilon")
    ap.add_argument("--eps-decay", type=int, default=30_000, help="epsilon decay steps")

    args = ap.parse_args()

    env = GridSurvivalEnv(
        width=args.w,
        height=args.h,
        n_hazards=args.hazards,
        energy_start=args.energy_start,
        energy_food_gain=args.energy_food,
        energy_step_cost=args.energy_step,
        seed=args.seed,
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
        # if you changed knobs, keep the new cfg (makes tinkering easier)
        agent.cfg = cfg
    else:
        agent = QLearningAgent(n_actions=4, cfg=cfg, seed=args.seed)

    if args.play:
        print(f"Loaded Q-table size: {{len(agent.Q)}} states")
        play(env, agent, max_steps=args.max_steps, sleep_s=args.sleep)
        return

    # ---- TRAIN ----
    recent_steps = []
    recent_foods = []
    recent_rewards = []

    for ep in range(1, args.episodes + 1):
        steps, foods, total, term = run_episode(env, agent, max_steps=args.max_steps, train=True)

        recent_steps.append(steps)
        recent_foods.append(foods)
        recent_rewards.append(total)

        if len(recent_steps) > 100:
            recent_steps.pop(0)
            recent_foods.pop(0)
            recent_rewards.pop(0)

        if ep % 25 == 0:
            print(
                f"ep={{ep:05d}} steps={{steps:03d}} foods={{foods:02d}} reward={{total:8.2f}} term={{term:7s}} "
                f"eps={{agent.epsilon():.3f}} Q_states={{len(agent.Q)}} "
                f"last100(avg_steps={{statistics.mean(recent_steps):.1f}}, avg_foods={{statistics.mean(recent_foods):.2f}})"
            )

        if args.eval_every and ep % args.eval_every == 0:
            print("  " + eval_stats(env, agent, episodes=args.eval_episodes, max_steps=args.max_steps))

        # save a checkpoint sometimes so you don't lose progress
        if ep % 500 == 0:
            agent.save(args.save)

    agent.save(args.save)
    print(f"\nSaved model: {{args.save}} (Q states={{len(agent.Q)}})")


if __name__ == "__main__":
    main()