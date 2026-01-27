# Copilot Instructions for mini-rl-survival

## Project Overview
- **mini-rl-survival** is a minimal, kid-friendly tabular Reinforcement Learning project. The agent learns to survive and collect food in a 10x10 grid world using Q-learning (no deep learning, no big libraries).
- The environment, agent, and training loop are each in their own file for clarity and hackability.

## Key Components
- **env.py**: Defines `GridSurvivalEnv`, a grid world with food, hazards, and energy mechanics. State is `(agent_x, agent_y, food_x, food_y, energy)`. Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT. Rewards: +10 (food), -10 (hazard), -0.01 (step), -1 (energy out).
- **qlearn.py**: Implements `QLearningAgent` (tabular Q-learning, epsilon-greedy) and `QLearningConfig` for hyperparameters.
- **train.py**: Main training loop. Handles training, periodic evaluation, and ASCII rendering. Key parameters (episodes, learning rate, etc.) are at the top for easy tweaking.

## Developer Workflows
- **Run training:**
  ```sh
  python train.py
  ```
  This prints episode stats and periodically runs a greedy evaluation with ASCII grid rendering.
- **Tweak learning:**
  - Edit `EPISODES`, `ALPHA`, `GAMMA`, `EPS_START`, `EPS_END`, `EPS_DECAY_STEPS` in `train.py`.
  - Change grid size, energy, hazards, or food reward in `env.py`.
- **No external dependencies** by default. Add to `requirements.txt` if needed.

## Project Conventions & Patterns
- **No deep RL libraries**: All logic is implemented from scratch for transparency.
- **Tabular Q-learning**: Q-table is a Python dict mapping state tuples to action-value lists.
- **Epsilon-greedy**: Exploration decays linearly over steps.
- **Episode/step loop**: Training loop is explicit and easy to follow.
- **Evaluation**: Greedy policy is evaluated every `EVAL_EVERY` episodes with ASCII rendering.
- **Modifiability**: All key parameters are exposed at the top of `train.py` or as class arguments.

## Examples
- To increase learning, set `EPISODES = 2000` in `train.py`.
- To make the world harder, increase `n_hazards` in `GridSurvivalEnv`.
- To change exploration, adjust `EPS_START`/`EPS_END`/`EPS_DECAY_STEPS`.

## Integration Points
- No external APIs or frameworks. All code is self-contained.
- Add dependencies to `requirements.txt` if you extend the project.

## References
- See `README.md` for a user-friendly summary and tweak suggestions.
- Key files: `env.py`, `qlearn.py`, `train.py`.

---
For AI agents: Prioritize clarity, hackability, and minimalism. Keep new code simple and easy to follow. If adding features, match the explicit, transparent style of the existing codebase.
