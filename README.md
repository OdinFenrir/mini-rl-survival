# Mini RL Survival

A minimal, hackable tabular Reinforcement Learning project. The agent learns to survive and collect food in a grid world using Q-learning. All logic is implemented from scratch for transparency and educational value.

## Project Structure

- `core/`
  - `env.py` — Grid world environment (food, hazards, energy)
  - `qlearning.py` — Tabular Q-learning agent and config
  - `viz.py` — Minimal ASCII renderer for terminal output
- `scripts/`
  - `train_agent.py` — Main training loop and evaluation (all parameters via argparse)
  - `gen_qtable.py` — Utility to generate a small random Q-table for testing/demo
  - `qtable_heatmap.py` — Export Q-table heatmaps as images
- `viewers/`
  - `pygame_viewer.py` — Interactive Pygame viewer for visualizing agent behavior and Q-tables
- `data/` (empty by default; for generated artifacts like Q-tables or heatmaps)

## Quick Start

1. **Train the agent:**
   ```sh
   python scripts/train_agent.py --episodes 2000 --eval-every 200 --eval-episodes 50 --save data/qtable.pkl
   ```
2. **Watch the learned agent play:**
   ```sh
   python scripts/train_agent.py --play --load data/qtable.pkl --max-steps 200 --sleep 0.05
   ```
3. **Visualize Q-table heatmap:**
   ```sh
   python scripts/qtable_heatmap.py --qtable data/qtable.pkl --out data/q_heatmap.png
   ```
4. **Interactive viewer:**
   ```sh
   python viewers/pygame_viewer.py --load data/qtable.pkl
   ```

## Customization & Experiments
- All environment and learning parameters are argparse flags in `train_agent.py` (see `--help`).
- Example: `--hazards 14 --energy-step 2` makes survival harder.
- Example: `--eps-decay 80000` slows exploration decay.

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies (`pygame`, `numpy`, `imageio`, `matplotlib` for full features)

## Notes
- No deep RL libraries; all logic is explicit and minimal.
- Q-table is a Python dict mapping state tuples to action-value lists.
- All generated data (Q-tables, heatmaps) should be stored in `data/`.

---
For more details, see the code and comments in each file. Contributions and experiments are welcome!
