# Mini RL Survival

A minimal, hackable tabular reinforcement learning project. The agent learns to navigate grid mazes with walls, food, and a goal using Q-learning. Everything is implemented from scratch for transparency and education.

## Highlights
- 50 curated preset mazes (plus random mode) with food and goal gating.
- Goal tile unlocks only after the agent eats the fruit.
- Interactive Pygame viewer with overlays, telemetry, and run history.
- In-app training screen with curriculum mode and per-map success stats.
- CLI training with curriculum flags for large-scale runs.
- Save/load Q-tables and environment snapshots.

## Requirements
- Python 3.10+
- Install deps:
  ```sh
  pip install -r requirements.txt
  ```

## Quick Start (Viewer)
```sh
python -m viewers
```
From the menu you can:
- Start or continue a simulation
- Train in the Training tab
- Load or save Q-tables
- Change settings and visuals

## Training (CLI)
Train a Q-table and save it:
```sh
python -m scripts.train_agent --episodes 2000 --eval-every 200 --eval-episodes 50 --save data/qtable.pkl
```
Watch the learned policy:
```sh
python -m scripts.train_agent --play --load data/qtable.pkl --max-steps 200 --sleep 0.05
```

### Curriculum Training (all 50 maps)
```sh
python -m scripts.train_agent \
  --episodes 20000 \
  --eval-every 500 \
  --eval-episodes 50 \
  --level-mode preset \
  --curriculum \
  --curriculum-start 5 \
  --curriculum-step 5 \
  --curriculum-window 50 \
  --curriculum-threshold 0.8 \
  --energy-start 60 \
  --energy-max 80 \
  --energy-food 30 \
  --save data/qtable.pkl
```

## In-app Training
The Training screen lets you:
- Start/pause training
- Enable curriculum mode and adjust its settings
- View average reward/steps/foods and worst maps
- Save/load Q-tables
- Delete training artifacts (Q-tables, run stats, snapshots)

## Key Controls (Simulation)
- Space: Pause/Resume
- . (period): Step once
- R: Reset episode
- M: Toggle policy mode
- H: Toggle heatmap
- P: Toggle policy arrows
- Q: Toggle Q-value hover panel
- D: Toggle debug overlay
- Ctrl+S / Ctrl+L: Save / Load Q-table
- Ctrl+O / Ctrl+I: Save / Load env snapshot
- Ctrl+E: Export screenshot
- Ctrl+X: Export run stats
- Ctrl+T: Telemetry overlay
- Ctrl+K: Run history overlay
- ?: Help overlay
- Esc: Back

## Data Files
Generated artifacts live in `data/`:
- `*.pkl` Q-tables
- `env_snapshot.json`
- `run_history.jsonl`
- `run_stats.json` + `run_stats.csv`

You can delete these from the Training tab to reset to a clean state.

## Project Structure
- `core/` environment and Q-learning implementation
- `viewers/` Pygame UI, scenes, overlays, and widgets
- `scripts/` CLI tools (train, heatmap, utilities)
- `assets/` fonts, icons, and maze packs
- `data/` generated artifacts (empty by default)

## Maze Pack
`assets/levels/maze_pack.json` was generated from the Maze Curriculum Dataset.
To regenerate:
```sh
python scripts/fetch_maze_pack.py
```

## Troubleshooting
- If you see `ModuleNotFoundError`, use module mode:
  - `python -m scripts.train_agent`
  - `python -m viewers`
- If the goal never appears: the goal unlocks only after food is collected.

## Third-party assets
- Kenney 1-Bit Pack (CC0) in `assets/1bitpack_kenney_1.1`
- Kenney Game Icons (CC0) in `assets/kenney_game_icons`
- Food Pixel Art (CC0) in `assets/food_pixel_art`
- Press Start 2P font (SIL OFL 1.1) in `assets/press-start-2p`
- Maze Curriculum Dataset (MIT) via `selimaktas/maze-curriculum-dataset`
