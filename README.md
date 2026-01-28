# Mini RL Survival

A minimal, hackable tabular reinforcement learning project. The agent learns to navigate grid mazes with walls, a key, and a door using Q-learning. Everything is implemented from scratch for transparency and education.

## Highlights
- 50 curated preset mazes (plus random mode) with key and door gating.
- Door unlocks only after the agent collects the key.
- Difficulty modes (easy/medium/hard) adjust key and door spacing.
- Interactive Pygame viewer with overlays, telemetry, and run history.
- In-app training with curriculum and per-map success stats.
- Guided "New Q-table" wizard for beginners.
- Batch evaluate all maps and export per-map stats (JSON/CSV).
- Optional 3D state-space visualization (PCA) for analysis.
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
- Cycle menu backgrounds (Settings -> View)

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

### Batch Eval All Maps (CLI)
```sh
python -m scripts.train_agent --load data/qtable.pkl --eval-all-maps --eval-all-episodes 20 --eval-all-out data/map_stats.json
```

## In-app Training
The Training screen lets you:
- Start/pause training
- Enable curriculum mode and adjust its settings
- View average reward/steps/foods and worst maps
- Create a fresh Q-table with a guided wizard
- Save/load Q-tables
- Delete training artifacts (Q-tables, run stats, snapshots)
- Export per-map stats and run batch "Test all maps"

## Guided New Q-table (Wizard)
Training -> "New Q-table (guided)" walks you through environment and training
settings, then asks for a filename before creating the new table.
This avoids overwriting existing models and helps new users learn the options.

## 3D State Space (Optional)
From Training you can open the 3D view, or run the script directly:
```sh
python -m scripts.plot_state_space_3d --load data/qtable.pkl --mode state
```
You can filter by level, color by value or action, and limit points for speed.


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
- `map_stats.json` (from "Test all maps")
- `qtable_visualize.pkl` (auto-saved for 3D view)

You can delete these from the Training tab to reset to a clean state.

## Project Structure
- `core/` environment and Q-learning implementation
- `viewers/` Pygame UI, scenes, overlays, and widgets
- `scripts/` CLI tools (train, heatmap, utilities)
- `scripts/plot_state_space_3d.py` 3D state-space visualization
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
- If the door never appears: it unlocks only after the key is collected.

## Third-party assets
- Kenney Game Icons (CC0) in `assets/kenney_game_icons`
- Press Start 2P font (SIL OFL 1.1) in `assets/press-start-2p`
- Maze Curriculum Dataset (MIT) via `selimaktas/maze-curriculum-dataset`
