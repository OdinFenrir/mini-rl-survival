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

---

## Project Implementation Roadmap & Checklist

### 0. Foundation: Architecture & Structure
- [ ] Refactor `viewers/pygame_viewer.py` to be entry point only
- [ ] Create `viewers/app.py` for main loop, scene switching, global state
- [ ] Create `viewers/scenes/`:
  - [ ] `menu.py` (MainMenuScene)
  - [ ] `sim.py` (SimulationScene)
  - [ ] `settings.py` (SettingsScene)
  - [ ] `help.py` (HelpScene)
- [ ] Create `viewers/ui/`:
  - [ ] `widgets.py` (Button, Toggle, Slider, Dropdown, Text, Panel)
  - [ ] `layout.py` (anchors, responsive layout helpers)
  - [ ] `theme.py` (colors, fonts, scaling, colorblind palettes)
- [ ] Create `viewers/io/`:
  - [ ] `save_load.py` (qtable/env/state, error-safe)
  - [ ] `export.py` (screenshots, csv/json exports)
- [ ] Create `viewers/overlays/`:
  - [ ] `stats.py`
  - [ ] `help_overlay.py`
  - [ ] `debug.py`
  - [ ] `policy.py` (Q arrows)
  - [ ] `qvalues.py` (hover panel)
- [ ] Ensure all input routes through a single place
- [ ] Ensure new screens/overlays can be added without touching the core loop

### 1. Menu System & Navigation
- [ ] Implement `Scene` interface: `handle_event()`, `update(dt)`, `render(surface)`
- [ ] Implement `FocusManager` for widgets (tab order)
- [ ] Persist settings in an `AppConfig` object
- [ ] Main Menu: Start Simulation, Settings, Help/About, Exit
- [ ] Settings Menu (in-app): Env, Agent, View, Buttons
- [ ] Keyboard navigation for all menus
- [ ] Visible focus highlight
- [ ] “Press ? for help” hint in every screen

### 2. Simulation Controls & Overlays
- [ ] Move current simulation logic into `SimulationScene`
- [ ] Add controls: Pause/resume, Step once, Reset, etc.
- [ ] Overlays: Stats, Help, Debug, Policy, Q-value hover
- [ ] Implement `RenderContext`
- [ ] Add overlay manager

### 3. Save/Load/Export Functionality
- [ ] Q-table save/load (menu + hotkey)
- [ ] Environment snapshot save/load
- [ ] Export: Screenshot, Stats, Optional: heatmap/policy images
- [ ] File picker
- [ ] Feedback banner for save/load/export
- [ ] Graceful error handling
- [ ] Implement `ToastManager`
- [ ] Implement all save/load/export helpers

### 4. Accessibility & Visual Quality
- [ ] Colorblind palettes
- [ ] Font scaling
- [ ] High-contrast mode
- [ ] “Reduced motion” toggle
- [ ] Responsive layout
- [ ] Consistent spacing, iconography
- [ ] Clear focus rings
- [ ] Heatmap opacity control
- [ ] Centralize theme
- [ ] UI scale multiplier

### 5. Advanced Features & Final Polish
- [ ] Advanced settings
- [ ] Global exception hook
- [ ] Safe defaults if config missing
- [ ] Scene transitions
- [ ] Consistent prompts
- [ ] Tooltips on widgets

### 6. Testing & Documentation
- [ ] Headless smoke test
- [ ] IO tests
- [ ] Determinism test
- [ ] README: Controls, Menus, Save/Load, Troubleshooting
- [ ] In-app Help/About screen
- [ ] “Troubleshooting” section
- [ ] Ensure a new machine can run viewer with only README
- [ ] Tests cover file formats, no manual clicking required

### 7. Quality-of-life & Professional Polish
#### A. Quality-of-life
- [ ] Config persistence: save/load `AppConfig` to `data/viewer_config.json` (or `.ini`)
- [ ] Centralized hotkey map + command palette (Ctrl+K)
- [ ] Deterministic replay hooks: record `(seed, env_state, actions)` for exact episode replay
#### B. Telemetry & Episode Tracking
- [ ] `EpisodeStats` / `RunStats` dataclass for per-episode tracking
- [ ] Ring buffer history overlay (sparkline for last N episodes)
- [ ] Event logging: optional JSONL log of sessions (start/stop, settings, outcomes)
#### C. Performance / Correctness
- [ ] Separate `render_fps` from `sim_steps_per_frame`
- [ ] Cached policy/heatmap overlays (recompute only when needed)
- [ ] State-key sanity tools: dev overlay showing state tuple/indexing
#### D. UX Polish
- [ ] Confirmation prompts for unsaved changes, file overwrites
- [ ] Modal system for dialogs, errors, file pickers
- [ ] Cursor + hover feedback for UI elements
- [ ] Screenshot watermark (settings/seed/episode #)
#### E. Safety / Resilience
- [ ] Crash screen + auto dump (`data/crash_dump.txt` with config/env snapshot)
- [ ] Versioned save formats for env snapshot/config
#### F. Packaging & Dev Workflow
- [ ] `python -m viewers` entry via `viewers/__main__.py`
- [ ] Pre-commit/formatting: `ruff` + `black`
- [ ] CI smoke test with dummy video driver (`SDL_VIDEODRIVER=dummy`)

### Must-Add for Scaling
- [ ] Separate sim stepping from rendering, cache overlays (policy/heatmap)

### Recommended Build Order
1. [ ] Scene system + Main Menu skeleton
2. [ ] SimulationScene refactor
3. [ ] SettingsScene (wired to AppConfig)
4. [ ] Overlays: Stats + Help
5. [ ] Save/load Q-table + toast feedback
6. [ ] Env snapshot save/load
7. [ ] Policy arrows + Q hover panel
8. [ ] Accessibility themes + font scaling
9. [ ] Export + docs + tests
10. [ ] Quality-of-life, telemetry, performance, UX, safety, packaging


## Polished Viewer

Run:

```sh
python -m viewers
```
