from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from collections import deque

import pygame

from core.env import GridSurvivalEnv
from core.qlearning import QLearningAgent, QLearningConfig
from viewers.io.export import export_csv, export_json
from viewers.scenes.sim import SimulationScene
from viewers.scenes.settings import SettingsScene
from viewers.ui.modals import FileDialogModal, ConfirmDialog, TrainingSetupWizard
from viewers.ui.widgets import Button, Slider, Toggle, FocusManager, Label


class TrainingScene:
    def __init__(self) -> None:
        self.focus = FocusManager()
        self.widgets: list = []
        self.training = False
        self.episodes_done = 0
        self.last_eval = ""
        self.agent: QLearningAgent | None = None
        self.env: GridSurvivalEnv | None = None
        self._avg_rewards = deque(maxlen=50)
        self._avg_steps = deque(maxlen=50)
        self._avg_foods = deque(maxlen=50)
        self.map_stats: dict[int, dict[str, int]] = {}
        self._recent_success: deque[int] = deque(maxlen=50)
        self.curriculum_limit: int | None = None
        self.last_train_settings: dict[str, object] | None = None
        self.last_map_eval: list[dict[str, object]] | None = None

    def _build_env_agent(self, app) -> None:
        cfg = app.cfg
        self._sync_curriculum(app)
        self.last_train_settings = self._collect_env_settings(cfg)
        self.env = GridSurvivalEnv(
            width=int(cfg.w),
            height=int(cfg.h),
            n_hazards=int(cfg.hazards),
            energy_start=int(cfg.energy_start),
            energy_food_gain=int(cfg.energy_food),
            energy_step_cost=int(cfg.energy_step),
            energy_max=int(getattr(cfg, "energy_max", 0)),
            seed=int(cfg.seed),
            level_mode=str(getattr(cfg, "level_mode", "preset")),
            level_index=int(getattr(cfg, "level_index", 0)),
            level_cycle=bool(getattr(cfg, "level_cycle", True)),
            level_limit=self.curriculum_limit,
            n_walls=int(getattr(cfg, "n_walls", 18)),
            n_traps=int(getattr(cfg, "n_traps", int(cfg.hazards))),
            food_enabled=bool(getattr(cfg, "food_enabled", True)),
            placement_difficulty=str(getattr(cfg, "placement_difficulty", "medium")),
        )
        qcfg = QLearningConfig(
            alpha=float(cfg.alpha),
            gamma=float(cfg.gamma),
            eps_start=float(cfg.eps_start),
            eps_end=float(cfg.eps_end),
            eps_decay_steps=int(cfg.eps_decay),
        )
        if self.agent is None:
            self.agent = QLearningAgent(n_actions=4, cfg=qcfg, seed=int(cfg.seed))

    @staticmethod
    def _collect_env_settings(cfg) -> dict[str, object]:
        return {
            "width": int(cfg.w),
            "height": int(cfg.h),
            "n_hazards": int(cfg.hazards),
            "energy_start": int(cfg.energy_start),
            "energy_food_gain": int(cfg.energy_food),
            "energy_step_cost": int(cfg.energy_step),
            "energy_max": int(getattr(cfg, "energy_max", 0)),
            "seed": int(cfg.seed),
            "level_mode": str(getattr(cfg, "level_mode", "preset")),
            "level_index": int(getattr(cfg, "level_index", 0)),
            "level_cycle": bool(getattr(cfg, "level_cycle", True)),
            "n_walls": int(getattr(cfg, "n_walls", 18)),
            "n_traps": int(getattr(cfg, "n_traps", int(cfg.hazards))),
            "food_enabled": bool(getattr(cfg, "food_enabled", True)),
            "placement_difficulty": str(getattr(cfg, "placement_difficulty", "medium")),
        }

    def _run_episode_env(self, env: GridSurvivalEnv, agent: QLearningAgent, max_steps: int, train: bool) -> tuple[int, int, float, str]:
        obs = env.reset()
        total_reward = 0.0
        foods = 0
        terminal = ""
        done = False
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
                terminal = res.info.get("terminal", "")
                done = True
                break
        if not done:
            terminal = "timeout"
        return env.steps, foods, float(total_reward), terminal

    def _run_episode(self, train: bool, max_steps: int) -> tuple[int, int, float, str]:
        assert self.env is not None and self.agent is not None
        return self._run_episode_env(self.env, self.agent, max_steps=max_steps, train=train)

    def _eval_stats(self, episodes: int, max_steps: int) -> str:
        steps_list = []
        foods_list = []
        rewards_list = []
        terminals: dict[str, int] = {}
        for _ in range(episodes):
            steps, foods, total, term = self._run_episode(train=False, max_steps=max_steps)
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

    def _curriculum_active(self, cfg) -> bool:
        return bool(getattr(cfg, "train_curriculum", False)) and str(getattr(cfg, "level_mode", "preset")) == "preset"

    def _sync_curriculum(self, app) -> None:
        cfg = app.cfg
        if not self._curriculum_active(cfg):
            self.curriculum_limit = None
            if self.env is not None:
                self.env.level_limit = None
            return
        total = GridSurvivalEnv.preset_level_count()
        if total <= 0:
            self.curriculum_limit = None
            return
        start = max(1, min(int(getattr(cfg, "train_curriculum_start", 5)), total))
        if self.curriculum_limit is None:
            self.curriculum_limit = start
        else:
            self.curriculum_limit = max(1, min(self.curriculum_limit, total))
        if self.env is not None:
            self.env.level_limit = self.curriculum_limit

    def _maybe_promote_curriculum(self, app) -> None:
        cfg = app.cfg
        if not self._curriculum_active(cfg):
            return
        total = GridSurvivalEnv.preset_level_count()
        if not total or self.curriculum_limit is None:
            return
        window = max(1, int(getattr(cfg, "train_curriculum_window", 50)))
        if len(self._recent_success) < window:
            return
        rate = sum(self._recent_success) / max(1, len(self._recent_success))
        threshold = float(getattr(cfg, "train_curriculum_threshold", 0.8))
        if rate < threshold:
            return
        if self.curriculum_limit >= total:
            return
        step = max(1, int(getattr(cfg, "train_curriculum_step", 5)))
        self.curriculum_limit = min(total, self.curriculum_limit + step)
        if self.env is not None:
            self.env.level_limit = self.curriculum_limit
        self._recent_success.clear()
        if self.agent is not None:
            rewind = float(getattr(cfg, "train_curriculum_eps_rewind", 0.5))
            rewind = max(0.0, min(1.0, rewind))
            if rewind < 1.0:
                self.agent.total_steps = int(self.agent.total_steps * rewind)
        self._save_checkpoint(app, label=f"Curriculum unlocked: {self.curriculum_limit}/{total} levels")

    def _layout(self, app) -> None:
        w, h = app.screen.get_size()
        scale = float(app.theme.ui_scale)
        pad = int(50 * scale)
        gap = int(8 * scale)
        row_h = int(44 * scale)
        left_w = int(w * 0.52)
        x0 = pad
        y0 = int(110 * scale)

        left_panel_w = left_w - pad
        col_w = int((left_panel_w - gap) / 2)
        x_left = x0
        x_right = x0 + col_w + gap

        def rect_left(i: int) -> pygame.Rect:
            return pygame.Rect(x_left, y0 + i * (row_h + gap), col_w, row_h)

        def rect_right(i: int) -> pygame.Rect:
            return pygame.Rect(x_right, y0 + i * (row_h + gap), col_w, row_h)

        cfg = app.cfg
        items: list = []
        left_i = 0
        right_i = 0

        items.append(Label(rect_left(left_i), "Training")); left_i += 1
        items.append(Slider(rect_left(left_i), "Episodes", 50, 10000, 50,
                            lambda: float(cfg.train_episodes), lambda v: setattr(cfg, "train_episodes", int(v)), fmt="{:.0f}")); left_i += 1
        items.append(Slider(rect_left(left_i), "Max steps/ep", 50, 800, 10,
                            lambda: float(cfg.train_max_steps), lambda v: setattr(cfg, "train_max_steps", int(v)), fmt="{:.0f}")); left_i += 1
        items.append(Slider(rect_left(left_i), "Eval every", 0, 1000, 50,
                            lambda: float(cfg.train_eval_every), lambda v: setattr(cfg, "train_eval_every", int(v)), fmt="{:.0f}")); left_i += 1
        items.append(Slider(rect_left(left_i), "Eval episodes", 10, 200, 10,
                            lambda: float(cfg.train_eval_episodes), lambda v: setattr(cfg, "train_eval_episodes", int(v)), fmt="{:.0f}")); left_i += 1
        items.append(Slider(rect_left(left_i), "Speed (eps/update)", 1, 50, 1,
                            lambda: float(cfg.train_speed), lambda v: setattr(cfg, "train_speed", int(v)), fmt="{:.0f}")); left_i += 1
        items.append(Toggle(rect_left(left_i), "Autosave on finish",
                            lambda: bool(cfg.train_autosave), lambda b: setattr(cfg, "train_autosave", bool(b)))); left_i += 1
        items.append(Slider(rect_left(left_i), "Checkpoint every", 0, 5000, 100,
                            lambda: float(getattr(cfg, "train_checkpoint_every", 0)),
                            lambda v: setattr(cfg, "train_checkpoint_every", int(v)), fmt="{:.0f}")); left_i += 1

        items.append(Button(rect_left(left_i), "Test all maps", lambda: self._eval_all_maps(app))); left_i += 1
        items.append(Button(rect_left(left_i), "Export map stats", lambda: self._open_export_map_stats(app))); left_i += 1
        items.append(Button(rect_left(left_i), "Start / Resume", lambda: self._start(app))); left_i += 1
        items.append(Button(rect_left(left_i), "Pause", lambda: self._pause(app))); left_i += 1
        items.append(Button(rect_left(left_i), "Reset training (keep file)", lambda: self._confirm_reset(app))); left_i += 1

        items.append(Label(rect_left(left_i), "Files")); left_i += 1
        items.append(Button(rect_left(left_i), "New Q-table (guided)", lambda: self._open_new_qtable_wizard(app))); left_i += 1
        items.append(Button(rect_left(left_i), "Save Q-table", lambda: self._open_save(app))); left_i += 1
        items.append(Button(rect_left(left_i), "Load Q-table", lambda: self._open_load(app))); left_i += 1
        items.append(Button(rect_left(left_i), "Delete Q-table file", lambda: self._open_delete_file(app))); left_i += 1
        items.append(Button(rect_left(left_i), "Delete all training files", lambda: self._confirm_delete_training(app))); left_i += 1

        total_levels = GridSurvivalEnv.preset_level_count()
        max_levels = max(1, total_levels)
        items.append(Label(rect_right(right_i), "Curriculum")); right_i += 1
        items.append(Toggle(rect_right(right_i), "Curriculum mode",
                            lambda: bool(getattr(cfg, "train_curriculum", False)),
                            lambda b: setattr(cfg, "train_curriculum", bool(b)))); right_i += 1
        items.append(Slider(rect_right(right_i), "Start levels", 1, max_levels, 1,
                            lambda: float(getattr(cfg, "train_curriculum_start", 5)),
                            lambda v: setattr(cfg, "train_curriculum_start", int(v)), fmt="{:.0f}")); right_i += 1
        items.append(Slider(rect_right(right_i), "Add levels", 1, max_levels, 1,
                            lambda: float(getattr(cfg, "train_curriculum_step", 5)),
                            lambda v: setattr(cfg, "train_curriculum_step", int(v)), fmt="{:.0f}")); right_i += 1
        items.append(Slider(rect_right(right_i), "Success threshold", 0.5, 1.0, 0.05,
                            lambda: float(getattr(cfg, "train_curriculum_threshold", 0.8)),
                            lambda v: setattr(cfg, "train_curriculum_threshold", float(v)), fmt="{:.2f}")); right_i += 1
        items.append(Slider(rect_right(right_i), "Window (episodes)", 10, 200, 10,
                            lambda: float(getattr(cfg, "train_curriculum_window", 50)),
                            lambda v: setattr(cfg, "train_curriculum_window", int(v)), fmt="{:.0f}")); right_i += 1
        items.append(Slider(rect_right(right_i), "Eps rewind", 0.0, 1.0, 0.05,
                            lambda: float(getattr(cfg, "train_curriculum_eps_rewind", 0.5)),
                            lambda v: setattr(cfg, "train_curriculum_eps_rewind", float(v)), fmt="{:.2f}")); right_i += 1
        items.append(Toggle(rect_right(right_i), "Use training settings",
                            lambda: bool(getattr(cfg, "train_use_settings_for_play", True)),
                            lambda b: setattr(cfg, "train_use_settings_for_play", bool(b)))); right_i += 1

        items.append(Label(rect_right(right_i), "3D View")); right_i += 1
        total_levels = GridSurvivalEnv.preset_level_count()
        def cycle_viz_level() -> None:
            if total_levels <= 0:
                cfg.viz_level_filter = -1
                app.toast.push("3D level: All")
                self._layout(app)
                return
            cur = int(getattr(cfg, "viz_level_filter", -1))
            if cur < 0:
                cur = 0
            elif cur >= total_levels - 1:
                cur = -1
            else:
                cur += 1
            cfg.viz_level_filter = cur
            label = "All" if cur < 0 else f"{cur + 1}/{total_levels}"
            app.toast.push(f"3D level: {label}")
            self._layout(app)

        level_label = "All"
        if total_levels > 0 and int(getattr(cfg, "viz_level_filter", -1)) >= 0:
            cur_level = int(cfg.viz_level_filter)
            level_label = f"L{cur_level} ({cur_level + 1}/{total_levels})"
        items.append(Button(rect_right(right_i), f"3D level: {level_label}", cycle_viz_level)); right_i += 1
        items.append(Slider(rect_right(right_i), "3D min visits", 0, 50, 1,
                            lambda: float(getattr(cfg, "viz_min_visits", 0)),
                            lambda v: setattr(cfg, "viz_min_visits", int(v)), fmt="{:.0f}")); right_i += 1
        def cycle_viz_max_points() -> None:
            options = [5000, 10000, 20000]
            cur = int(getattr(cfg, "viz_max_points", 20000))
            if cur not in options:
                cur = 20000
            cfg.viz_max_points = options[(options.index(cur) + 1) % len(options)]
            app.toast.push(f"3D max points: {cfg.viz_max_points}")
            self._layout(app)

        items.append(Button(rect_right(right_i), f"3D max points: {int(getattr(cfg, 'viz_max_points', 20000))}", cycle_viz_max_points)); right_i += 1

        def cycle_viz_color() -> None:
            modes = ["value", "action", "confidence", "energy", "level", "d_goal", "d_food", "visits"]
            cur = str(getattr(cfg, "viz_color", "value"))
            if cur not in modes:
                cur = "value"
            cfg.viz_color = modes[(modes.index(cur) + 1) % len(modes)]
            app.toast.push(f"3D color: {cfg.viz_color}")
            self._layout(app)

        items.append(Button(rect_right(right_i), f"3D color: {getattr(cfg, 'viz_color', 'value')}", cycle_viz_color)); right_i += 1
        def cycle_viz_size() -> None:
            modes = ["count", "confidence", "value", "none"]
            cur = str(getattr(cfg, "viz_size", "count"))
            if cur not in modes:
                cur = "count"
            cfg.viz_size = modes[(modes.index(cur) + 1) % len(modes)]
            app.toast.push(f"3D size: {cfg.viz_size}")
            self._layout(app)

        items.append(Button(rect_right(right_i), f"3D size: {getattr(cfg, 'viz_size', 'count')}", cycle_viz_size)); right_i += 1
        items.append(Toggle(rect_right(right_i), "3D include level feature",
                            lambda: bool(getattr(cfg, "viz_level_feature", False)),
                            lambda b: setattr(cfg, "viz_level_feature", bool(b)))); right_i += 1
        items.append(Toggle(rect_right(right_i), "3D action figure",
                            lambda: bool(getattr(cfg, "viz_action_figure", True)),
                            lambda b: setattr(cfg, "viz_action_figure", bool(b)))); right_i += 1

        items.append(Button(rect_right(right_i), "Env + Agent settings",
                            lambda: app.push(SettingsScene(mode="training")))); right_i += 1
        items.append(Button(rect_right(right_i), "Open 3D state view",
                            lambda: self._open_state_space(app, mode="state"))); right_i += 1
        items.append(Button(rect_right(right_i), "Open 3D Q-space",
                            lambda: self._open_state_space(app, mode="qvec"))); right_i += 1
        items.append(Button(rect_right(right_i), "Play in simulation", lambda: self._play(app))); right_i += 1

        self.widgets = items
        self.focus.set(self.widgets)

    def _start(self, app) -> None:
        if self.env is None or self.agent is None:
            self._build_env_agent(app)
        else:
            self._sync_curriculum(app)
            self.last_train_settings = self._collect_env_settings(app.cfg)
        if self.episodes_done >= int(getattr(app.cfg, "train_episodes", 0)):
            self.episodes_done = 0
            self._avg_rewards.clear()
            self._avg_steps.clear()
            self._avg_foods.clear()
            self._recent_success.clear()
            self.map_stats.clear()
            self.last_eval = ""
            app.toast.push("Restarted training run (kept Q-table)")
        self.training = True

    def _pause(self, app) -> None:
        self.training = False
        if self.env is not None:
            self.last_train_settings = self._collect_env_settings(app.cfg)

    def _reset(self, app) -> None:
        self.training = False
        self.episodes_done = 0
        self.agent = None
        self.env = None
        self._avg_rewards.clear()
        self._avg_steps.clear()
        self._avg_foods.clear()
        self.map_stats.clear()
        self._recent_success.clear()
        self.curriculum_limit = None
        self.last_map_eval = None
        self.last_eval = ""
        self._build_env_agent(app)

    def _open_delete_file(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center

        def on_confirm(path: str) -> None:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    if path == getattr(app.cfg, "qtable_path", ""):
                        app.cfg.qtable_path = os.path.join("data", "qtable_saved.pkl")
                    app.toast.push(f"Deleted file -> {path}")
                else:
                    app.toast.push("File does not exist")
            except Exception as exc:
                app.toast.push(f"Delete failed: {exc}")

        modal = FileDialogModal(
            rect,
            "Delete Q-table file",
            on_confirm,
            lambda: None,
            initial_path=getattr(app.cfg, "qtable_path", os.path.join("data", "qtable_saved.pkl")),
            must_exist=True,
            ext_filter=".pkl",
        )
        app.push_modal(modal)

    def _confirm_delete_training(self, app) -> None:
        rect = pygame.Rect(0, 0, int(520 * app.theme.ui_scale), int(200 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center

        def on_confirm() -> None:
            self._delete_training_files(app)

        modal = ConfirmDialog(
            rect,
            "Delete training files",
            "Delete Q-tables, run stats, snapshots, and run history in data/?",
            on_confirm,
            lambda: None,
        )
        app.push_modal(modal)

    def _confirm_reset(self, app) -> None:
        rect = pygame.Rect(0, 0, int(520 * app.theme.ui_scale), int(200 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center

        def on_confirm() -> None:
            self._reset(app)
            app.toast.push("Training reset (Q-table cleared)")

        modal = ConfirmDialog(
            rect,
            "Reset training",
            "This clears the in-memory Q-table and stats (file stays).",
            on_confirm,
            lambda: None,
        )
        app.push_modal(modal)

    def _open_new_qtable_wizard(self, app) -> None:
        rect = pygame.Rect(0, 0, int(900 * app.theme.ui_scale), int(640 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center

        def on_apply() -> None:
            if bool(getattr(app.cfg, "train_curriculum", False)):
                app.cfg.level_mode = "preset"
                app.cfg.level_cycle = True
            app.cfg.train_use_settings_for_play = True
            default_path = self._default_new_qtable_path()
            dialog_rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
            dialog_rect.center = app.screen.get_rect().center

            def on_confirm(path: str) -> None:
                app.cfg.qtable_path = path
                self._reset(app)
                if self.agent:
                    self.agent.save(path)
                app.toast.push(f"New Q-table: {os.path.basename(path)}")

            modal = FileDialogModal(
                dialog_rect,
                "Save new Q-table as",
                on_confirm,
                lambda: None,
                initial_path=default_path,
            )
            app.push_modal(modal)

        modal = TrainingSetupWizard(
            rect,
            app.cfg,
            on_apply,
            lambda: None,
        )
        app.push_modal(modal)

    def _default_new_qtable_path(self) -> str:
        os.makedirs("data", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join("data", f"qtable_{stamp}.pkl")
        if not os.path.exists(base):
            return base
        i = 1
        while True:
            path = os.path.join("data", f"qtable_{stamp}_{i}.pkl")
            if not os.path.exists(path):
                return path
            i += 1

    def _delete_training_files(self, app) -> None:
        data_dir = "data"
        removed = 0
        targets = {
            "run_history.jsonl",
            "run_stats.json",
            "run_stats.csv",
            "env_snapshot.json",
        }
        if os.path.isdir(data_dir):
            for name in os.listdir(data_dir):
                path = os.path.join(data_dir, name)
                if not os.path.isfile(path):
                    continue
                if name in targets or os.path.splitext(name)[1].lower() == ".pkl":
                    try:
                        os.remove(path)
                        removed += 1
                    except Exception:
                        pass
        app.cfg.qtable_path = os.path.join("data", "qtable_saved.pkl")
        self._reset(app)
        if removed:
            app.toast.push(f"Deleted {removed} training files")
        else:
            app.toast.push("No training files found")

    def _save_checkpoint(self, app, label: str = "Checkpoint saved") -> None:
        cfg = app.cfg
        if not self.agent:
            return
        path = getattr(cfg, "qtable_path", os.path.join("data", "qtable_saved.pkl"))
        try:
            self.agent.save(path)
            if label:
                app.toast.push(label)
        except Exception as exc:
            app.toast.push(f"Checkpoint failed: {exc}")

    def _map_stats_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        source = self.last_map_eval if self.last_map_eval else []
        if source:
            return list(source)
        for level_id, stats in self.map_stats.items():
            plays = max(1, stats.get("plays", 0))
            goals = stats.get("goals", 0)
            foods = stats.get("foods", 0)
            template = GridSurvivalEnv.get_level_template(level_id)
            rows.append(
                {
                    "level_id": level_id,
                    "level_index": level_id + 1,
                    "name": (template or {}).get("name", f"Level {level_id + 1}"),
                    "plays": plays,
                    "goals": goals,
                    "foods": foods,
                    "success_rate": goals / plays,
                    "food_rate": foods / plays,
                }
            )
        return rows

    def _eval_all_maps(self, app) -> None:
        if not self.agent:
            app.toast.push("Train or load a Q-table first.")
            return
        self.training = False
        total = GridSurvivalEnv.preset_level_count()
        if total <= 0:
            app.toast.push("No preset maps available.")
            return
        cfg = app.cfg
        settings = self.last_train_settings or self._collect_env_settings(cfg)
        max_steps = int(cfg.train_max_steps)
        episodes = max(1, int(cfg.train_eval_episodes))
        rows: list[dict[str, object]] = []
        for level_id in range(total):
            env_kwargs = dict(settings)
            env_kwargs.update(
                {
                    "level_mode": "preset",
                    "level_index": level_id,
                    "level_cycle": False,
                    "level_limit": None,
                }
            )
            env = GridSurvivalEnv(**env_kwargs)
            steps_list: list[int] = []
            foods_list: list[int] = []
            rewards_list: list[float] = []
            goals = 0
            for _ in range(episodes):
                steps, foods, total_reward, terminal = self._run_episode_env(env, self.agent, max_steps=max_steps, train=False)
                steps_list.append(steps)
                foods_list.append(foods)
                rewards_list.append(total_reward)
                if terminal == "goal":
                    goals += 1
            template = GridSurvivalEnv.get_level_template(level_id)
            rows.append(
                {
                    "level_id": level_id,
                    "level_index": level_id + 1,
                    "name": (template or {}).get("name", f"Level {level_id + 1}"),
                    "episodes": episodes,
                    "plays": episodes,
                    "goals": goals,
                    "avg_steps": statistics.mean(steps_list) if steps_list else 0.0,
                    "avg_foods": statistics.mean(foods_list) if foods_list else 0.0,
                    "avg_reward": statistics.mean(rewards_list) if rewards_list else 0.0,
                    "success_rate": goals / max(1, episodes),
                }
            )
        self.last_map_eval = rows
        avg_success = sum(r["success_rate"] for r in rows) / max(1, len(rows))
        app.toast.push(f"Map eval done: {avg_success * 100:.0f}% avg success")

    def _open_export_map_stats(self, app) -> None:
        rows = self._map_stats_rows()
        if not rows:
            app.toast.push("No map stats to export yet.")
            return
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center

        def on_confirm(path: str) -> None:
            try:
                export_json(rows, path)
                export_csv(rows, os.path.splitext(path)[0] + ".csv")
                app.toast.push(f"Exported map stats -> {path}")
            except Exception as exc:
                app.toast.push(f"Export failed: {exc}")

        modal = FileDialogModal(
            rect,
            "Export map stats",
            on_confirm,
            lambda: None,
            initial_path=os.path.join("data", "map_stats.json"),
        )
        app.push_modal(modal)

    def _open_save(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center

        def on_confirm(path: str) -> None:
            if self.agent:
                self.agent.save(path)
                app.cfg.qtable_path = path
                app.toast.push(f"Saved Q-table -> {path}")

        modal = FileDialogModal(
            rect,
            "Save Q-table",
            on_confirm,
            lambda: None,
            initial_path=getattr(app.cfg, "qtable_path", os.path.join("data", "qtable_saved.pkl")),
        )
        app.push_modal(modal)

    def _open_load(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center

        def on_confirm(path: str) -> None:
            try:
                self.agent = QLearningAgent.load(path, seed=int(app.cfg.seed))
                app.cfg.qtable_path = path
                app.toast.push(f"Loaded Q-table <- {path}")
            except Exception as exc:
                app.toast.push(f"Load failed: {exc}")

        modal = FileDialogModal(
            rect,
            "Load Q-table",
            on_confirm,
            lambda: None,
            initial_path=getattr(app.cfg, "qtable_path", os.path.join("data", "qtable_saved.pkl")),
            must_exist=True,
        )
        app.push_modal(modal)

    def _play(self, app) -> None:
        if self.agent is None:
            app.toast.push("No Q-table loaded. Train or load first.")
            return
        env_overrides = None
        if bool(getattr(app.cfg, "train_use_settings_for_play", True)):
            env_overrides = self.last_train_settings or self._collect_env_settings(app.cfg)
        app.push(SimulationScene(agent_override=self.agent, env_overrides=env_overrides))

    def _open_state_space(self, app, mode: str = "state") -> None:
        if self.agent is None:
            app.toast.push("Train or load a Q-table first.")
            return
        qpath = getattr(app.cfg, "qtable_path", "") or ""
        needs_save = not qpath or not os.path.isfile(qpath)
        if needs_save:
            qpath = os.path.join("data", "qtable_visualize.pkl")
            try:
                os.makedirs(os.path.dirname(qpath) or ".", exist_ok=True)
                self.agent.save(qpath)
                app.toast.push(f"Saved Q-table for visualization -> {qpath}")
            except Exception as exc:
                app.toast.push(f"Save failed: {exc}")
                return
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        script_path = os.path.join(repo_root, "scripts", "plot_state_space_3d.py")
        if not os.path.exists(script_path):
            app.toast.push("3D script not found.")
            return
        try:
            args = [sys.executable, "-m", "scripts.plot_state_space_3d", "--load", qpath, "--mode", mode]
            color = str(getattr(app.cfg, "viz_color", "value") or "value")
            args += ["--color", color]
            size = str(getattr(app.cfg, "viz_size", "count") or "count")
            args += ["--size", size]
            min_visits = int(getattr(app.cfg, "viz_min_visits", 0))
            if min_visits > 0:
                args += ["--min-visits", str(min_visits)]
            max_points = int(getattr(app.cfg, "viz_max_points", 20000))
            if max_points > 0:
                args += ["--max-points", str(max_points)]
            level_filter = int(getattr(app.cfg, "viz_level_filter", -1))
            if level_filter >= 0:
                args += ["--level", str(level_filter)]
            if bool(getattr(app.cfg, "viz_level_feature", False)):
                args.append("--level-feature")
            if bool(getattr(app.cfg, "viz_action_figure", True)):
                args.append("--action-figure")
            subprocess.Popen(args, cwd=repo_root)
        except Exception as exc:
            app.toast.push(f"Open failed: {exc}")

    def handle_event(self, app, event: pygame.event.Event) -> None:
        if not self.widgets:
            self._layout(app)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_TAB, pygame.K_DOWN):
                self.focus.next()
            elif event.key == pygame.K_UP:
                self.focus.prev()
            elif event.key == pygame.K_ESCAPE:
                app.pop()
                return
        focused = self.focus.focused()
        for w in self.widgets:
            w.handle_event(event, focused=(w is focused))

    def update(self, app, dt: float) -> None:
        if not self.training:
            return
        if self.env is None or self.agent is None:
            self._build_env_agent(app)
        cfg = app.cfg
        if self._curriculum_active(cfg):
            window = max(1, int(getattr(cfg, "train_curriculum_window", 50)))
            if self._recent_success.maxlen != window:
                self._recent_success = deque(list(self._recent_success), maxlen=window)
            self._sync_curriculum(app)
        for _ in range(int(cfg.train_speed)):
            if self.episodes_done >= int(cfg.train_episodes):
                self.training = False
                if cfg.train_autosave and self.agent:
                    path = getattr(cfg, "qtable_path", os.path.join("data", "qtable_saved.pkl"))
                    self.agent.save(path)
                    app.toast.push(f"Autosaved Q-table -> {path}")
                break
            steps, foods, total, terminal = self._run_episode(train=True, max_steps=int(cfg.train_max_steps))
            self._avg_steps.append(steps)
            self._avg_foods.append(foods)
            self._avg_rewards.append(total)
            self.episodes_done += 1
            checkpoint_every = int(getattr(cfg, "train_checkpoint_every", 0))
            if checkpoint_every > 0 and self.episodes_done % checkpoint_every == 0:
                self._save_checkpoint(app, label=f"Checkpoint saved ({self.episodes_done})")
            if self.env is not None:
                level_id = int(self.env.level_id)
                stats = self.map_stats.setdefault(level_id, {"plays": 0, "goals": 0, "foods": 0})
                stats["plays"] += 1
                if foods > 0:
                    stats["foods"] += 1
                if terminal == "goal":
                    stats["goals"] += 1
            if self._curriculum_active(cfg) and self.env is not None:
                success = 1 if terminal == "goal" else 0
                self._recent_success.append(success)
                self._maybe_promote_curriculum(app)
            if int(cfg.train_eval_every) > 0 and self.episodes_done % int(cfg.train_eval_every) == 0:
                self.last_eval = self._eval_stats(int(cfg.train_eval_episodes), int(cfg.train_max_steps))

    def render(self, app, screen: pygame.Surface) -> None:
        if not self.widgets:
            self._layout(app)
        self._sync_curriculum(app)
        w, h = screen.get_size()
        if app.theme.ui_style == "pixel":
            screen.fill(app.theme.palette.bg)
        else:
            bg = pygame.Surface((w, h))
            top = app.theme.palette.bg
            bottom = app.theme.palette.grid0
            for y in range(h):
                t = y / max(1, h - 1)
                c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
                pygame.draw.line(bg, c, (0, y), (w, y))
            screen.blit(bg, (0, 0))
        title_font = app.theme.font(int(app.theme.font_size_title * 0.7 * app.theme.ui_scale))
        font = app.theme.font(int(app.theme.font_size * app.theme.ui_scale))
        small = app.theme.font(int(app.theme.font_size * 0.85 * app.theme.ui_scale))

        title = title_font.render("Training / Learning", True, app.theme.palette.fg)
        screen.blit(title, (int(50 * app.theme.ui_scale), int(40 * app.theme.ui_scale)))

        focused = self.focus.focused()
        if app.theme.ui_style != "pixel" and self.widgets:
            pad = int(24 * app.theme.ui_scale)
            min_x = min(wi.rect.x for wi in self.widgets)
            max_x = max(wi.rect.right for wi in self.widgets)
            min_y = min(wi.rect.y for wi in self.widgets)
            max_y = max(wi.rect.bottom for wi in self.widgets)
            panel_rect = pygame.Rect(min_x - pad, min_y - pad, (max_x - min_x) + 2 * pad, (max_y - min_y) + 2 * pad)
            shadow = pygame.Surface((panel_rect.w, panel_rect.h), pygame.SRCALPHA)
            pygame.draw.rect(shadow, (0, 0, 0, 80), shadow.get_rect(), border_radius=int(16 * app.theme.ui_scale))
            screen.blit(shadow, (panel_rect.x + int(6 * app.theme.ui_scale), panel_rect.y + int(8 * app.theme.ui_scale)))
            app.theme.draw_gradient_panel(screen, panel_rect, app.theme.palette.panel, app.theme.palette.grid1, border_radius=int(16 * app.theme.ui_scale))
        for w in self.widgets:
            w.draw(screen, app.theme, focused=(w is focused))

        # Right-side stats panel
        pad = int(50 * app.theme.ui_scale)
        stats_x = int(screen.get_width() * 0.56)
        stats_y = int(110 * app.theme.ui_scale)
        stats_w = screen.get_width() - stats_x - pad
        stats_h = screen.get_height() - stats_y - pad
        panel = pygame.Rect(stats_x, stats_y, stats_w, stats_h)
        if app.theme.ui_style == "pixel":
            app.theme.draw_rounded_panel(screen, panel, color=app.theme.palette.panel, border_radius=0)
        else:
            app.theme.draw_gradient_panel(screen, panel, app.theme.palette.panel, app.theme.palette.grid1, border_radius=int(16 * app.theme.ui_scale))

        y = panel.y + int(16 * app.theme.ui_scale)
        level_mode = getattr(app.cfg, "level_mode", "preset")
        if level_mode == "preset":
            total_levels = GridSurvivalEnv.preset_level_count()
            if getattr(app.cfg, "level_cycle", False) and total_levels > 0:
                level_label = f"Cycle ({total_levels})"
            elif total_levels > 0:
                level_label = f"{int(app.cfg.level_index) + 1}/{total_levels}"
            else:
                level_label = str(int(app.cfg.level_index) + 1)
        else:
            level_label = "Random"
        lines = [
            f"Mode: {level_mode}  Level: {level_label}",
            f"Status: {'TRAINING' if self.training else 'PAUSED'}",
            f"Episodes: {self.episodes_done}/{app.cfg.train_episodes}",
        ]
        if self.agent is not None:
            lines.append(f"Epsilon: {self.agent.epsilon():.3f}")
            lines.append(f"Q-states: {len(self.agent.Q)}")

        avg_steps = statistics.mean(self._avg_steps) if self._avg_steps else 0.0
        avg_foods = statistics.mean(self._avg_foods) if self._avg_foods else 0.0
        avg_reward = statistics.mean(self._avg_rewards) if self._avg_rewards else 0.0
        lines += [
            f"Avg steps (last {len(self._avg_steps)}): {avg_steps:.1f}",
            f"Avg foods (last {len(self._avg_foods)}): {avg_foods:.2f}",
            f"Avg reward (last {len(self._avg_rewards)}): {avg_reward:.2f}",
        ]
        if self._curriculum_active(app.cfg):
            total_levels = GridSurvivalEnv.preset_level_count()
            limit = self.curriculum_limit or total_levels
            lines.append(f"Curriculum: ON ({limit}/{total_levels})")
            if self._recent_success:
                rate = sum(self._recent_success) / max(1, len(self._recent_success))
                lines.append(f"Recent success ({len(self._recent_success)}): {rate * 100:.0f}%")
        else:
            lines.append("Curriculum: OFF")

        for line in lines:
            surf = font.render(line, True, app.theme.palette.fg)
            screen.blit(surf, (panel.x + int(16 * app.theme.ui_scale), y))
            y += int(28 * app.theme.ui_scale)

        rows = self._map_stats_rows()
        if rows:
            y += int(6 * app.theme.ui_scale)
            title = small.render("Worst maps (success rate):", True, app.theme.palette.muted)
            screen.blit(title, (panel.x + int(16 * app.theme.ui_scale), y))
            y += int(22 * app.theme.ui_scale)
            entries = []
            for row in rows:
                level_id = int(row.get("level_id", 0))
                rate = float(row.get("success_rate", 0.0))
                name = str(row.get("name", f"Level {level_id + 1}"))
                plays = int(row.get("plays", row.get("episodes", 0)) or 0)
                goals = int(row.get("goals", int(rate * max(1, plays))))
                entries.append((rate, level_id, goals, plays, name))
            entries.sort(key=lambda item: item[0])
            for rate, level_id, goals, plays, name in entries[:4]:
                label = f"#{level_id + 1} {rate * 100:.0f}% ({goals}/{plays}) {name}"
                s = small.render(label, True, app.theme.palette.fg)
                screen.blit(s, (panel.x + int(16 * app.theme.ui_scale), y))
                y += int(22 * app.theme.ui_scale)

        if self.last_eval:
            y += int(10 * app.theme.ui_scale)
            eval_title = font.render("Last eval:", True, app.theme.palette.muted)
            screen.blit(eval_title, (panel.x + int(16 * app.theme.ui_scale), y))
            y += int(26 * app.theme.ui_scale)
            for part in self.last_eval.split():
                s = small.render(part, True, app.theme.palette.fg)
                screen.blit(s, (panel.x + int(16 * app.theme.ui_scale), y))
                y += int(22 * app.theme.ui_scale)
