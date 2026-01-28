from __future__ import annotations

import os
import statistics
from collections import deque

import pygame

from core.env import GridSurvivalEnv
from core.qlearning import QLearningAgent, QLearningConfig
from viewers.scenes.sim import SimulationScene
from viewers.ui.modals import FileDialogModal
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

    def _build_env_agent(self, app) -> None:
        cfg = app.cfg
        self.env = GridSurvivalEnv(
            width=int(cfg.w),
            height=int(cfg.h),
            n_hazards=int(cfg.hazards),
            energy_start=int(cfg.energy_start),
            energy_food_gain=int(cfg.energy_food),
            energy_step_cost=int(cfg.energy_step),
            seed=int(cfg.seed),
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

    def _run_episode(self, train: bool, max_steps: int) -> tuple[int, int, float, str]:
        assert self.env is not None and self.agent is not None
        obs = self.env.reset()
        total_reward = 0.0
        foods = 0
        terminal = ""
        for _ in range(max_steps):
            a = self.agent.act(obs, greedy=not train)
            res = self.env.step(a)
            if train:
                self.agent.learn(obs, a, res.reward, res.obs, res.done)
            total_reward += res.reward
            if res.info.get("got_food"):
                foods += 1
            obs = res.obs
            if res.done:
                terminal = res.info.get("terminal", "")
                break
        return self.env.steps, foods, float(total_reward), terminal

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

    def _layout(self, app) -> None:
        w, h = app.screen.get_size()
        scale = float(app.theme.ui_scale)
        pad = int(50 * scale)
        gap = int(12 * scale)
        row_h = int(54 * scale)
        left_w = int(w * 0.52)
        x0 = pad
        y0 = int(110 * scale)

        def rect(i: int) -> pygame.Rect:
            return pygame.Rect(x0, y0 + i * (row_h + gap), left_w - pad, row_h)

        cfg = app.cfg
        items: list = []
        i = 0

        items.append(Label(rect(i), "Training")); i += 1
        items.append(Slider(rect(i), "Episodes", 50, 10000, 50,
                            lambda: float(cfg.train_episodes), lambda v: setattr(cfg, "train_episodes", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Max steps/ep", 50, 800, 10,
                            lambda: float(cfg.train_max_steps), lambda v: setattr(cfg, "train_max_steps", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Eval every", 0, 1000, 50,
                            lambda: float(cfg.train_eval_every), lambda v: setattr(cfg, "train_eval_every", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Eval episodes", 10, 200, 10,
                            lambda: float(cfg.train_eval_episodes), lambda v: setattr(cfg, "train_eval_episodes", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Speed (eps/update)", 1, 50, 1,
                            lambda: float(cfg.train_speed), lambda v: setattr(cfg, "train_speed", int(v)), fmt="{:.0f}")); i += 1
        items.append(Toggle(rect(i), "Autosave on finish",
                            lambda: bool(cfg.train_autosave), lambda b: setattr(cfg, "train_autosave", bool(b)))); i += 1

        items.append(Label(rect(i), "Actions")); i += 1
        items.append(Button(rect(i), "Start / Resume", lambda: self._start(app))); i += 1
        items.append(Button(rect(i), "Pause", lambda: self._pause())); i += 1
        items.append(Button(rect(i), "Reset progress", lambda: self._reset(app))); i += 1
        items.append(Button(rect(i), "Save Q-table", lambda: self._open_save(app))); i += 1
        items.append(Button(rect(i), "Load Q-table", lambda: self._open_load(app))); i += 1
        items.append(Button(rect(i), "Play in simulation", lambda: self._play(app))); i += 1
        items.append(Button(rect(i), "Back", lambda: app.pop())); i += 1

        self.widgets = items
        self.focus.set(self.widgets)

    def _start(self, app) -> None:
        if self.env is None or self.agent is None:
            self._build_env_agent(app)
        self.training = True

    def _pause(self) -> None:
        self.training = False

    def _reset(self, app) -> None:
        self.training = False
        self.episodes_done = 0
        self.agent = None
        self.env = None
        self._avg_rewards.clear()
        self._avg_steps.clear()
        self._avg_foods.clear()
        self.last_eval = ""
        self._build_env_agent(app)

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
        app.push(SimulationScene())

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
        for _ in range(int(cfg.train_speed)):
            if self.episodes_done >= int(cfg.train_episodes):
                self.training = False
                if cfg.train_autosave and self.agent:
                    path = getattr(cfg, "qtable_path", os.path.join("data", "qtable_saved.pkl"))
                    self.agent.save(path)
                    app.toast.push(f"Autosaved Q-table -> {path}")
                break
            steps, foods, total, _ = self._run_episode(train=True, max_steps=int(cfg.train_max_steps))
            self._avg_steps.append(steps)
            self._avg_foods.append(foods)
            self._avg_rewards.append(total)
            self.episodes_done += 1
            if int(cfg.train_eval_every) > 0 and self.episodes_done % int(cfg.train_eval_every) == 0:
                self.last_eval = self._eval_stats(int(cfg.train_eval_episodes), int(cfg.train_max_steps))

    def render(self, app, screen: pygame.Surface) -> None:
        if not self.widgets:
            self._layout(app)
        screen.fill(app.theme.palette.bg)
        title_font = app.theme.font(int(app.theme.font_size_title * 0.7 * app.theme.ui_scale))
        font = app.theme.font(int(app.theme.font_size * app.theme.ui_scale))
        small = app.theme.font(int(app.theme.font_size * 0.85 * app.theme.ui_scale))

        title = title_font.render("Training / Learning", True, app.theme.palette.fg)
        screen.blit(title, (int(50 * app.theme.ui_scale), int(40 * app.theme.ui_scale)))

        focused = self.focus.focused()
        for w in self.widgets:
            w.draw(screen, app.theme, focused=(w is focused))

        # Right-side stats panel
        pad = int(50 * app.theme.ui_scale)
        stats_x = int(screen.get_width() * 0.56)
        stats_y = int(110 * app.theme.ui_scale)
        stats_w = screen.get_width() - stats_x - pad
        stats_h = screen.get_height() - stats_y - pad
        panel = pygame.Rect(stats_x, stats_y, stats_w, stats_h)
        app.theme.draw_rounded_panel(screen, panel, color=app.theme.palette.panel, border_radius=int(12 * app.theme.ui_scale))

        y = panel.y + int(16 * app.theme.ui_scale)
        lines = [
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

        for line in lines:
            surf = font.render(line, True, app.theme.palette.fg)
            screen.blit(surf, (panel.x + int(16 * app.theme.ui_scale), y))
            y += int(28 * app.theme.ui_scale)

        if self.last_eval:
            y += int(10 * app.theme.ui_scale)
            eval_title = font.render("Last eval:", True, app.theme.palette.muted)
            screen.blit(eval_title, (panel.x + int(16 * app.theme.ui_scale), y))
            y += int(26 * app.theme.ui_scale)
            for part in self.last_eval.split():
                s = small.render(part, True, app.theme.palette.fg)
                screen.blit(s, (panel.x + int(16 * app.theme.ui_scale), y))
                y += int(22 * app.theme.ui_scale)
