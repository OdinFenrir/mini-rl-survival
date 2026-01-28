from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pygame

from core.env import GridSurvivalEnv
from core.qlearning import QLearningAgent, QLearningConfig
from viewers.io.export import export_csv, export_json, export_screenshot
from viewers.io.run_history import append_entry
from viewers.io.save_load import load_env_snapshot, load_qtable, save_env_snapshot, save_qtable
from viewers.overlays.action_toolbar import ActionToolbar, ToolbarAction
from viewers.overlays.debug import DebugOverlay
from viewers.overlays.help_overlay import HelpOverlay
from viewers.overlays.policy import PolicyOverlay
from viewers.overlays.qvalues import QValuesOverlay
from viewers.overlays.run_history import RunHistoryOverlay
from viewers.overlays.stats import RunStats, StatsOverlay
from viewers.overlays.telemetry import TelemetryOverlay
from viewers.scenes.render_context import RenderContext
from viewers.ui.modals import FileDialogModal


def _normalize(hm: np.ndarray) -> np.ndarray:
    mx = float(hm.max()) if hm.size else 0.0
    mn = float(hm.min()) if hm.size else 0.0
    if mx <= mn:
        return np.zeros_like(hm, dtype=float)
    return (hm - mn) / (mx - mn)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: tuple[int, int, int]
    size: float
    gravity: float = 0.0


class SimulationScene:
    def __init__(self, agent_override: QLearningAgent | None = None, env_overrides: dict | None = None) -> None:
        self.env: GridSurvivalEnv | None = None
        self.agent: QLearningAgent | None = None
        self.obs = None
        self._agent_override = agent_override
        self._env_overrides = env_overrides or {}

        self.paused = False
        self.step_once = False
        self.greedy = True

        self.stats = RunStats()
        self.stats_overlay = StatsOverlay(self.stats)
        self.help_overlay = HelpOverlay()
        self.debug_overlay = DebugOverlay()
        self.policy_overlay = PolicyOverlay()
        self.qvalues_overlay = QValuesOverlay()
        self.telemetry_overlay = TelemetryOverlay()
        self.run_history_overlay = RunHistoryOverlay()

        self.toolbar_overlay: ActionToolbar | None = None
        self.loaded_qtable_path: str | None = None

        self.show_help = False
        self.show_debug = False
        self.show_policy = False
        self.show_heatmap = True
        self.show_qhover = True
        self.show_telemetry = False
        self.show_run_history = False

        self.last_action = None
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}

        self._heat_cache_key = None
        self._heat_cache = None
        self._tileset_surface = None
        self._tileset_meta = None
        self._tile_cache: dict[tuple[int, int], pygame.Surface] = {}
        self._pixel_sprites: dict[str, pygame.Surface] = {}
        self._pixel_sprite_cache: dict[tuple[str, int], pygame.Surface] = {}
        self._particles: list[Particle] = []
        self._time = 0.0
        self._agent_start = (0.0, 0.0)
        self._agent_target = (0.0, 0.0)
        self._agent_t = 1.0
        self._agent_speed = 8.0
        self._agent_render = (0.0, 0.0)
        self._shake_time = 0.0
        self._shake_mag = 0.0
        self._food_bob_seed = random.random() * 10.0

    def _load_tileset(self) -> None:
        if self._tileset_surface is not None or self._tileset_meta is False:
            return
        try:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "1bitpack_kenney_1.1"))
            tileset_path = os.path.join(base, "Tilemap", "tileset_legacy.png")
            if not os.path.exists(tileset_path):
                self._tileset_meta = False
                return
            self._tileset_surface = pygame.image.load(tileset_path).convert_alpha()
            # tileset_colored.tsx defines these values for tileset_legacy.png
            self._tileset_meta = {
                "tile": 16,
                "spacing": 1,
                "columns": 32,
            }
        except Exception:
            self._tileset_surface = None
            self._tileset_meta = False

    def _set_agent_anim(self, start: tuple[int, int], target: tuple[int, int]) -> None:
        self._agent_start = (float(start[0]), float(start[1]))
        self._agent_target = (float(target[0]), float(target[1]))
        self._agent_t = 0.0

    def _agent_pos(self, dt: float) -> tuple[float, float]:
        if self._agent_t < 1.0:
            self._agent_t = min(1.0, self._agent_t + dt * self._agent_speed)
        t = self._agent_t
        # smoothstep easing
        t = t * t * (3.0 - 2.0 * t)
        ax = self._agent_start[0] + (self._agent_target[0] - self._agent_start[0]) * t
        ay = self._agent_start[1] + (self._agent_target[1] - self._agent_start[1]) * t
        return (ax, ay)

    def _spawn_particles(self, pos: tuple[float, float], color: tuple[int, int, int], count: int, spread: float, size: float) -> None:
        for _ in range(count):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(20.0, 80.0) * spread
            vx = math.cos(ang) * spd
            vy = math.sin(ang) * spd
            self._particles.append(Particle(pos[0], pos[1], vx, vy, life=random.uniform(0.35, 0.9), color=color, size=size, gravity=40.0))

    def _update_particles(self, dt: float) -> None:
        if not self._particles:
            return
        alive: list[Particle] = []
        for p in self._particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.vy += p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            alive.append(p)
        self._particles = alive

    def _shake_offset(self) -> tuple[int, int]:
        if self._shake_time <= 0.0:
            return (0, 0)
        t = self._time * 60.0
        dx = int(math.sin(t * 1.7) * self._shake_mag)
        dy = int(math.cos(t * 1.3) * self._shake_mag)
        return (dx, dy)

    def _tile_surface(self, index: int, cell: int) -> pygame.Surface | None:
        if cell <= 0:
            return None
        self._load_tileset()
        if not self._tileset_surface or not self._tileset_meta:
            return None
        key = (index, cell)
        if key in self._tile_cache:
            return self._tile_cache[key]
        tile = int(self._tileset_meta["tile"])
        spacing = int(self._tileset_meta["spacing"])
        columns = int(self._tileset_meta["columns"])
        ix = index % columns
        iy = index // columns
        sx = ix * (tile + spacing)
        sy = iy * (tile + spacing)
        rect = pygame.Rect(sx, sy, tile, tile)
        if rect.right > self._tileset_surface.get_width() or rect.bottom > self._tileset_surface.get_height():
            return None
        sub = self._tileset_surface.subsurface(rect)
        scaled = pygame.transform.smoothscale(sub, (cell, cell))
        self._tile_cache[key] = scaled
        return scaled

    def _load_pixel_sprites(self) -> None:
        if self._pixel_sprites:
            return
        try:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
            sheet_path = os.path.join(base, "food_pixel_art", "Food Pixel Art", "Food Pixel Art.png")
            if os.path.exists(sheet_path):
                sheet = pygame.image.load(sheet_path).convert_alpha()
                tw, th = 19, 17
                def crop(col: int, row: int) -> pygame.Surface:
                    rect = pygame.Rect(col * tw, row * th, tw, th)
                    return sheet.subsurface(rect)
                self._pixel_sprites["food"] = crop(2, 5)
                self._pixel_sprites["agent"] = crop(1, 4)
            cross_path = os.path.join(base, "kenney_game_icons", "PNG", "Black", "1x", "cross.png")
            if os.path.exists(cross_path):
                self._pixel_sprites["hazard"] = pygame.image.load(cross_path).convert_alpha()
        except Exception:
            self._pixel_sprites = {}

    def _pixel_sprite(self, name: str, size: int) -> pygame.Surface | None:
        if size <= 0:
            return None
        self._load_pixel_sprites()
        if name not in self._pixel_sprites:
            return None
        key = (name, size)
        if key in self._pixel_sprite_cache:
            return self._pixel_sprite_cache[key]
        base = self._pixel_sprites[name]
        scaled = pygame.transform.scale(base, (size, size))
        self._pixel_sprite_cache[key] = scaled
        return scaled

    def _build(self, app) -> None:
        cfg = app.cfg
        env_kwargs = {
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
        }
        if self._env_overrides:
            env_kwargs.update(self._env_overrides)
        self.env = GridSurvivalEnv(**env_kwargs)
        qcfg = QLearningConfig(
            alpha=float(cfg.alpha),
            gamma=float(cfg.gamma),
            eps_start=float(cfg.eps_start),
            eps_end=float(cfg.eps_end),
            eps_decay_steps=int(cfg.eps_decay),
        )
        self.agent = QLearningAgent(n_actions=4, cfg=qcfg, seed=int(cfg.seed))
        self.obs = self.env.reset()
        self._set_agent_anim(self.env.agent, self.env.agent)
        self._agent_render = (float(self.env.agent[0]), float(self.env.agent[1]))
        self._particles = []
        self._shake_time = 0.0
        self._shake_mag = 0.0

        # Auto-load a saved policy if available so "Start Simulation" feels purposeful.
        self.loaded_qtable_path = None
        if self._agent_override is not None:
            self.agent = self._agent_override
            self.loaded_qtable_path = "<in-memory>"
            app.toast.push("Using in-memory Q-table from training")
            self._heat_cache_key = None
            self._heat_cache = None
        else:
            qpath = getattr(cfg, "qtable_path", os.path.join("data", "qtable_saved.pkl"))
            if qpath and os.path.exists(qpath):
                try:
                    self.agent = load_qtable(qpath, seed=int(cfg.seed))
                    self.loaded_qtable_path = qpath
                    app.toast.push(f"Auto-loaded Q-table: {os.path.basename(qpath)}")
                    self._heat_cache_key = None
                    self._heat_cache = None
                except Exception as exc:
                    self.paused = True
                    app.toast.push(f"Auto-load failed: {exc}")
            else:
                # No trained policy yet; start paused instead of running an untrained agent.
                self.paused = True
                app.toast.push("No Q-table loaded. Use 'Load Q-table' to play a trained agent.")

        self.show_heatmap = bool(cfg.heatmap)
        self.show_policy = bool(cfg.policy)
        self.show_qhover = bool(cfg.qhover)
        self.show_debug = bool(cfg.debug)
        self.toolbar_overlay = ActionToolbar(lambda: self._toolbar_actions(app))

    def _toolbar_actions(self, app) -> list[ToolbarAction]:
        actions: list[ToolbarAction] = []

        cfg = app.cfg
        qname = os.path.basename(self.loaded_qtable_path) if self.loaded_qtable_path else "None"
        wall_count = len(self.env.walls) if self.env else 0
        actions += [
            ToolbarAction("Session", "", kind="header"),
            ToolbarAction(f"Q-table: {qname}", "", kind="text"),
            ToolbarAction(f"Seed {cfg.seed}  Grid {cfg.w}x{cfg.h}", "", kind="text"),
            ToolbarAction(f"Walls {wall_count}  Food {'ON' if getattr(cfg, 'food_enabled', True) else 'OFF'}", "", kind="text"),
        ]
        if self.env:
            exit_state = "Unlocked" if (not getattr(self.env, "food_enabled", True) or bool(getattr(self.env, "food_collected", False))) else "Locked"
            actions.append(ToolbarAction(f"Exit: {exit_state}", "", kind="text"))
            total_levels = GridSurvivalEnv.preset_level_count()
            if getattr(self.env, "level_mode", "preset") == "preset" and total_levels > 0:
                level_label = f"{self.env.level_id + 1}/{total_levels}"
            elif getattr(self.env, "level_mode", "preset") == "preset":
                level_label = str(self.env.level_id + 1)
            else:
                level_label = "RND"
            actions.append(ToolbarAction(f"Mode {self.env.level_mode}  Level {level_label}", "", kind="text"))
            actions.append(ToolbarAction(f"Level: {self.env.level_name or 'Random'}", "", kind="text"))
            if getattr(self.env, "level_style", ""):
                actions.append(ToolbarAction(f"Style: {self.env.level_style}", "", kind="text"))
            if getattr(self.env, "level_source", ""):
                actions.append(ToolbarAction(f"Source: {self.env.level_source}", "", kind="text"))

        actions += [
            ToolbarAction('Controls', '', kind='header'),
            ToolbarAction(f"Pause: {'ON' if self.paused else 'OFF'}", 'Space', lambda: self._toggle_pause(app)),
            ToolbarAction('Step once', '.', lambda: self._step_once(app)),
            ToolbarAction('Reset episode', 'R', lambda: self._reset_episode(app)),
            ToolbarAction(f"Policy: {'GREEDY' if self.greedy else 'EPS'}", 'M', lambda: self._toggle_policy_mode(app)),
        ]

        actions += [
            ToolbarAction('View', '', kind='header'),
            ToolbarAction(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}", 'H', lambda: self._toggle_heatmap(app)),
            ToolbarAction(f"Policy arrows: {'ON' if self.show_policy else 'OFF'}", 'P', lambda: self._toggle_policy_overlay(app)),
            ToolbarAction(f"Q hover: {'ON' if self.show_qhover else 'OFF'}", 'Q', lambda: self._toggle_qhover(app)),
            ToolbarAction(f"Debug: {'ON' if self.show_debug else 'OFF'}", 'D', lambda: self._toggle_debug(app)),
            ToolbarAction(f"Telemetry: {'ON' if self.show_telemetry else 'OFF'}", 'Ctrl+T', lambda: self._toggle_telemetry(app)),
            ToolbarAction(f"Run history: {'ON' if self.show_run_history else 'OFF'}", 'Ctrl+K', lambda: self._toggle_run_history(app)),
            ToolbarAction(f"Help: {'ON' if self.show_help else 'OFF'}", '?', lambda: self._toggle_help(app)),
        ]

        actions += [
            ToolbarAction('Files', '', kind='header'),
            ToolbarAction('Load Q-table', 'Ctrl+L', lambda: self._open_load_qtable_modal(app)),
            ToolbarAction('Save Q-table', 'Ctrl+S', lambda: self._open_save_qtable_modal(app)),
            ToolbarAction('Load Env Snapshot', 'Ctrl+I', lambda: self._open_load_env_modal(app)),
            ToolbarAction('Save Env Snapshot', 'Ctrl+O', lambda: self._open_save_env_modal(app)),
        ]

        actions += [
            ToolbarAction('Export', '', kind='header'),
            ToolbarAction('Screenshot', 'Ctrl+E', lambda: self._open_export_screenshot_modal(app)),
            ToolbarAction('Export stats', 'Ctrl+X', lambda: self._open_export_stats_modal(app)),
        ]

        if self.show_run_history:
            actions.append(ToolbarAction('Recent episodes', '', kind='header'))
            if not self.run_history_overlay.entries:
                actions.append(ToolbarAction('(no history yet)', '', kind='text'))
            else:
                for entry in self.run_history_overlay.entries[-6:]:
                    ep = entry.get('episode', '?')
                    r = entry.get('reward', 0.0)
                    st = entry.get('steps', 0)
                    term = entry.get('terminal', '')
                    actions.append(ToolbarAction(f"Ep {ep}  R {r:.1f}  S {st}  {term}", '', kind='text'))

        return actions

    def _toggle_pause(self, app) -> None:
        self.paused = not self.paused
        app.toast.push('Paused' if self.paused else 'Resumed')

    def _step_once(self, app) -> None:
        self.step_once = True

    def _reset_episode(self, app) -> None:
        self.obs = self.env.reset()
        self._set_agent_anim(self.env.agent, self.env.agent)
        self._heat_cache_key = None
        self._heat_cache = None
        app.toast.push('Episode reset')

    def _toggle_policy_mode(self, app) -> None:
        self.greedy = not self.greedy
        app.toast.push('Policy: ' + ('GREEDY' if self.greedy else 'EPSILON'))

    def _toggle_heatmap(self, app) -> None:
        self.show_heatmap = not self.show_heatmap

    def _toggle_policy_overlay(self, app) -> None:
        self.show_policy = not self.show_policy

    def _toggle_qhover(self, app) -> None:
        self.show_qhover = not self.show_qhover

    def _toggle_debug(self, app) -> None:
        self.show_debug = not self.show_debug

    def _toggle_help(self, app) -> None:
        self.show_help = not self.show_help

    def _toggle_telemetry(self, app) -> None:
        self.show_telemetry = not self.show_telemetry
        app.toast.push('Telemetry: ' + ('ON' if self.show_telemetry else 'OFF'))

    def _toggle_run_history(self, app) -> None:
        self.show_run_history = not self.show_run_history
        if self.show_run_history:
            self.run_history_overlay.reload()
        app.toast.push('Run history: ' + ('ON' if self.show_run_history else 'OFF'))

    def _open_save_qtable_modal(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center
        def on_confirm(path: str) -> None:
            try:
                save_qtable(self.agent, path)
                app.cfg.qtable_path = path
                self.loaded_qtable_path = path
                app.toast.push(f'Saved Q-table -> {path}')
            except Exception as exc:
                app.toast.push(f'Save failed: {exc}')
        modal = FileDialogModal(
            rect,
            'Save Q-table',
            on_confirm,
            lambda: None,
            initial_path=getattr(app.cfg, "qtable_path", os.path.join('data', 'qtable_saved.pkl')),
        )
        app.push_modal(modal)

    def _open_load_qtable_modal(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center
        def on_confirm(path: str) -> None:
            try:
                self.agent = load_qtable(path, seed=int(app.cfg.seed))
                app.cfg.qtable_path = path
                self.loaded_qtable_path = path
                app.toast.push(f'Loaded Q-table <- {path}')
                self._heat_cache_key = None
                self._heat_cache = None
            except Exception as exc:
                app.toast.push(f'Load failed: {exc}')
        modal = FileDialogModal(
            rect,
            'Load Q-table',
            on_confirm,
            lambda: None,
            initial_path=getattr(app.cfg, "qtable_path", os.path.join('data', 'qtable_saved.pkl')),
            must_exist=True,
        )
        app.push_modal(modal)

    def _open_save_env_modal(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center
        def on_confirm(path: str) -> None:
            try:
                save_env_snapshot(self.env, path)
                app.cfg.env_snapshot_path = path
                app.toast.push(f'Saved env snapshot -> {path}')
            except Exception as exc:
                app.toast.push(f'Snapshot save failed: {exc}')
        modal = FileDialogModal(
            rect,
            'Save Env Snapshot',
            on_confirm,
            lambda: None,
            initial_path=getattr(app.cfg, "env_snapshot_path", os.path.join('data', 'env_snapshot.json')),
        )
        app.push_modal(modal)

    def _open_load_env_modal(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center
        def on_confirm(path: str) -> None:
            try:
                self.env = load_env_snapshot(path)
                self.obs = self.env._obs()  # type: ignore[attr-defined]
                app.cfg.env_snapshot_path = path
                app.toast.push(f'Loaded env snapshot <- {path}')
            except Exception as exc:
                app.toast.push(f'Snapshot load failed: {exc}')
        modal = FileDialogModal(
            rect,
            'Load Env Snapshot',
            on_confirm,
            lambda: None,
            initial_path=getattr(app.cfg, "env_snapshot_path", os.path.join('data', 'env_snapshot.json')),
            must_exist=True,
        )
        app.push_modal(modal)

    def _open_export_screenshot_modal(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center
        def on_confirm(path: str) -> None:
            try:
                export_screenshot(app.screen, path)
                app.toast.push(f'Exported screenshot -> {path}')
            except Exception as exc:
                app.toast.push(f'Export failed: {exc}')
        modal = FileDialogModal(
            rect,
            'Export Screenshot',
            on_confirm,
            lambda: None,
            initial_path=os.path.join('data', f'screenshot_{int(time.time())}.png'),
        )
        app.push_modal(modal)

    def _open_export_stats_modal(self, app) -> None:
        rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
        rect.center = app.screen.get_rect().center
        def on_confirm(path: str) -> None:
            try:
                export_json(self.stats.to_rows(), path)
                export_csv(self.stats.to_rows(), os.path.splitext(path)[0] + '.csv')
                app.toast.push(f'Exported stats -> {path}')
            except Exception as exc:
                app.toast.push(f'Export failed: {exc}')
        modal = FileDialogModal(
            rect,
            'Export Stats',
            on_confirm,
            lambda: None,
            initial_path=os.path.join('data', 'run_stats.json'),
        )
        app.push_modal(modal)

    def handle_event(self, app, event: pygame.event.Event) -> None:
        if self.env is None or self.agent is None:
            self._build(app)

        if self.toolbar_overlay and self.toolbar_overlay.handle_event(event):
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                app.pop()
                return
            if event.key == pygame.K_SPACE:
                self._toggle_pause(app)
            elif event.key == pygame.K_PERIOD:
                self._step_once(app)
            elif event.key == pygame.K_r:
                self._reset_episode(app)
            elif event.key == pygame.K_m:
                self._toggle_policy_mode(app)
            elif event.key == pygame.K_h:
                self._toggle_heatmap(app)
            elif event.key == pygame.K_p:
                self._toggle_policy_overlay(app)
            elif event.key == pygame.K_q:
                self._toggle_qhover(app)
            elif event.key == pygame.K_d:
                self._toggle_debug(app)
            elif event.key in (pygame.K_SLASH, pygame.K_QUESTION):
                self._toggle_help(app)
            elif event.key == pygame.K_s and (event.mod & pygame.KMOD_CTRL):
                self._open_save_qtable_modal(app)
            elif event.key == pygame.K_l and (event.mod & pygame.KMOD_CTRL):
                self._open_load_qtable_modal(app)
            elif event.key == pygame.K_e and (event.mod & pygame.KMOD_CTRL):
                self._open_export_screenshot_modal(app)
            elif event.key == pygame.K_o and (event.mod & pygame.KMOD_CTRL):
                self._open_save_env_modal(app)
            elif event.key == pygame.K_i and (event.mod & pygame.KMOD_CTRL):
                self._open_load_env_modal(app)
            elif event.key == pygame.K_x and (event.mod & pygame.KMOD_CTRL):
                self._open_export_stats_modal(app)
            elif event.key == pygame.K_t and (event.mod & pygame.KMOD_CTRL):
                self._toggle_telemetry(app)
            elif event.key == pygame.K_k and (event.mod & pygame.KMOD_CTRL):
                self._toggle_run_history(app)

    def update(self, app, dt: float) -> None:
        if self.env is None or self.agent is None:
            self._build(app)

        self._time += dt
        self._update_particles(dt)
        if self._shake_time > 0.0:
            self._shake_time = max(0.0, self._shake_time - dt)
        self._agent_render = self._agent_pos(dt)

        steps = int(app.cfg.sim_steps_per_frame)
        if self.paused and not self.step_once:
            return
        if self.step_once:
            steps = 1
        self.step_once = False

        for _ in range(steps):
            prev_agent = self.env.agent
            a = self.agent.act(self.obs, greedy=self.greedy)
            res = self.env.step(a)

            self._set_agent_anim(prev_agent, self.env.agent)
            self.last_action = a
            self.last_reward = float(res.reward)
            self.last_done = bool(res.done)
            self.last_info = dict(res.info)

            ep = self.stats.last()
            ep.steps = int(self.env.steps)
            ep.total_reward += float(res.reward)
            if res.info.get('got_food'):
                ep.foods += 1
                cx, cy = self.env.agent
                self._spawn_particles((float(cx) + 0.5, float(cy) + 0.5), app.theme.palette.ok, count=12, spread=1.0, size=2.0)
                if hasattr(app, "sfx"):
                    app.sfx.play("food")
                app.toast.push("Food collected! Exit unlocked.")

            if res.done:
                ep.terminal = str(res.info.get('terminal', ''))
                if ep.terminal == "trap":
                    self._shake_time = 0.25
                    self._shake_mag = 4.0
                    cx, cy = self.env.agent
                    self._spawn_particles((float(cx) + 0.5, float(cy) + 0.5), app.theme.palette.danger, count=18, spread=1.2, size=2.4)
                    if hasattr(app, "sfx"):
                        app.sfx.play("hazard")
                elif ep.terminal == "goal":
                    cx, cy = self.env.agent
                    self._spawn_particles((float(cx) + 0.5, float(cy) + 0.5), app.theme.palette.warn, count=14, spread=1.0, size=2.2)
                    if hasattr(app, "sfx"):
                        app.sfx.play("confirm")
                self.telemetry_overlay.log_episode(ep.total_reward, ep.steps, ep.terminal)
                self.stats.new_episode()
                append_entry(
                    episode=ep.episode,
                    steps=ep.steps,
                    reward=ep.total_reward,
                    terminal=ep.terminal,
                    cfg=app.cfg,
                    timestamp=time.time(),
                )
                self.run_history_overlay.reload()
                self.obs = self.env.reset()
                self._set_agent_anim(self.env.agent, self.env.agent)
            else:
                self.obs = res.obs

    def _rc(self, app, board_rect: pygame.Rect) -> RenderContext:
        w, h = self.env.width, self.env.height
        cell = max(10, min(board_rect.w // w, board_rect.h // h))
        mx = board_rect.x + max(0, (board_rect.w - cell * w) // 2)
        my = board_rect.y + max(0, (board_rect.h - cell * h) // 2)
        return RenderContext(w=w, h=h, cell=cell, mx=mx, my=my, hud_h=0)

    def _heatmap(self) -> np.ndarray:
        key = (len(self.agent.Q), self.env.width, self.env.height, self.env.level_id)
        if key == self._heat_cache_key and self._heat_cache is not None:
            return self._heat_cache
        heat = np.zeros((self.env.height, self.env.width), dtype=float)
        cnt = np.zeros_like(heat, dtype=int)
        for s, qvals in self.agent.Q.items():
            try:
                if len(s) >= 8:
                    lvl = int(s[0])
                    ax, ay = int(s[1]), int(s[2])
                    if lvl != self.env.level_id:
                        continue
                else:
                    ax, ay = int(s[0]), int(s[1])
            except Exception:
                continue
            if 0 <= ax < self.env.width and 0 <= ay < self.env.height and qvals:
                heat[ay, ax] += float(max(qvals))
                cnt[ay, ax] += 1
        out = np.zeros_like(heat, dtype=float)
        m = cnt > 0
        out[m] = heat[m] / cnt[m]
        out = _normalize(out)
        self._heat_cache_key = key
        self._heat_cache = out
        return out

    def _sidebar_width(self, app, screen) -> int:
        sw = screen.get_width()
        scale = float(app.theme.ui_scale)
        min_w = int(260 * scale)
        ideal = int(520 * scale)
        max_w = int(sw * 0.52)
        return max(min_w, min(ideal, max_w))

    def _layout_rects(self, app, screen) -> tuple[pygame.Rect, pygame.Rect, int]:
        pad = int(12 * app.theme.ui_scale)
        sw, sh = screen.get_size()
        sidebar_width = self._sidebar_width(app, screen)

        hud_h = int(72 * app.theme.ui_scale)
        board_w = max(int(64 * app.theme.ui_scale), sw - sidebar_width - 3 * pad)
        board_h = max(int(64 * app.theme.ui_scale), sh - 3 * pad - hud_h)
        x = sidebar_width + pad
        y = pad
        board_rect = pygame.Rect(x, y, board_w, board_h)
        hud_rect = pygame.Rect(x, y + board_h + pad, board_w, hud_h)
        return board_rect, hud_rect, sidebar_width

    def _draw_hud_cards(self, screen: pygame.Surface, theme, area: pygame.Rect) -> None:
        pad = int(10 * theme.ui_scale)
        gap = int(10 * theme.ui_scale)
        card_h = area.h
        card_w = max(120, (area.w - 3 * gap) // 4)
        x = area.x
        y = area.y
        font = theme.font(int(theme.font_size * theme.ui_scale))
        small = theme.font(int(theme.font_size * 0.8 * theme.ui_scale))

        def card(rect: pygame.Rect, title: str, value: str, bar: float | None = None) -> None:
            theme.draw_rounded_panel(screen, rect, color=theme.palette.panel, border_radius=int(10 * theme.ui_scale))
            t = small.render(title, True, theme.palette.muted)
            v = font.render(value, True, theme.palette.fg)
            screen.blit(t, (rect.x + pad, rect.y + pad))
            screen.blit(v, (rect.x + pad, rect.y + pad + t.get_height() + 4))
            if bar is not None:
                bar = max(0.0, min(1.0, bar))
                bw = rect.w - 2 * pad
                bh = max(6, int(8 * theme.ui_scale))
                by = rect.bottom - pad - bh
                pygame.draw.rect(screen, theme.palette.grid0, (rect.x + pad, by, bw, bh), border_radius=4)
                pygame.draw.rect(screen, theme.palette.accent, (rect.x + pad, by, int(bw * bar), bh), border_radius=4)

        ep = self.stats.last()
        energy = self.obs[-1] if self.obs else 0
        cfg = self.env
        if cfg and getattr(self.env, "energy_max", None) is not None:
            energy_max = int(self.env.energy_max)
        else:
            energy_max = int(cfg.energy_start + cfg.energy_food_gain) if cfg else max(1, energy)
            energy_max = max(energy_max, int(energy))
        epsilon = self.agent.epsilon() if self.agent else 0.0

        r1 = pygame.Rect(x + 0 * (card_w + gap), y, card_w, card_h)
        r2 = pygame.Rect(x + 1 * (card_w + gap), y, card_w, card_h)
        r3 = pygame.Rect(x + 2 * (card_w + gap), y, card_w, card_h)
        r4 = pygame.Rect(x + 3 * (card_w + gap), y, card_w, card_h)
        level_text = "RND"
        if getattr(self.env, "level_mode", "preset") == "preset":
            total_levels = GridSurvivalEnv.preset_level_count()
            if total_levels > 0:
                level_text = f"{self.env.level_id + 1}/{total_levels}"
            else:
                level_text = str(self.env.level_id + 1)
        card(r1, "Level", level_text)
        card(r2, "Steps", str(ep.steps))
        card(r3, "Energy", f"{energy}", bar=energy / max(1, energy_max))
        card(r4, "Epsilon", f"{epsilon:.2f}", bar=epsilon)

    def _render_sidebar(self, app, screen, sidebar_width: int) -> None:
        theme = app.theme
        pad = int(12 * theme.ui_scale)
        sidebar_rect = pygame.Rect(0, 0, sidebar_width, screen.get_height())
        if theme.ui_style == "pixel":
            sidebar_color = theme.palette.panel
            pygame.draw.rect(screen, sidebar_color, sidebar_rect)
        else:
            theme.draw_gradient_panel(screen, sidebar_rect, theme.palette.panel, theme.palette.grid1, border_radius=0)

        inner_x = pad
        inner_y = pad
        inner_width = max(0, sidebar_width - 2 * pad)

        if self.toolbar_overlay:
            inner_height = max(1, screen.get_height() - 2 * pad)
            self.toolbar_overlay.render(screen, theme, x=inner_x, y=inner_y, width=inner_width, height=inner_height)

    def render(self, app, screen: pygame.Surface) -> None:
        if self.env is None or self.agent is None:
            self._build(app)
        screen.fill(app.theme.palette.bg)
        board_rect, hud_rect, sidebar_width = self._layout_rects(app, screen)
        dx, dy = self._shake_offset()
        board_rect = board_rect.move(dx, dy)
        hud_rect = hud_rect.move(dx, dy)
        self._render_sidebar(app, screen, sidebar_width)
        shadow = pygame.Surface((board_rect.w + 16, board_rect.h + 16), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 90), shadow.get_rect(), border_radius=int(18 * app.theme.ui_scale))
        screen.blit(shadow, (board_rect.x - 8, board_rect.y - 8))
        scale = float(app.theme.ui_scale)
        border_radius = int(16 * scale)
        pixel_style = app.theme.ui_style == "pixel"
        use_tiles = not pixel_style and app.theme.ui_style not in ("neo", "modern")
        if pixel_style:
            pygame.draw.rect(screen, app.theme.palette.grid1, board_rect, border_radius=0)
            pygame.draw.rect(screen, app.theme.palette.grid_line, board_rect, width=2, border_radius=0)
        else:
            app.theme.draw_gradient_panel(
                screen,
                board_rect,
                app.theme.palette.grid1,
                app.theme.palette.grid0,
                border_radius=border_radius,
            )
        rc = self._rc(app, board_rect)
        rc_local = self._rc(app, pygame.Rect(0, 0, board_rect.w, board_rect.h))

        hm = self._heatmap() if self.show_heatmap else None
        heatmap_opacity = float(getattr(app.cfg, "heatmap_opacity", 0.7))
        board_surface = pygame.Surface((board_rect.w, board_rect.h), pygame.SRCALPHA)

        floor_a = None if not use_tiles else self._tile_surface(0, rc_local.cell)
        floor_b = None if not use_tiles else (self._tile_surface(1, rc_local.cell) or floor_a)

        wall_set = set(getattr(self.env, "walls", []))
        for y in range(self.env.height):
            for x in range(self.env.width):
                r = rc_local.cell_rect(x, y)
                if floor_a:
                    tile = floor_a if (x + y) % 2 == 0 else floor_b
                    board_surface.blit(tile, r.topleft)
                else:
                    base = app.theme.palette.grid0 if (x + y) % 2 == 0 else app.theme.palette.grid1
                    pygame.draw.rect(board_surface, base, r, border_radius=0 if pixel_style else max(1, rc_local.cell // 8))
                if hm is not None:
                    if (x, y) in wall_set:
                        continue
                    v = float(hm[y, x])
                    if v > 0:
                        shade = int(40 + 180 * v)
                        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                        alpha = int(255 * heatmap_opacity)
                        if pixel_style:
                            alpha = int(alpha * 0.55)
                        s.fill((shade, shade, shade, alpha))
                        board_surface.blit(s, r.topleft)

        # Soft grid lines
        grid_overlay = pygame.Surface(board_surface.get_size(), pygame.SRCALPHA)
        line_alpha = 120 if pixel_style else 70
        line_color = (*app.theme.palette.grid_line, line_alpha)
        gx0, gy0 = rc_local.mx, rc_local.my
        gw = rc_local.w * rc_local.cell
        gh = rc_local.h * rc_local.cell
        for i in range(self.env.width + 1):
            px = gx0 + i * rc_local.cell
            pygame.draw.line(grid_overlay, line_color, (px, gy0), (px, gy0 + gh), 2 if pixel_style else 1)
        for j in range(self.env.height + 1):
            py = gy0 + j * rc_local.cell
            pygame.draw.line(grid_overlay, line_color, (gx0, py), (gx0 + gw, py), 2 if pixel_style else 1)
        board_surface.blit(grid_overlay, (0, 0))

        def draw_glow(surface: pygame.Surface, pos: tuple[int, int], radius: int, color: tuple[int, int, int], alpha: int) -> None:
            if radius <= 0:
                return
            glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*color, alpha), (radius, radius), radius)
            surface.blit(glow, (pos[0] - radius, pos[1] - radius))

        sprite_scale = max(8, int(rc_local.cell * (0.8 if pixel_style else 0.9)))
        if pixel_style:
            hazard_sprite = self._pixel_sprite("hazard", sprite_scale)
            food_sprite = self._pixel_sprite("food", sprite_scale)
            agent_sprite = self._pixel_sprite("agent", sprite_scale)
        elif use_tiles:
            hazard_sprite = self._tile_surface(357, sprite_scale)
            food_sprite = self._tile_surface(371, sprite_scale)
            agent_sprite = self._tile_surface(168, sprite_scale)
        else:
            hazard_sprite = None
            food_sprite = None
            agent_sprite = None

        def blit_center(surface: pygame.Surface, sprite: pygame.Surface, center: tuple[int, int]) -> None:
            surface.blit(sprite, (center[0] - sprite.get_width() // 2, center[1] - sprite.get_height() // 2))

        def cell_center_float(pos: tuple[float, float]) -> tuple[float, float]:
            return (
                rc_local.mx + (pos[0] + 0.5) * rc_local.cell,
                rc_local.my + (pos[1] + 0.5) * rc_local.cell,
            )

        # Walls
        wall_color = app.theme.palette.grid_line if pixel_style else app.theme.palette.grid1
        wall_inner = app.theme.palette.grid0
        for wx, wy in getattr(self.env, "walls", []):
            r = rc_local.cell_rect(wx, wy)
            pygame.draw.rect(board_surface, wall_color, r, border_radius=0 if pixel_style else max(2, rc_local.cell // 6))
            if not pixel_style:
                inset = max(1, rc_local.cell // 10)
                inner = pygame.Rect(r.x + inset, r.y + inset, r.w - 2 * inset, r.h - 2 * inset)
                pygame.draw.rect(board_surface, wall_inner, inner, border_radius=max(1, rc_local.cell // 8))

        # Goal (unlock after food is collected)
        gx, gy = self.env.goal
        gcx, gcy = rc_local.cell_center(gx, gy)
        goal_visible = (not getattr(self.env, "food_enabled", True)) or bool(getattr(self.env, "food_collected", False))
        if goal_visible:
            if pixel_style:
                gr = rc_local.cell_rect(gx, gy)
                pygame.draw.rect(board_surface, app.theme.palette.warn, gr, border_radius=0)
                pygame.draw.rect(board_surface, app.theme.palette.grid_line, gr, width=2)
            else:
                draw_glow(board_surface, (gcx, gcy), int(rc_local.cell * 0.5), app.theme.palette.warn, 120)
                pygame.draw.circle(board_surface, app.theme.palette.warn, (gcx, gcy), max(4, rc_local.cell // 3))
                pygame.draw.circle(board_surface, app.theme.palette.bg, (gcx, gcy), max(2, rc_local.cell // 6))
        else:
            lock_color = app.theme.palette.muted
            if pixel_style:
                gr = rc_local.cell_rect(gx, gy)
                pygame.draw.rect(board_surface, app.theme.palette.grid0, gr, border_radius=0)
                pygame.draw.rect(board_surface, app.theme.palette.grid_line, gr, width=2)
                pygame.draw.line(board_surface, lock_color, gr.topleft, gr.bottomright, 2)
                pygame.draw.line(board_surface, lock_color, gr.topright, gr.bottomleft, 2)
            else:
                r = rc_local.cell
                pygame.draw.circle(board_surface, (*lock_color, 90), (gcx, gcy), max(6, r // 3))
                body_w = max(8, r // 3)
                body_h = max(6, r // 4)
                body = pygame.Rect(gcx - body_w // 2, gcy - body_h // 2, body_w, body_h)
                pygame.draw.rect(board_surface, lock_color, body, border_radius=4)
                pygame.draw.circle(board_surface, lock_color, (gcx, gcy - body_h // 2), max(4, body_w // 2), 2)

        for hx, hy in self.env.hazards:
            cx, cy = rc_local.cell_center(hx, hy)
            if not pixel_style:
                draw_glow(board_surface, (cx, cy), int(rc_local.cell * 0.45), app.theme.palette.danger, 110)
            if hazard_sprite:
                blit_center(board_surface, hazard_sprite, (cx, cy))
            else:
                # Draw trap: triangle with exclamation
                points = [
                    (cx, cy - rc_local.cell // 3),
                    (cx - rc_local.cell // 3, cy + rc_local.cell // 3),
                    (cx + rc_local.cell // 3, cy + rc_local.cell // 3)
                ]
                shadow = [(px + 2, py + 2) for px, py in points]
                pygame.draw.polygon(board_surface, (0, 0, 0, 120), shadow)
                pygame.draw.polygon(board_surface, app.theme.palette.danger, points)
                font = app.theme.font(int(rc_local.cell * 0.5))
                ex = font.render('!', True, app.theme.palette.fg)
                board_surface.blit(ex, (cx - ex.get_width() // 2, cy - ex.get_height() // 2))

        fx, fy = self.env.food
        if fx >= 0 and fy >= 0:
            cx, cy = rc_local.cell_center(fx, fy)
            # Draw food
            if not pixel_style:
                draw_glow(board_surface, (cx, cy), int(rc_local.cell * 0.55), app.theme.palette.ok, 120)
            if food_sprite:
                bob = math.sin(self._time * 3.2 + self._food_bob_seed) * (rc_local.cell * 0.06)
                blit_center(board_surface, food_sprite, (cx, int(cy + bob)))
            else:
                pygame.draw.circle(board_surface, app.theme.palette.ok, (cx, cy), max(3, rc_local.cell // 2 - 10))
                leaf_x = cx + rc_local.cell // 6
                leaf_y = cy - rc_local.cell // 3
                pygame.draw.ellipse(board_surface, (60, 200, 60), (leaf_x, leaf_y, rc_local.cell // 6, rc_local.cell // 8))

        ax, ay = self.env.agent
        axf, ayf = self._agent_render
        cx, cy = cell_center_float((axf, ayf))
        # Draw agent
        if not pixel_style:
            draw_glow(board_surface, (cx, cy), int(rc_local.cell * 0.55), app.theme.palette.accent, 110)
        if agent_sprite:
            bob = math.sin(self._time * 4.0) * (rc_local.cell * 0.04)
            blit_center(board_surface, agent_sprite, (int(cx), int(cy + bob)))
        else:
            pygame.draw.circle(board_surface, app.theme.palette.accent, (cx, cy), max(4, rc_local.cell // 2 - 6))
            pygame.draw.circle(board_surface, app.theme.palette.fg, (cx, cy), max(2, rc_local.cell // 2 - 12), 2)
            # Eyes
            eye_dx = rc_local.cell // 8
            eye_dy = rc_local.cell // 8
            pygame.draw.circle(board_surface, (30, 30, 60), (cx - eye_dx, cy - eye_dy), max(1, rc_local.cell // 12))
            pygame.draw.circle(board_surface, (30, 30, 60), (cx + eye_dx, cy - eye_dy), max(1, rc_local.cell // 12))

        # Particles
        for p in self._particles:
            px = int(rc_local.mx + p.x * rc_local.cell)
            py = int(rc_local.my + p.y * rc_local.cell)
            pygame.draw.circle(board_surface, (*p.color, 200), (px, py), max(1, int(p.size)))

        screen.blit(board_surface, board_rect.topleft)

        level_label = getattr(self.env, "level_name", "")
        level_desc = getattr(self.env, "level_desc", "")
        if level_label:
            info_font = app.theme.font(int(app.theme.font_size * 0.78 * app.theme.ui_scale))
            info_pad = int(6 * app.theme.ui_scale)
            label_surf = info_font.render(level_label, True, app.theme.palette.accent)
            info_y = max(0, board_rect.y - label_surf.get_height() - info_pad)
            screen.blit(label_surf, (board_rect.x, info_y))
            if level_desc:
                desc_surf = app.theme.font(int(app.theme.font_size * 0.7 * app.theme.ui_scale)).render(level_desc, True, app.theme.palette.muted)
                screen.blit(desc_surf, (board_rect.x, info_y + label_surf.get_height() + info_pad // 2))
        if self.show_policy:
            self.policy_overlay.render(screen, app.theme, self.agent, rc, level_id=self.env.level_id, blocked=wall_set)
        if self.show_qhover:
            self.qvalues_overlay.render(screen, app.theme, self.agent, rc, pygame.mouse.get_pos(), level_id=self.env.level_id)

        ep = self.stats.last()
        self._draw_hud_cards(screen, app.theme, hud_rect)

        if self.show_debug:
            self.debug_overlay.render(
                screen,
                app.theme,
                self.obs,
                self.last_action,
                self.last_reward,
                self.last_done,
                self.last_info,
            )

        if self.show_help:
            self.help_overlay.render(screen, app.theme)

        if self.show_telemetry:
            pad = int(12 * app.theme.ui_scale)
            self.telemetry_overlay.render(screen, app.theme, pos=(board_rect.x + pad, board_rect.y + pad))
