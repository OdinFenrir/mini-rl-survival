from __future__ import annotations

import os
import time

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


class SimulationScene:
    def __init__(self) -> None:
        self.env: GridSurvivalEnv | None = None
        self.agent: QLearningAgent | None = None
        self.obs = None

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
        self.agent = QLearningAgent(n_actions=4, cfg=qcfg, seed=int(cfg.seed))
        self.obs = self.env.reset()

        # Auto-load a saved policy if available so "Start Simulation" feels purposeful.
        self.loaded_qtable_path = None
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
        actions += [
            ToolbarAction("Session", "", kind="header"),
            ToolbarAction(f"Q-table: {qname}", "", kind="text"),
            ToolbarAction(f"Seed {cfg.seed}  Grid {cfg.w}x{cfg.h}  Haz {cfg.hazards}", "", kind="text"),
        ]

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

        steps = int(app.cfg.sim_steps_per_frame)
        if self.paused and not self.step_once:
            return
        if self.step_once:
            steps = 1
        self.step_once = False

        for _ in range(steps):
            a = self.agent.act(self.obs, greedy=self.greedy)
            res = self.env.step(a)

            self.last_action = a
            self.last_reward = float(res.reward)
            self.last_done = bool(res.done)
            self.last_info = dict(res.info)

            ep = self.stats.last()
            ep.steps = int(self.env.steps)
            ep.total_reward += float(res.reward)
            if res.info.get('got_food'):
                ep.foods += 1

            if res.done:
                ep.terminal = str(res.info.get('terminal', ''))
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
            else:
                self.obs = res.obs

    def _rc(self, app, board_rect: pygame.Rect) -> RenderContext:
        w, h = self.env.width, self.env.height
        cell = max(10, min(board_rect.w // w, board_rect.h // h))
        mx = board_rect.x + max(0, (board_rect.w - cell * w) // 2)
        my = board_rect.y + max(0, (board_rect.h - cell * h) // 2)
        return RenderContext(w=w, h=h, cell=cell, mx=mx, my=my, hud_h=0)

    def _heatmap(self) -> np.ndarray:
        key = (len(self.agent.Q), self.env.width, self.env.height)
        if key == self._heat_cache_key and self._heat_cache is not None:
            return self._heat_cache
        heat = np.zeros((self.env.height, self.env.width), dtype=float)
        cnt = np.zeros_like(heat, dtype=int)
        for s, qvals in self.agent.Q.items():
            try:
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

    def _render_sidebar(self, app, screen, sidebar_width: int) -> None:
        theme = app.theme
        pad = int(12 * theme.ui_scale)
        sidebar_rect = pygame.Rect(0, 0, sidebar_width, screen.get_height())
        sidebar_color = theme.palette.panel if theme.ui_style == "pixel" else theme.palette.grid1
        pygame.draw.rect(screen, sidebar_color, sidebar_rect)

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
        self._render_sidebar(app, screen, sidebar_width)
        shadow = pygame.Surface((board_rect.w + 16, board_rect.h + 16), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 90), shadow.get_rect(), border_radius=int(18 * app.theme.ui_scale))
        screen.blit(shadow, (board_rect.x - 8, board_rect.y - 8))
        scale = float(app.theme.ui_scale)
        border_radius = int(16 * scale)
        pixel_style = app.theme.ui_style == "pixel"
        pygame.draw.rect(screen, app.theme.palette.grid1, board_rect, border_radius=border_radius if not pixel_style else 0)
        pygame.draw.rect(screen, app.theme.palette.grid_line, board_rect, width=2, border_radius=border_radius if not pixel_style else 0)
        rc = self._rc(app, board_rect)
        rc_local = self._rc(app, pygame.Rect(0, 0, board_rect.w, board_rect.h))

        hm = self._heatmap() if self.show_heatmap else None
        heatmap_opacity = float(getattr(app.cfg, "heatmap_opacity", 0.7))
        board_surface = pygame.Surface((board_rect.w, board_rect.h), pygame.SRCALPHA)

        floor_a = None if pixel_style else self._tile_surface(0, rc_local.cell)
        floor_b = None if pixel_style else (self._tile_surface(1, rc_local.cell) or floor_a)

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
        else:
            hazard_sprite = self._tile_surface(357, sprite_scale)
            food_sprite = self._tile_surface(371, sprite_scale)
            agent_sprite = self._tile_surface(168, sprite_scale)

        def blit_center(surface: pygame.Surface, sprite: pygame.Surface, center: tuple[int, int]) -> None:
            surface.blit(sprite, (center[0] - sprite.get_width() // 2, center[1] - sprite.get_height() // 2))

        for hx, hy in self.env.hazards:
            cx, cy = rc_local.cell_center(hx, hy)
            if not pixel_style:
                draw_glow(board_surface, (cx, cy), int(rc_local.cell * 0.45), app.theme.palette.danger, 110)
            if hazard_sprite:
                blit_center(board_surface, hazard_sprite, (cx, cy))
            else:
                # Draw hazard: triangle with exclamation
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
        cx, cy = rc_local.cell_center(fx, fy)
        # Draw food
        if not pixel_style:
            draw_glow(board_surface, (cx, cy), int(rc_local.cell * 0.55), app.theme.palette.ok, 120)
        if food_sprite:
            blit_center(board_surface, food_sprite, (cx, cy))
        else:
            pygame.draw.circle(board_surface, app.theme.palette.ok, (cx, cy), max(3, rc_local.cell // 2 - 10))
            leaf_x = cx + rc_local.cell // 6
            leaf_y = cy - rc_local.cell // 3
            pygame.draw.ellipse(board_surface, (60, 200, 60), (leaf_x, leaf_y, rc_local.cell // 6, rc_local.cell // 8))

        ax, ay = self.env.agent
        cx, cy = rc_local.cell_center(ax, ay)
        # Draw agent
        if not pixel_style:
            draw_glow(board_surface, (cx, cy), int(rc_local.cell * 0.55), app.theme.palette.accent, 110)
        if agent_sprite:
            blit_center(board_surface, agent_sprite, (cx, cy))
        else:
            pygame.draw.circle(board_surface, app.theme.palette.accent, (cx, cy), max(4, rc_local.cell // 2 - 6))
            pygame.draw.circle(board_surface, app.theme.palette.fg, (cx, cy), max(2, rc_local.cell // 2 - 12), 2)
            # Eyes
            eye_dx = rc_local.cell // 8
            eye_dy = rc_local.cell // 8
            pygame.draw.circle(board_surface, (30, 30, 60), (cx - eye_dx, cy - eye_dy), max(1, rc_local.cell // 12))
            pygame.draw.circle(board_surface, (30, 30, 60), (cx + eye_dx, cy - eye_dy), max(1, rc_local.cell // 12))

        screen.blit(board_surface, board_rect.topleft)

        if self.show_policy:
            self.policy_overlay.render(screen, app.theme, self.agent, rc)
        if self.show_qhover:
            self.qvalues_overlay.render(screen, app.theme, self.agent, rc, pygame.mouse.get_pos())

        ep = self.stats.last()
        hud = (
            f'Ep {ep.episode}  Steps {ep.steps}  Foods {ep.foods}  '
            f'Energy {self.obs[4]}  Qstates {len(self.agent.Q)}  '
            f'Eps {self.agent.epsilon():.3f}  Mode {'GREEDY' if self.greedy else 'EPS'}  '
            f"{'PAUSED' if self.paused else ''}"
        )
        self.stats_overlay.render(screen, app.theme, hud, area=hud_rect)

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
            self.telemetry_overlay.render(screen, app.theme, pos=(24, 64))

        if self.show_telemetry:
            pad = int(12 * app.theme.ui_scale)
            self.telemetry_overlay.render(screen, app.theme, pos=(board_rect.x + pad, board_rect.y + pad))
