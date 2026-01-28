from __future__ import annotations

import os
import time

import numpy as np
import pygame

from core.env import GridSurvivalEnv
from core.qlearning import QLearningAgent, QLearningConfig
from viewers.io.export import export_csv, export_json, export_screenshot
from viewers.io.save_load import load_env_snapshot, load_qtable, save_env_snapshot, save_qtable
from viewers.overlays.debug import DebugOverlay
from viewers.overlays.help_overlay import HelpOverlay
from viewers.overlays.policy import PolicyOverlay
from viewers.overlays.qvalues import QValuesOverlay
from viewers.overlays.stats import RunStats, StatsOverlay
from viewers.scenes.render_context import RenderContext


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

        self.show_help = False
        self.show_debug = False
        self.show_policy = False
        self.show_heatmap = True
        self.show_qhover = True

        self.last_action = None
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}

        self._heat_cache_key = None
        self._heat_cache = None

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

        self.show_heatmap = bool(cfg.heatmap)
        self.show_policy = bool(cfg.policy)
        self.show_qhover = bool(cfg.qhover)
        self.show_debug = bool(cfg.debug)

    def handle_event(self, app, event: pygame.event.Event) -> None:
        if self.env is None or self.agent is None:
            self._build(app)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                app.pop()
                return
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_PERIOD:
                self.step_once = True
            elif event.key == pygame.K_r:
                self.obs = self.env.reset()
                self._heat_cache_key = None
                self._heat_cache = None
                app.toast.push('Episode reset')
            elif event.key == pygame.K_m:
                self.greedy = not self.greedy
                app.toast.push('Policy: ' + ('GREEDY' if self.greedy else 'EPSILON'))
            elif event.key == pygame.K_h:
                self.show_heatmap = not self.show_heatmap
            elif event.key == pygame.K_p:
                self.show_policy = not self.show_policy
            elif event.key == pygame.K_q:
                self.show_qhover = not self.show_qhover
            elif event.key == pygame.K_d:
                self.show_debug = not self.show_debug
            elif event.key in (pygame.K_SLASH, pygame.K_QUESTION):
                self.show_help = not self.show_help
            elif event.key == pygame.K_s and (event.mod & pygame.KMOD_CTRL):
                path = os.path.join('data', 'qtable_saved.pkl')
                try:
                    save_qtable(self.agent, path)
                    app.toast.push(f'Saved Q-table -> {path}')
                except Exception as e:
                    app.toast.push(f'Save failed: {e}')
            elif event.key == pygame.K_l and (event.mod & pygame.KMOD_CTRL):
                path = os.path.join('data', 'qtable_saved.pkl')
                try:
                    self.agent = load_qtable(path, seed=int(app.cfg.seed))
                    app.toast.push(f'Loaded Q-table <- {path}')
                    self._heat_cache_key = None
                    self._heat_cache = None
                except Exception as e:
                    app.toast.push(f'Load failed: {e}')
            elif event.key == pygame.K_e and (event.mod & pygame.KMOD_CTRL):
                path = os.path.join('data', f'screenshot_{int(time.time())}.png')
                try:
                    export_screenshot(app.screen, path)
                    app.toast.push(f'Exported screenshot -> {path}')
                except Exception as e:
                    app.toast.push(f'Export failed: {e}')
            elif event.key == pygame.K_o and (event.mod & pygame.KMOD_CTRL):
                path = os.path.join('data', 'env_snapshot.json')
                try:
                    save_env_snapshot(self.env, path)
                    app.toast.push(f'Saved env snapshot -> {path}')
                except Exception as e:
                    app.toast.push(f'Snapshot save failed: {e}')
            elif event.key == pygame.K_i and (event.mod & pygame.KMOD_CTRL):
                path = os.path.join('data', 'env_snapshot.json')
                try:
                    self.env = load_env_snapshot(path)
                    self.obs = self.env._obs()  # type: ignore[attr-defined]
                    app.toast.push(f'Loaded env snapshot <- {path}')
                except Exception as e:
                    app.toast.push(f'Snapshot load failed: {e}')
            elif event.key == pygame.K_x and (event.mod & pygame.KMOD_CTRL):
                try:
                    export_json(self.stats.to_rows(), os.path.join('data', 'run_stats.json'))
                    export_csv(self.stats.to_rows(), os.path.join('data', 'run_stats.csv'))
                    app.toast.push('Exported stats -> data/run_stats.(json|csv)')
                except Exception as e:
                    app.toast.push(f'Export failed: {e}')

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
                self.stats.new_episode()
                self.obs = self.env.reset()
            else:
                self.obs = res.obs

    def _rc(self, app) -> RenderContext:
        w, h = self.env.width, self.env.height
        sw, sh = app.screen.get_size()
        hud_h = int(56 * app.theme.ui_scale)
        usable_h = max(1, sh - hud_h)
        margin = int(14 * app.theme.ui_scale)
        cell = max(10, min((sw - 2 * margin) // w, (usable_h - 2 * margin) // h))
        mx = max(margin, (sw - cell * w) // 2)
        my = max(margin, (usable_h - cell * h) // 2)
        return RenderContext(w=w, h=h, cell=cell, mx=mx, my=my, hud_h=hud_h)

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

    def render(self, app, screen: pygame.Surface) -> None:
        if self.env is None or self.agent is None:
            self._build(app)
        rc = self._rc(app)
        screen.fill(app.theme.palette.bg)

        hm = self._heatmap() if self.show_heatmap else None
        for y in range(self.env.height):
            for x in range(self.env.width):
                r = rc.cell_rect(x, y)
                base = app.theme.palette.grid0 if (x + y) % 2 == 0 else app.theme.palette.grid1
                pygame.draw.rect(screen, base, r)
                if hm is not None:
                    v = float(hm[y, x])
                    if v > 0:
                        shade = int(40 + 180 * v)
                        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                        s.fill((shade, shade, shade, 80))
                        screen.blit(s, r.topleft)
                pygame.draw.rect(screen, app.theme.palette.grid_line, r, 1)

        for hx, hy in self.env.hazards:
            cx, cy = rc.cell_center(hx, hy)
            pygame.draw.circle(screen, app.theme.palette.danger, (cx, cy), max(4, rc.cell // 2 - 8))
            pygame.draw.line(screen, app.theme.palette.fg, (cx - 7, cy - 7), (cx + 7, cy + 7), 2)
            pygame.draw.line(screen, app.theme.palette.fg, (cx + 7, cy - 7), (cx - 7, cy + 7), 2)

        fx, fy = self.env.food
        cx, cy = rc.cell_center(fx, fy)
        pygame.draw.circle(screen, app.theme.palette.ok, (cx, cy), max(3, rc.cell // 2 - 10))

        ax, ay = self.env.agent
        cx, cy = rc.cell_center(ax, ay)
        pygame.draw.circle(screen, app.theme.palette.accent, (cx, cy), max(4, rc.cell // 2 - 6))

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
        self.stats_overlay.render(screen, app.theme, hud)

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
