from __future__ import annotations

import os

import pygame

from viewers.scenes.help import HelpScene
from viewers.scenes.settings import SettingsScene
from viewers.scenes.sim import SimulationScene
from viewers.scenes.training import TrainingScene
from viewers.ui.modals import FileDialogModal
from viewers.ui.widgets import Button, FocusManager


class MainMenuScene:
    def __init__(self) -> None:
        self.focus = FocusManager()
        self.widgets = []
        self._icons: dict[str, pygame.Surface] = {}

    def _load_icons(self, app) -> None:
        if self._icons:
            return
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "kenney_game_icons", "PNG", "Black", "1x"))
        names = {
            "start": "buttonStart.png",
            "load": "forward.png",
            "train": "gamepad1.png",
            "settings": "gear.png",
            "help": "question.png",
            "exit": "exitRight.png",
        }
        color = app.theme.palette.ui_text if app.theme.ui_style == "pixel" else app.theme.palette.fg

        def tint_icon(img: pygame.Surface, tint_color: tuple[int, int, int]) -> pygame.Surface:
            mask = pygame.mask.from_surface(img)
            surf = pygame.Surface(img.get_size(), pygame.SRCALPHA)
            mask.to_surface(surf, setcolor=(*tint_color, 255), unsetcolor=(0, 0, 0, 0))
            return surf

        def make_train_icon(size: int, tint_color: tuple[int, int, int]) -> pygame.Surface:
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pad = max(2, size // 6)
            pygame.draw.rect(surf, tint_color, (pad, size // 2 - 2, size - 2 * pad, 4), border_radius=3)
            pygame.draw.circle(surf, tint_color, (pad, size // 2), max(2, size // 6))
            pygame.draw.circle(surf, tint_color, (size - pad, size // 2), max(2, size // 6))
            pygame.draw.circle(surf, tint_color, (size // 2, size // 2 - size // 5), max(2, size // 6), width=2)
            return surf

        for key, fname in names.items():
            path = os.path.join(base, fname)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                # scale to menu icon size
                target = int(24 * app.theme.ui_scale)
                img = pygame.transform.smoothscale(img, (target, target))
                self._icons[key] = tint_icon(img, color)
        if "train" not in self._icons:
            size = int(24 * app.theme.ui_scale)
            self._icons["train"] = make_train_icon(size, color)

    def _layout(self, app) -> None:
        self._load_icons(app)
        w, h = app.screen.get_size()
        bw = int(460 * app.theme.ui_scale)
        bh = int(60 * app.theme.ui_scale)
        gap = int(14 * app.theme.ui_scale)
        x = (w - bw) // 2
        y = h // 2 - int(3.5 * (bh + gap))

        def load_qtable_and_start() -> None:
            rect = pygame.Rect(0, 0, int(560 * app.theme.ui_scale), int(420 * app.theme.ui_scale))
            rect.center = app.screen.get_rect().center

            def on_confirm(path: str) -> None:
                app.cfg.qtable_path = path
                app.push(SimulationScene())

            initial = getattr(app.cfg, "qtable_path", os.path.join("data", "qtable_saved.pkl"))
            modal = FileDialogModal(
                rect,
                "Load Q-table & Start",
                on_confirm,
                lambda: None,
                initial_path=initial,
                must_exist=True,
            )
            app.push_modal(modal)

        self.widgets = [
            Button(pygame.Rect(x, y + 0 * (bh + gap), bw, bh), 'Start / Continue', lambda: app.push(SimulationScene()), icon=self._icons.get("start")),
            Button(pygame.Rect(x, y + 1 * (bh + gap), bw, bh), 'Load Q-table & Start', load_qtable_and_start, icon=self._icons.get("load")),
            Button(pygame.Rect(x, y + 2 * (bh + gap), bw, bh), 'Train / Learn', lambda: app.push(TrainingScene()), icon=self._icons.get("train")),
            Button(pygame.Rect(x, y + 3 * (bh + gap), bw, bh), 'Settings', lambda: app.push(SettingsScene()), icon=self._icons.get("settings")),
            Button(pygame.Rect(x, y + 4 * (bh + gap), bw, bh), 'Help / About', lambda: app.push(HelpScene()), icon=self._icons.get("help")),
            Button(pygame.Rect(x, y + 5 * (bh + gap), bw, bh), 'Exit', lambda: setattr(app, '_running', False), icon=self._icons.get("exit")),
        ]
        self.focus.set(self.widgets)

    def handle_event(self, app, event: pygame.event.Event) -> None:
        if not self.widgets:
            self._layout(app)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_TAB, pygame.K_DOWN):
                self.focus.next()
            elif event.key == pygame.K_UP:
                self.focus.prev()
            elif event.key == pygame.K_ESCAPE:
                setattr(app, '_running', False)
        focused = self.focus.focused()
        for w in self.widgets:
            w.handle_event(event, focused=(w is focused))

    def update(self, app, dt: float) -> None:
        pass

    def render(self, app, screen: pygame.Surface) -> None:
        if not self.widgets:
            self._layout(app)
        w, h = screen.get_size()
        if app.theme.ui_style == "pixel":
            screen.fill(app.theme.palette.bg)
        else:
            # Soft vertical gradient background
            bg = pygame.Surface((w, h))
            top = app.theme.palette.bg
            bottom = app.theme.palette.grid0
            for y in range(h):
                t = y / max(1, h - 1)
                c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
                pygame.draw.line(bg, c, (0, y), (w, y))
            screen.blit(bg, (0, 0))
        title_font = app.theme.font(int(app.theme.font_size_title * app.theme.ui_scale))
        subtitle_font = app.theme.font(int(app.theme.font_size * 0.85 * app.theme.ui_scale))
        title = title_font.render('Mini RL Survival', True, app.theme.palette.fg)
        subtitle = subtitle_font.render('Modular Pygame viewer + Q-learning', True, app.theme.palette.muted)
        screen.blit(title, (w // 2 - title.get_width() // 2, int(80 * app.theme.ui_scale)))
        screen.blit(subtitle, (w // 2 - subtitle.get_width() // 2, int(140 * app.theme.ui_scale)))
        line_y = int(190 * app.theme.ui_scale)
        pygame.draw.line(screen, app.theme.palette.grid_line, (int(w * 0.2), line_y), (int(w * 0.8), line_y), 2)

        if app.theme.ui_style != "pixel" and self.widgets:
            pad = int(26 * app.theme.ui_scale)
            min_x = min(wi.rect.x for wi in self.widgets)
            max_x = max(wi.rect.right for wi in self.widgets)
            min_y = min(wi.rect.y for wi in self.widgets)
            max_y = max(wi.rect.bottom for wi in self.widgets)
            panel_rect = pygame.Rect(min_x - pad, min_y - pad, (max_x - min_x) + 2 * pad, (max_y - min_y) + 2 * pad)
            shadow = pygame.Surface((panel_rect.w, panel_rect.h), pygame.SRCALPHA)
            pygame.draw.rect(shadow, (0, 0, 0, 80), shadow.get_rect(), border_radius=int(18 * app.theme.ui_scale))
            screen.blit(shadow, (panel_rect.x + int(6 * app.theme.ui_scale), panel_rect.y + int(8 * app.theme.ui_scale)))
            app.theme.draw_gradient_panel(screen, panel_rect, app.theme.palette.panel, app.theme.palette.grid1, border_radius=int(18 * app.theme.ui_scale))

        focused = self.focus.focused()
        for wi in self.widgets:
            wi.draw(screen, app.theme, focused=(wi is focused))

        hint = subtitle_font.render('Tip: Press ? for help in simulation', True, app.theme.palette.muted)
        screen.blit(hint, (w // 2 - hint.get_width() // 2, screen.get_height() - int(60 * app.theme.ui_scale)))
