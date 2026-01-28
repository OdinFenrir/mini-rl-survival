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
        for key, fname in names.items():
            path = os.path.join(base, fname)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                # scale to menu icon size
                target = int(24 * app.theme.ui_scale)
                img = pygame.transform.smoothscale(img, (target, target))
                self._icons[key] = img

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
        screen.fill(app.theme.palette.bg)
        title_font = app.theme.font(int(app.theme.font_size_title * app.theme.ui_scale))
        subtitle_font = app.theme.font(int(app.theme.font_size * 0.85 * app.theme.ui_scale))
        title = title_font.render('Mini RL Survival', True, app.theme.palette.fg)
        subtitle = subtitle_font.render('Modular Pygame viewer + Q-learning', True, app.theme.palette.muted)
        w, _ = screen.get_size()
        screen.blit(title, (w // 2 - title.get_width() // 2, int(80 * app.theme.ui_scale)))
        screen.blit(subtitle, (w // 2 - subtitle.get_width() // 2, int(140 * app.theme.ui_scale)))
        line_y = int(190 * app.theme.ui_scale)
        pygame.draw.line(screen, app.theme.palette.grid_line, (int(w * 0.2), line_y), (int(w * 0.8), line_y), 2)

        focused = self.focus.focused()
        for wi in self.widgets:
            wi.draw(screen, app.theme, focused=(wi is focused))

        hint = subtitle_font.render('Tip: Press ? for help in simulation', True, app.theme.palette.muted)
        screen.blit(hint, (w // 2 - hint.get_width() // 2, screen.get_height() - int(60 * app.theme.ui_scale)))
