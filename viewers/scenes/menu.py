from __future__ import annotations

import pygame

from viewers.scenes.help import HelpScene
from viewers.scenes.settings import SettingsScene
from viewers.scenes.sim import SimulationScene
from viewers.ui.widgets import Button, FocusManager


class MainMenuScene:
    def __init__(self) -> None:
        self.focus = FocusManager()
        self.widgets = []

    def _layout(self, app) -> None:
        w, h = app.screen.get_size()
        bw = int(360 * app.theme.ui_scale)
        bh = int(56 * app.theme.ui_scale)
        gap = int(14 * app.theme.ui_scale)
        x = (w - bw) // 2
        y = h // 2 - int(2.5 * (bh + gap))
        self.widgets = [
            Button(pygame.Rect(x, y + 0 * (bh + gap), bw, bh), 'Start Simulation', lambda: app.push(SimulationScene())),
            Button(pygame.Rect(x, y + 1 * (bh + gap), bw, bh), 'Settings', lambda: app.push(SettingsScene())),
            Button(pygame.Rect(x, y + 2 * (bh + gap), bw, bh), 'Help / About', lambda: app.push(HelpScene())),
            Button(pygame.Rect(x, y + 3 * (bh + gap), bw, bh), 'Exit', lambda: setattr(app, '_running', False)),
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
        title_font = pygame.font.SysFont(app.theme.font_name, int(app.theme.font_size_title * app.theme.ui_scale))
        subtitle_font = pygame.font.SysFont(app.theme.font_name, int(app.theme.font_size * app.theme.ui_scale))
        title = title_font.render('Mini RL Survival', True, app.theme.palette.fg)
        subtitle = subtitle_font.render('Modular Pygame viewer + Q-learning', True, app.theme.palette.muted)
        w, _ = screen.get_size()
        screen.blit(title, (w // 2 - title.get_width() // 2, int(80 * app.theme.ui_scale)))
        screen.blit(subtitle, (w // 2 - subtitle.get_width() // 2, int(140 * app.theme.ui_scale)))

        focused = self.focus.focused()
        for wi in self.widgets:
            wi.draw(screen, app.theme, focused=(wi is focused))

        hint = subtitle_font.render('Tip: Press ? for help in simulation', True, app.theme.palette.muted)
        screen.blit(hint, (w // 2 - hint.get_width() // 2, screen.get_height() - int(60 * app.theme.ui_scale)))
