from __future__ import annotations

import pygame


class HelpScene:
    def handle_event(self, app, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
            app.pop()

    def update(self, app, dt: float) -> None:
        pass

    def render(self, app, screen: pygame.Surface) -> None:
        screen.fill(app.theme.palette.bg)
        title_font = pygame.font.SysFont(app.theme.font_name, int(app.theme.font_size_title * 0.65 * app.theme.ui_scale))
        font = pygame.font.SysFont(app.theme.font_name, int(app.theme.font_size * app.theme.ui_scale))
        lines = [
            'Mini RL Survival — Help',
            '',
            'Simulation:',
            '  Space pause/resume, . step once (paused), R reset, M greedy/eps',
            '  H heatmap, P policy, Q Q-hover, D debug, ? help',
            '  Ctrl+S save Q-table, Ctrl+L load Q-table, Ctrl+E screenshot',
            '  Ctrl+O save env snapshot, Ctrl+I load snapshot, Ctrl+X export stats',
            '',
            'Press Esc to return.',
        ]
        x = int(60 * app.theme.ui_scale)
        y = int(60 * app.theme.ui_scale)
        t = title_font.render(lines[0], True, app.theme.palette.fg)
        screen.blit(t, (x, y))
        y += int(60 * app.theme.ui_scale)
        for line in lines[1:]:
            s = font.render(line, True, app.theme.palette.fg)
            screen.blit(s, (x, y))
            y += int(30 * app.theme.ui_scale)
