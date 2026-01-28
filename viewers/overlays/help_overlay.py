from __future__ import annotations

import pygame

from viewers.ui.theme import Theme


class HelpOverlay:
    def __init__(self) -> None:
        self.lines = [
            'Controls:',
            '  Space: pause/resume',
            '  . : step once (when paused)',
            '  R: reset episode',
            '  M: toggle greedy/epsilon',
            '  H: toggle heatmap',
            '  P: toggle policy arrows',
            '  Q: toggle Q hover panel',
            '  D: toggle debug overlay',
            '  Ctrl+S: save Q-table',
            '  Ctrl+L: load Q-table',
            '  Ctrl+E: export screenshot',
            '  Ctrl+O: save env snapshot',
            '  Ctrl+I: load env snapshot',
            '  Ctrl+X: export stats json/csv',
            '  Esc: back',
            '  ?: toggle this help',
        ]

    def render(self, screen: pygame.Surface, theme: Theme) -> None:
        font = pygame.font.SysFont(theme.font_name, int(theme.font_size * theme.ui_scale))
        big = pygame.font.SysFont(theme.font_name, int(theme.font_size_title * 0.6 * theme.ui_scale))
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        y = int(40 * theme.ui_scale)
        title = big.render('Help', True, theme.palette.fg)
        overlay.blit(title, (int(40 * theme.ui_scale), y))
        y += int(60 * theme.ui_scale)
        for line in self.lines:
            s = font.render(line, True, theme.palette.fg)
            overlay.blit(s, (int(40 * theme.ui_scale), y))
            y += int(28 * theme.ui_scale)
        screen.blit(overlay, (0, 0))
