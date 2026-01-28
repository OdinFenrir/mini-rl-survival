from __future__ import annotations

import pygame

from viewers.keymap import get_keymap
from viewers.ui.theme import Theme


class HelpOverlay:
    def __init__(self, scene_name="Simulation") -> None:
        self.scene_name = scene_name

    def render(self, screen: pygame.Surface, theme: Theme) -> None:
        font = theme.font(int(theme.font_size * theme.ui_scale))
        big = theme.font(int(theme.font_size_title * 0.6 * theme.ui_scale))
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        y = int(40 * theme.ui_scale)
        title = big.render('Help', True, theme.palette.fg)
        overlay.blit(title, (int(40 * theme.ui_scale), y))
        y += int(60 * theme.ui_scale)
        for k in get_keymap(self.scene_name):
            line = f"{k['keys']}: {k['action']} - {k['desc']}"
            s = font.render(line, True, theme.palette.fg)
            overlay.blit(s, (int(40 * theme.ui_scale), y))
            y += int(28 * theme.ui_scale)
        screen.blit(overlay, (0, 0))
