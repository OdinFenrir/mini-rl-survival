from __future__ import annotations

import os
from typing import Iterable

import pygame

from viewers.io.run_history import read_entries
from viewers.ui.theme import Theme


class RunHistoryOverlay:
    def __init__(self, max_entries: int = 6) -> None:
        self.max_entries = max_entries
        self.entries: list[dict] = []
        self.reload()

    def reload(self) -> None:
        self.entries = read_entries(self.max_entries)

    def render(self, screen: pygame.Surface, theme: Theme, x: int = 0, y: int = 0, width: int | None = None) -> None:
        if not self.entries:
            return

        font = theme.font(int(theme.font_size * theme.ui_scale))
        pad = int(8 * theme.ui_scale)
        if width is None or width <= 0:
            width = 360
        height = len(self.entries) * int(24 * theme.ui_scale) + 3 * pad

        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((*theme.palette.panel, theme.palette.panel_alpha))
        pygame.draw.rect(panel, theme.palette.grid_line, panel.get_rect(), 1, border_radius=10)
        screen.blit(panel, (x, y))

        title_surf = font.render('Run History', True, theme.palette.accent)
        screen.blit(title_surf, (x + pad, y + pad))
        y_offset = y + pad + title_surf.get_height() + pad // 2

        for entry in reversed(self.entries):
            text = f"Ep {entry.get('episode', '?')}  Reward {entry.get('reward', 0.0):.1f}  Steps {entry.get('steps', 0)}  Term {entry.get('terminal', '')}"
            surf = font.render(text, True, theme.palette.fg)
            screen.blit(surf, (x + pad, y_offset))
            y_offset += surf.get_height() + pad // 2
