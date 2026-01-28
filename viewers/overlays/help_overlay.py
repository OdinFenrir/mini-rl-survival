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
        pad = int(40 * theme.ui_scale)
        col_gap = int(32 * theme.ui_scale)
        line_h = int(26 * theme.ui_scale)
        title = big.render('Help', True, theme.palette.fg)
        overlay.blit(title, (pad, pad))
        start_y = pad + title.get_height() + int(20 * theme.ui_scale)

        max_w = max(200, screen.get_width() - 2 * pad)
        cols = 2 if max_w >= 900 else 1
        col_w = max(240, (max_w - (cols - 1) * col_gap) // cols)

        def wrap_line(text: str) -> list[str]:
            if font.size(text)[0] <= col_w:
                return [text]
            words = text.split(" ")
            out: list[str] = []
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if font.size(test)[0] <= col_w:
                    cur = test
                else:
                    if cur:
                        out.append(cur)
                    cur = w
            if cur:
                out.append(cur)
            return out

        x = pad
        y = start_y
        col_limit = screen.get_height() - pad
        for k in get_keymap(self.scene_name):
            line = f"{k['keys']}: {k['action']} - {k['desc']}"
            for part in wrap_line(line):
                if y + line_h > col_limit and cols > 1:
                    x += col_w + col_gap
                    y = start_y
                if y + line_h > col_limit:
                    break
                s = font.render(part, True, theme.palette.fg)
                overlay.blit(s, (x, y))
                y += line_h
            if y + line_h > col_limit and x + col_w + col_gap <= screen.get_width() - pad:
                x += col_w + col_gap
                y = start_y
        screen.blit(overlay, (0, 0))
