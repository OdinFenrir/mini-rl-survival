from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass
class RenderContext:
    w: int
    h: int
    cell: int
    mx: int
    my: int
    hud_h: int

    def cell_rect(self, x: int, y: int) -> pygame.Rect:
        return pygame.Rect(self.mx + x * self.cell, self.my + y * self.cell, self.cell, self.cell)

    def cell_center(self, x: int, y: int):
        r = self.cell_rect(x, y)
        return (r.x + r.w // 2, r.y + r.h // 2)

    def pixel_to_cell(self, px: int, py: int):
        if py > self.my + self.h * self.cell:
            return (None, None)
        x = (px - self.mx) // self.cell
        y = (py - self.my) // self.cell
        if 0 <= x < self.w and 0 <= y < self.h:
            return (int(x), int(y))
        return (None, None)
