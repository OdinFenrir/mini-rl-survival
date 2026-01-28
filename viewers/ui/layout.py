
from __future__ import annotations

import pygame

# Minimal layout helpers.

def pad(px: int, scale: float) -> int:
    return int(px * scale)

def vstack(x: int, y: int, w: int, h: int, n: int, gap: int) -> list[pygame.Rect]:
    rects: list[pygame.Rect] = []
    if n <= 0:
        return rects
    item_h = (h - gap * (n - 1)) // n
    for i in range(n):
        rects.append(pygame.Rect(x, y + i * (item_h + gap), w, item_h))
    return rects


def hstack(x: int, y: int, w: int, h: int, n: int, gap: int) -> list[pygame.Rect]:
    rects: list[pygame.Rect] = []
    if n <= 0:
        return rects
    item_w = (w - gap * (n - 1)) // n
    for i in range(n):
        rects.append(pygame.Rect(x + i * (item_w + gap), y, item_w, h))
    return rects
