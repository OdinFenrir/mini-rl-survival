from __future__ import annotations

import math

import numpy as np
import pygame

from viewers.ui.theme import Theme

DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


class PolicyOverlay:
    def __init__(self) -> None:
        self._cache_key = None
        self._cache = None

    def _compute(self, agent, w: int, h: int, level_id: int | None = None):
        best = np.full((h, w), -1, dtype=int)
        for s, qvals in getattr(agent, 'Q', {}).items():
            try:
                if len(s) >= 8:
                    lvl = int(s[0])
                    ax, ay = int(s[1]), int(s[2])
                    if level_id is not None and lvl != level_id:
                        continue
                else:
                    ax, ay = int(s[0]), int(s[1])
            except Exception:
                continue
            if 0 <= ax < w and 0 <= ay < h and qvals:
                a = int(max(range(len(qvals)), key=lambda i: qvals[i]))
                best[ay, ax] = a
        return best

    def render(self, screen: pygame.Surface, theme: Theme, agent, rc, level_id: int | None = None, blocked: set | None = None) -> None:
        key = (len(getattr(agent, 'Q', {})), rc.w, rc.h, level_id)
        if key != self._cache_key:
            self._cache_key = key
            self._cache = self._compute(agent, rc.w, rc.h, level_id=level_id)
        best = self._cache
        if best is None:
            return
        for y in range(rc.h):
            for x in range(rc.w):
                if blocked and (x, y) in blocked:
                    continue
                a = int(best[y, x])
                if a < 0:
                    continue
                dx, dy = DIRS.get(a, (0, 0))
                cx, cy = rc.cell_center(x, y)
                L = max(6, int(rc.cell * 0.25))
                ex, ey = cx + dx * L, cy + dy * L
                pygame.draw.line(screen, theme.palette.muted, (cx, cy), (ex, ey), 2)
                ang = math.atan2(ey - cy, ex - cx)
                ah = max(4, int(rc.cell * 0.10))
                left = (ex - ah * math.cos(ang - 0.6), ey - ah * math.sin(ang - 0.6))
                right = (ex - ah * math.cos(ang + 0.6), ey - ah * math.sin(ang + 0.6))
                pygame.draw.polygon(screen, theme.palette.muted, [(ex, ey), left, right])
