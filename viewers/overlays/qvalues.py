from __future__ import annotations

import pygame

from viewers.ui.theme import Theme


class QValuesOverlay:
    def render(self, screen: pygame.Surface, theme: Theme, agent, rc, mouse_pos) -> None:
        x, y = rc.pixel_to_cell(mouse_pos[0], mouse_pos[1])
        if x is None:
            return
        candidates = []
        for s, q in getattr(agent, 'Q', {}).items():
            try:
                ax, ay = int(s[0]), int(s[1])
            except Exception:
                continue
            if ax == x and ay == y:
                candidates.append((s, q))
        if not candidates:
            return
        s, q = max(candidates, key=lambda sq: int(sq[0][4]) if len(sq[0]) > 4 else 0)
        font = theme.font(int(theme.font_size * theme.ui_scale))
        pad = int(10 * theme.ui_scale)
        lines = [
            f'cell=({x},{y}) state={tuple(s)}',
            f'Q: [U={q[0]:.2f} R={q[1]:.2f} D={q[2]:.2f} L={q[3]:.2f}]',
        ]
        widths = [font.size(l)[0] for l in lines]
        box_w = max(widths) + 2 * pad
        box_h = sum(font.size(l)[1] for l in lines) + pad * (len(lines) + 1)
        bx = min(screen.get_width() - box_w - pad, mouse_pos[0] + pad)
        by = min(screen.get_height() - box_h - pad, mouse_pos[1] + pad)
        panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        panel.fill((*theme.palette.panel, theme.palette.panel_alpha))
        screen.blit(panel, (bx, by))
        yy = by + pad
        for l in lines:
            surf = font.render(l, True, theme.palette.fg)
            screen.blit(surf, (bx + pad, yy))
            yy += surf.get_height() + pad // 2
