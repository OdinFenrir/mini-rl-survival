from __future__ import annotations

import pygame

from viewers.ui.theme import Theme


class DebugOverlay:
    def render(self, screen: pygame.Surface, theme: Theme, obs, last_action, last_reward, last_done, last_info) -> None:
        font = theme.font(int(theme.font_size * theme.ui_scale))
        pad = int(10 * theme.ui_scale)
        lines = [
            f'obs={tuple(obs)}',
            f'action={last_action} reward={last_reward:.3f} done={last_done}',
            f'info={last_info}',
        ]
        widths = [font.size(l)[0] for l in lines]
        h = sum(font.size(l)[1] for l in lines) + pad * (len(lines) + 1)
        w = max(widths) + 2 * pad
        box = pygame.Rect(screen.get_width() - w - pad, pad, w, h)
        panel = pygame.Surface((box.w, box.h), pygame.SRCALPHA)
        panel.fill((*theme.palette.panel, theme.palette.panel_alpha))
        screen.blit(panel, box.topleft)
        y = box.y + pad
        for l in lines:
            s = font.render(l, True, theme.palette.fg)
            screen.blit(s, (box.x + pad, y))
            y += s.get_height() + pad // 2
