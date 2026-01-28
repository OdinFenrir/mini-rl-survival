from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pygame


@dataclass
class ToolbarAction:
    title: str
    shortcut: str
    callback: Optional[Callable[[], None]] = None
    kind: str = "action"  # action | header | text


class ActionToolbar:
    def __init__(self, actions_factory: Callable[[], list[ToolbarAction]]) -> None:
        self.actions_factory = actions_factory
        self._click_targets: list[tuple[pygame.Rect, ToolbarAction]] = []
        self._viewport: pygame.Rect | None = None
        self._max_scroll = 0
        self.scroll_y = 0
        self.last_height = 0

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEWHEEL:
            if self._viewport and self._viewport.collidepoint(pygame.mouse.get_pos()):
                # pygame wheel: positive y means scroll up
                self.scroll_y = max(0, min(self._max_scroll, self.scroll_y - int(event.y * 28)))
                return True
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, action in self._click_targets:
                if rect.collidepoint(event.pos) and action.callback:
                    action.callback()
                    return True
        return False

    def render(
        self,
        screen: pygame.Surface,
        theme,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        actions = self.actions_factory()
        if not actions:
            return
        pad = int(12 * theme.ui_scale)
        gap = int(10 * theme.ui_scale)
        header_h = int(24 * theme.ui_scale)
        btn_h = int(50 * theme.ui_scale)
        if width is None or width <= 0:
            width = int(260 * theme.ui_scale)
        if height is None or height <= 0:
            height = screen.get_height() - y

        viewport = pygame.Rect(x, y, width, height)
        self._viewport = viewport

        if theme.ui_style == "pixel":
            panel = pygame.Surface((viewport.w, viewport.h), pygame.SRCALPHA)
            panel.fill((*theme.palette.panel, theme.palette.panel_alpha))
            pygame.draw.rect(panel, theme.palette.grid_line, panel.get_rect(), 1, border_radius=0)
            screen.blit(panel, viewport.topleft)
        else:
            radius = int(14 * theme.ui_scale)
            shadow = pygame.Surface((viewport.w, viewport.h), pygame.SRCALPHA)
            pygame.draw.rect(shadow, (0, 0, 0, 70), shadow.get_rect(), border_radius=radius)
            screen.blit(shadow, (viewport.x + int(4 * theme.ui_scale), viewport.y + int(6 * theme.ui_scale)))
            theme.draw_gradient_panel(
                screen,
                viewport,
                theme.palette.panel,
                theme.palette.grid1,
                border_radius=radius,
            )

        scale = 0.9 if theme.ui_style == "pixel" else 1.0
        font = theme.font(int(theme.font_size * theme.ui_scale * scale))
        font_small = theme.font(max(12, int(theme.font_size * 0.82 * theme.ui_scale * scale)))

        content_y = viewport.y + pad - self.scroll_y
        content_w = max(1, viewport.w - 2 * pad)

        self._click_targets = []
        def _ellipsize(text: str, font: pygame.font.Font, max_w: int) -> str:
            if max_w <= 0:
                return ""
            if font.size(text)[0] <= max_w:
                return text
            ell = "..."
            ell_w = font.size(ell)[0]
            if ell_w > max_w:
                return ""
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if font.size(text[:mid])[0] + ell_w <= max_w:
                    lo = mid
                else:
                    hi = mid - 1
            return text[:lo] + ell

        for action in actions:
            if action.kind == "header":
                header_rect = pygame.Rect(viewport.x + pad, content_y, content_w, header_h)
                if header_rect.bottom >= viewport.y and header_rect.y <= viewport.bottom:
                    title = action.title.upper()
                    title = _ellipsize(title, font_small, header_rect.w)
                    surf = font_small.render(title, True, theme.palette.muted)
                    screen.blit(surf, (header_rect.x, header_rect.y))
                content_y += header_h + gap // 2
                continue

            if action.kind == "text":
                text_rect = pygame.Rect(viewport.x + pad, content_y, content_w, header_h)
                if text_rect.bottom >= viewport.y and text_rect.y <= viewport.bottom:
                    title = _ellipsize(action.title, font_small, text_rect.w)
                    surf = font_small.render(title, True, theme.palette.fg)
                    screen.blit(surf, (text_rect.x, text_rect.y))
                content_y += header_h + gap // 2
                continue

            rect = pygame.Rect(viewport.x + pad, content_y, content_w, btn_h)
            self._click_targets.append((rect, action))
            if rect.bottom >= viewport.y and rect.y <= viewport.bottom:
                pygame.draw.rect(screen, theme.palette.grid0, rect, border_radius=10)
                pygame.draw.rect(screen, theme.palette.grid_line, rect, 1, border_radius=10)
                hint_text = _ellipsize(action.shortcut, font_small, max(0, rect.w // 3))
                hint_surf = font_small.render(hint_text, True, theme.palette.muted)
                title_max = rect.w - 20 - hint_surf.get_width() - 16
                title_text = _ellipsize(action.title, font, max(0, title_max))
                title_surf = font.render(title_text, True, theme.palette.fg)
                screen.blit(title_surf, (rect.x + 10, rect.y + (rect.h - title_surf.get_height()) // 2))
                screen.blit(hint_surf, (rect.right - hint_surf.get_width() - 10, rect.y + (rect.h - hint_surf.get_height()) // 2))
            content_y += btn_h + gap

        content_height = (content_y + self.scroll_y) - (viewport.y + pad)
        self._max_scroll = max(0, int(content_height - viewport.h + pad))
        self.scroll_y = max(0, min(self._max_scroll, self.scroll_y))
        self.last_height = viewport.h
