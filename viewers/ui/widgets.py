from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pygame

from .theme import Theme


@dataclass
class Toast:
    text: str
    ttl: float


class ToastManager:
    def __init__(self) -> None:
        self._items: list[Toast] = []

    def push(self, text: str, ttl: float = 2.5) -> None:
        self._items.append(Toast(text=text, ttl=ttl))

    def update(self, dt: float) -> None:
        for t in self._items:
            t.ttl -= dt
        self._items = [t for t in self._items if t.ttl > 0]

    def render(self, screen: pygame.Surface, theme: Theme) -> None:
        if not self._items:
            return
        font = theme.font(int(theme.font_size * theme.ui_scale))
        pad = int(10 * theme.ui_scale)
        x, y = pad, pad
        for t in self._items[-3:]:
            surf = font.render(t.text, True, theme.palette.fg)
            box = pygame.Rect(x, y, surf.get_width() + 2 * pad, surf.get_height() + 2 * pad)
            panel = pygame.Surface((box.w, box.h), pygame.SRCALPHA)
            panel.fill((*theme.palette.panel, theme.palette.panel_alpha))
            screen.blit(panel, box.topleft)
            screen.blit(surf, (x + pad, y + pad))
            y += box.h + pad


class FocusManager:
    def __init__(self) -> None:
        self.items: list[Widget] = []
        self.index: int = 0

    def set(self, items: list['Widget']) -> None:
        self.items = [w for w in items if w.focusable]
        self.index = 0

    def next(self) -> None:
        if self.items:
            self.index = (self.index + 1) % len(self.items)

    def prev(self) -> None:
        if self.items:
            self.index = (self.index - 1) % len(self.items)

    def focused(self) -> Optional['Widget']:
        if not self.items:
            return None
        self.index = max(0, min(self.index, len(self.items) - 1))
        return self.items[self.index]


class Widget:
    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.focusable = True
        self.enabled = True

    def handle_event(self, event: pygame.event.Event, focused: bool) -> bool:
        return False

    def draw(self, screen: pygame.Surface, theme: Theme, focused: bool) -> None:
        pass


class Label(Widget):
    def __init__(self, rect: pygame.Rect, text: str) -> None:
        super().__init__(rect)
        self.text = text
        self.label = text
        self.focusable = False

    def draw(self, screen: pygame.Surface, theme: Theme, focused: bool) -> None:
        font = theme.font(int(theme.font_size * theme.ui_scale))
        surf = font.render(self.text, True, theme.palette.muted)
        screen.blit(surf, self.rect.topleft)


class Button(Widget):
    def __init__(self, rect: pygame.Rect, text: str, on_click: Callable[[], None], icon: pygame.Surface | None = None) -> None:
        super().__init__(rect)
        self.text = text
        self.on_click = on_click
        self.icon = icon

    def handle_event(self, event: pygame.event.Event, focused: bool) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.on_click()
            return True
        if focused and event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.on_click()
            return True
        return False

    def draw(self, screen: pygame.Surface, theme: Theme, focused: bool) -> None:
        r = self.rect
        scale = float(theme.ui_scale)
        if theme.ui_style == "pixel":
            base = theme.palette.ui_panel
            border = theme.palette.ui_border
            hi = theme.palette.ui_panel
            lo = theme.palette.ui_panel_dark
            pygame.draw.rect(screen, base, r)
            pygame.draw.rect(screen, border, r, width=2)
            pygame.draw.line(screen, hi, (r.x + 2, r.y + 2), (r.right - 3, r.y + 2), 2)
            pygame.draw.line(screen, hi, (r.x + 2, r.y + 2), (r.x + 2, r.bottom - 3), 2)
            pygame.draw.line(screen, lo, (r.x + 2, r.bottom - 3), (r.right - 3, r.bottom - 3), 2)
            pygame.draw.line(screen, lo, (r.right - 3, r.y + 2), (r.right - 3, r.bottom - 3), 2)
            if focused:
                pygame.draw.rect(screen, theme.palette.accent, r, width=2)
            font = theme.font(int(theme.font_size * scale))
            text_color = theme.palette.ui_text
            icon = getattr(self, "icon", None)
            icon_w = icon.get_width() if icon else 0
            gap = int(8 * scale) if icon else 0
            def ellipsize(text: str, max_w: int) -> str:
                if max_w <= 0:
                    return ""
                if font.size(text)[0] <= max_w:
                    return text
                ell = "..."
                ell_w = font.size(ell)[0]
                if ell_w >= max_w:
                    return ell
                lo_i, hi_i = 0, len(text)
                while lo_i < hi_i:
                    mid = (lo_i + hi_i + 1) // 2
                    if font.size(text[:mid])[0] + ell_w <= max_w:
                        lo_i = mid
                    else:
                        hi_i = mid - 1
                return text[:lo_i] + ell

            pad_x = int(14 * scale)
            text_x = r.x + pad_x + icon_w + gap
            max_w = r.w - pad_x * 2 - icon_w - gap
            label = ellipsize(self.text, max_w)
            text_y = r.centery - font.size(label)[1] // 2
            if icon:
                icon_y = r.centery - icon.get_height() // 2
                screen.blit(icon, (r.x + int(12 * scale), icon_y))
            surf = font.render(label, True, text_color)
            screen.blit(surf, (text_x, text_y))
        else:
            if focused:
                theme.draw_gradient_panel(screen, r, theme.palette.accent, theme.palette.accent, border_radius=int(14 * theme.ui_scale))
            else:
                theme.draw_rounded_panel(screen, r, color=theme.palette.grid1, border_radius=int(12 * theme.ui_scale))
            font = theme.font(int(theme.font_size * theme.ui_scale))
            surf = font.render(self.text, True, theme.palette.fg)
            screen.blit(surf, (r.centerx - surf.get_width() // 2, r.centery - surf.get_height() // 2))
            if focused:
                pygame.draw.rect(screen, theme.palette.accent, r, width=4, border_radius=int(14 * theme.ui_scale))


class Toggle(Widget):
    def __init__(self, rect: pygame.Rect, text: str, get_value: Callable[[], bool], set_value: Callable[[bool], None]) -> None:
        super().__init__(rect)
        self.text = text
        self.get_value = get_value
        self.set_value = set_value

    def handle_event(self, event: pygame.event.Event, focused: bool) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.set_value(not self.get_value())
            return True
        if focused and event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.set_value(not self.get_value())
            return True
        return False

    def draw(self, screen: pygame.Surface, theme: Theme, focused: bool) -> None:
        r = self.rect
        on = self.get_value()
        theme.draw_rounded_panel(screen, r, color=theme.palette.grid0, border_radius=int(12 * theme.ui_scale))
        font = theme.font(int(theme.font_size * theme.ui_scale))
        label = f"{self.text}: {'ON' if on else 'OFF'}"
        surf = font.render(label, True, theme.palette.fg if focused else theme.palette.muted)
        screen.blit(surf, (r.x + int(12 * theme.ui_scale), r.centery - surf.get_height() // 2))
        dot_r = int(8 * theme.ui_scale)
        cx = r.right - int(20 * theme.ui_scale)
        cy = r.centery
        col = theme.palette.ok if on else theme.palette.danger
        pygame.draw.circle(screen, col, (cx, cy), dot_r)
        if focused:
            pygame.draw.rect(screen, theme.palette.accent, r, width=4, border_radius=int(14 * theme.ui_scale))


class Slider(Widget):
    def __init__(self, rect: pygame.Rect, text: str, vmin: float, vmax: float, step: float,
                 get_value: Callable[[], float], set_value: Callable[[float], None], fmt: str = '{:.2f}') -> None:
        super().__init__(rect)
        self.text = text
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.step = float(step)
        self.get_value = get_value
        self.set_value = set_value
        self.fmt = fmt

    def _clamp(self, v: float) -> float:
        v = max(self.vmin, min(self.vmax, v))
        if self.step > 0:
            k = round((v - self.vmin) / self.step)
            v = self.vmin + k * self.step
            v = max(self.vmin, min(self.vmax, v))
        return v

    def handle_event(self, event: pygame.event.Event, focused: bool) -> bool:
        if not self.enabled:
            return False
        if focused and event.type == pygame.KEYDOWN and event.key in (pygame.K_LEFT, pygame.K_a):
            self.set_value(self._clamp(self.get_value() - self.step))
            return True
        if focused and event.type == pygame.KEYDOWN and event.key in (pygame.K_RIGHT, pygame.K_d):
            self.set_value(self._clamp(self.get_value() + self.step))
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            t = (event.pos[0] - self.rect.x) / max(1, self.rect.w)
            v = self.vmin + t * (self.vmax - self.vmin)
            self.set_value(self._clamp(v))
            return True
        return False

    def draw(self, screen: pygame.Surface, theme: Theme, focused: bool) -> None:
        r = self.rect
        theme.draw_rounded_panel(screen, r, color=theme.palette.panel, border_radius=int(12 * theme.ui_scale))
        font = theme.font(int(theme.font_size * theme.ui_scale))
        label_surf = font.render(self.text, True, theme.palette.fg)
        val = self.get_value()
        val_surf = font.render(self.fmt.format(val), True, theme.palette.accent if focused else theme.palette.fg)
        # Draw bar
        bar_rect = pygame.Rect(r.x + 10, r.centery - 8, r.w - 120, 16)
        pygame.draw.rect(screen, theme.palette.panel, bar_rect, border_radius=8)
        denom = max(1e-9, self.vmax - self.vmin)
        fill_w = int((val - self.vmin) / denom * (bar_rect.w - 4))
        pygame.draw.rect(screen, theme.palette.accent, (bar_rect.x + 2, bar_rect.y + 2, fill_w, bar_rect.h - 4), border_radius=6)
        if focused:
            pygame.draw.rect(screen, theme.palette.accent, r, width=4, border_radius=int(14 * theme.ui_scale))
        # Draw label and value
        screen.blit(label_surf, (r.x + 10, r.y + 4))
        screen.blit(val_surf, (r.right - val_surf.get_width() - 16, r.y + 4))
