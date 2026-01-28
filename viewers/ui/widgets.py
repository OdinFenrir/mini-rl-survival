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
        font = pygame.font.SysFont(theme.font_name, int(theme.font_size * theme.ui_scale))
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
        self.focusable = False

    def draw(self, screen: pygame.Surface, theme: Theme, focused: bool) -> None:
        font = pygame.font.SysFont(theme.font_name, int(theme.font_size * theme.ui_scale))
        surf = font.render(self.text, True, theme.palette.muted)
        screen.blit(surf, self.rect.topleft)


class Button(Widget):
    def __init__(self, rect: pygame.Rect, text: str, on_click: Callable[[], None]) -> None:
        super().__init__(rect)
        self.text = text
        self.on_click = on_click

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
        bg = theme.palette.grid1 if not focused else theme.palette.accent
        panel = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        panel.fill((*bg, 180 if focused else 140))
        screen.blit(panel, r.topleft)
        pygame.draw.rect(screen, theme.palette.grid_line, r, width=2, border_radius=int(12 * theme.ui_scale))
        font = pygame.font.SysFont(theme.font_name, int(theme.font_size * theme.ui_scale))
        surf = font.render(self.text, True, theme.palette.fg)
        screen.blit(surf, (r.centerx - surf.get_width() // 2, r.centery - surf.get_height() // 2))


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
        panel = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        panel.fill((*theme.palette.grid0, 140))
        screen.blit(panel, r.topleft)
        pygame.draw.rect(screen, theme.palette.grid_line, r, width=2, border_radius=int(12 * theme.ui_scale))
        font = pygame.font.SysFont(theme.font_name, int(theme.font_size * theme.ui_scale))
        label = f"{self.text}: {'ON' if on else 'OFF'}"
        surf = font.render(label, True, theme.palette.fg if focused else theme.palette.muted)
        screen.blit(surf, (r.x + int(12 * theme.ui_scale), r.centery - surf.get_height() // 2))
        dot_r = int(8 * theme.ui_scale)
        cx = r.right - int(20 * theme.ui_scale)
        cy = r.centery
        col = theme.palette.ok if on else theme.palette.danger
        pygame.draw.circle(screen, col, (cx, cy), dot_r)
        if focused:
            pygame.draw.rect(screen, theme.palette.accent, r, width=2, border_radius=int(12 * theme.ui_scale))


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
        panel = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        panel.fill((*theme.palette.grid0, 140))
        screen.blit(panel, r.topleft)
        pygame.draw.rect(screen, theme.palette.grid_line, r, width=2, border_radius=int(12 * theme.ui_scale))
        font = pygame.font.SysFont(theme.font_name, int(theme.font_size * theme.ui_scale))
        v = self.get_value()
        label = f"{self.text}: {self.fmt.format(v)}"
        surf = font.render(label, True, theme.palette.fg if focused else theme.palette.muted)
        screen.blit(surf, (r.x + int(12 * theme.ui_scale), r.y + int(8 * theme.ui_scale)))
        bar_y = r.y + r.h - int(14 * theme.ui_scale)
        bar_x = r.x + int(12 * theme.ui_scale)
        bar_w = r.w - int(24 * theme.ui_scale)
        bar_h = int(6 * theme.ui_scale)
        pygame.draw.rect(screen, theme.palette.grid_line, (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        t = 0.0 if self.vmax == self.vmin else (v - self.vmin) / (self.vmax - self.vmin)
        knob_x = int(bar_x + t * bar_w)
        pygame.draw.circle(screen, theme.palette.accent, (knob_x, bar_y + bar_h // 2), int(8 * theme.ui_scale))
        if focused:
            pygame.draw.rect(screen, theme.palette.accent, r, width=2, border_radius=int(12 * theme.ui_scale))
