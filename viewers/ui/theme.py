from __future__ import annotations

from dataclasses import dataclass, field
import os
import pygame


@dataclass
class Palette:
    bg: tuple[int,int,int] = (22, 22, 28)
    panel: tuple[int,int,int] = (0, 0, 0)
    panel_alpha: int = 165
    fg: tuple[int,int,int] = (245, 245, 255)
    muted: tuple[int,int,int] = (205, 205, 225)
    accent: tuple[int,int,int] = (80, 140, 255)
    danger: tuple[int,int,int] = (235, 70, 70)
    ok: tuple[int,int,int] = (70, 235, 120)
    warn: tuple[int,int,int] = (255, 215, 90)
    grid0: tuple[int,int,int] = (44, 44, 56)
    grid1: tuple[int,int,int] = (52, 52, 66)
    grid_line: tuple[int,int,int] = (95, 95, 120)
    ui_panel: tuple[int,int,int] = (230, 230, 235)
    ui_panel_dark: tuple[int,int,int] = (200, 200, 210)
    ui_border: tuple[int,int,int] = (40, 40, 50)
    ui_text: tuple[int,int,int] = (30, 30, 40)


@dataclass
class Theme:
    ui_scale: float = 1.0
    font_name: str | None = None
    font_size: int = 22
    font_size_title: int = 44
    palette: Palette = field(default_factory=Palette)
    reduced_motion: bool = False
    ui_style: str = "default"

    def font(self, size: int) -> pygame.font.Font:
        if self.font_name and os.path.exists(self.font_name):
            return pygame.font.Font(self.font_name, size)
        return pygame.font.SysFont(self.font_name, size)

    def draw_gradient_panel(self, surface, rect, color1, color2, vertical=True, border_radius=12):
        """Draw a vertical or horizontal gradient panel with rounded corners."""
        if self.ui_style == "pixel":
            border_radius = 0
        if rect.w <= 1 or rect.h <= 1:
            pygame.draw.rect(surface, color1, rect, border_radius=border_radius)
            return
        grad = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        denom = (rect.h - 1) if vertical else (rect.w - 1)
        denom = max(1, denom)
        for i in range(rect.h if vertical else rect.w):
            t = i / denom
            c = tuple(int(color1[j] * (1-t) + color2[j] * t) for j in range(3)) + (self.palette.panel_alpha,)
            if vertical:
                pygame.draw.line(grad, c, (0, i), (rect.w, i))
            else:
                pygame.draw.line(grad, c, (i, 0), (i, rect.h))
        grad = pygame.transform.smoothscale(grad, (rect.w, rect.h))
        surface.blit(grad, rect.topleft)
        pygame.draw.rect(surface, self.palette.grid_line, rect, width=2, border_radius=border_radius)

    def draw_rounded_panel(self, surface, rect, color=None, border_radius=12):
        if self.ui_style == "pixel":
            border_radius = 0
        color = color or self.palette.panel
        panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        panel.fill((*color, self.palette.panel_alpha))
        pygame.draw.rect(panel, color, panel.get_rect(), border_radius=border_radius)
        surface.blit(panel, rect.topleft)
        pygame.draw.rect(surface, self.palette.grid_line, rect, width=2, border_radius=border_radius)


def _pixel_palette() -> Palette:
    return Palette(
        bg=(8, 12, 10),
        panel=(6, 10, 8),
        panel_alpha=210,
        fg=(190, 255, 200),
        muted=(120, 200, 140),
        accent=(90, 255, 140),
        danger=(40, 220, 110),
        ok=(70, 255, 140),
        warn=(150, 255, 170),
        grid0=(16, 24, 18),
        grid1=(20, 30, 22),
        grid_line=(40, 120, 70),
        ui_panel=(20, 30, 22),
        ui_panel_dark=(12, 20, 14),
        ui_border=(80, 200, 120),
        ui_text=(170, 255, 190),
    )

def _neo_palette() -> Palette:
    return Palette(
        bg=(10, 14, 22),
        panel=(18, 24, 34),
        panel_alpha=220,
        fg=(235, 240, 255),
        muted=(165, 175, 195),
        accent=(96, 168, 255),
        danger=(255, 92, 112),
        ok=(86, 224, 160),
        warn=(255, 208, 120),
        grid0=(24, 30, 42),
        grid1=(30, 36, 52),
        grid_line=(72, 84, 112),
        ui_panel=(30, 36, 52),
        ui_panel_dark=(22, 28, 40),
        ui_border=(96, 168, 255),
        ui_text=(225, 232, 255),
    )

def palette_for_mode(mode: str) -> Palette:
    mode = (mode or "default").lower()
    if mode in ("neo", "modern"):
        return _neo_palette()
    if mode == "pixel":
        return _pixel_palette()
    if mode == "high_contrast":
        return Palette(
            bg=(0,0,0),
            panel=(0,0,0),
            panel_alpha=200,
            fg=(255,255,255),
            muted=(210,210,210),
            accent=(0,200,255),
            danger=(255,0,0),
            ok=(0,255,0),
            warn=(255,255,0),
            grid0=(16,16,16),
            grid1=(28,28,28),
            grid_line=(70,70,70),
        )
    if mode == "colorblind":
        # avoid red/green reliance
        return Palette(
            bg=(24,24,30),
            panel=(0,0,0),
            panel_alpha=170,
            fg=(245,245,255),
            muted=(205,205,225),
            accent=(80, 140, 255),
            danger=(180, 60, 255),  # purple
            ok=(255, 200, 60),      # amber
            warn=(255, 255, 90),
            grid0=(44, 44, 56),
            grid1=(52, 52, 66),
            grid_line=(95, 95, 120),
        )
    return Palette()
