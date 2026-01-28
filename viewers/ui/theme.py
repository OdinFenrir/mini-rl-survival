from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Palette:
    bg: tuple[int,int,int] = (24, 24, 30)
    panel: tuple[int,int,int] = (0, 0, 0)
    panel_alpha: int = 140
    fg: tuple[int,int,int] = (235, 235, 250)
    muted: tuple[int,int,int] = (180, 180, 210)
    accent: tuple[int,int,int] = (80, 140, 255)
    danger: tuple[int,int,int] = (235, 70, 70)
    ok: tuple[int,int,int] = (70, 235, 120)
    warn: tuple[int,int,int] = (255, 215, 90)
    grid0: tuple[int,int,int] = (44, 44, 56)
    grid1: tuple[int,int,int] = (52, 52, 66)
    grid_line: tuple[int,int,int] = (70, 70, 90)


@dataclass
class Theme:
    ui_scale: float = 1.0
    font_name: str | None = None
    font_size: int = 22
    font_size_title: int = 44
    palette: Palette = field(default_factory=Palette)  # <-- critical fix
    reduced_motion: bool = False


def palette_for_mode(mode: str) -> Palette:
    mode = (mode or "default").lower()
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
            panel_alpha=140,
            fg=(235,235,250),
            muted=(180,180,210),
            accent=(80, 140, 255),
            danger=(180, 60, 255),  # purple
            ok=(255, 200, 60),      # amber
            warn=(255, 255, 90),
            grid0=(44, 44, 56),
            grid1=(52, 52, 66),
            grid_line=(70, 70, 90),
        )
    return Palette()
