from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pygame

from viewers.ui.theme import Theme


@dataclass
class EpisodeStats:
    episode: int = 1
    steps: int = 0
    foods: int = 0
    total_reward: float = 0.0
    terminal: str = ''


@dataclass
class RunStats:
    episodes: List[EpisodeStats] = field(default_factory=list)

    def last(self) -> EpisodeStats:
        if not self.episodes:
            self.episodes.append(EpisodeStats())
        return self.episodes[-1]

    def new_episode(self) -> None:
        ep = self.last()
        if ep.steps or ep.total_reward or ep.terminal or ep.foods:
            self.episodes.append(EpisodeStats(episode=ep.episode + 1))

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {
                'episode': e.episode,
                'steps': e.steps,
                'foods': e.foods,
                'total_reward': e.total_reward,
                'terminal': e.terminal,
            }
            for e in self.episodes
        ]


class StatsOverlay:
    def __init__(self, stats: RunStats) -> None:
        self.stats = stats

    def render(self, screen: pygame.Surface, theme: Theme, hud_text: str, area: Optional[pygame.Rect] = None) -> None:
        font = theme.font(int(theme.font_size * theme.ui_scale))
        pad = int(10 * theme.ui_scale)
        max_w = area.w - 2 * pad if area else screen.get_width() - 2 * pad
        text = hud_text
        if max_w > 0 and font.size(text)[0] > max_w:
            ell = "..."
            ell_w = font.size(ell)[0]
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if font.size(text[:mid])[0] + ell_w <= max_w:
                    lo = mid
                else:
                    hi = mid - 1
            text = text[:lo] + ell if lo > 0 else ell
        surf = font.render(text, True, theme.palette.fg)
        if area:
            box = pygame.Rect(
                area.x + pad,
                area.bottom - surf.get_height() - 2 * pad,
                surf.get_width() + 2 * pad,
                surf.get_height() + 2 * pad,
            )
        else:
            box = pygame.Rect(
                pad,
                screen.get_height() - surf.get_height() - 2 * pad,
                surf.get_width() + 2 * pad,
                surf.get_height() + 2 * pad,
            )
        panel = pygame.Surface((box.w, box.h), pygame.SRCALPHA)
        panel.fill((*theme.palette.panel, theme.palette.panel_alpha))
        screen.blit(panel, box.topleft)
        screen.blit(surf, (box.x + pad, box.y + pad))
