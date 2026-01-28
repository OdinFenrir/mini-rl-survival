from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

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

    def render(self, screen: pygame.Surface, theme: Theme, hud_text: str) -> None:
        font = pygame.font.SysFont(theme.font_name, int(theme.font_size * theme.ui_scale))
        pad = int(10 * theme.ui_scale)
        surf = font.render(hud_text, True, theme.palette.fg)
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
