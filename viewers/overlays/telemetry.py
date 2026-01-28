from __future__ import annotations

import time
from collections import deque

import pygame


class TelemetryOverlay:
    def __init__(self, maxlen: int = 20):
        self.episode_rewards = deque(maxlen=maxlen)
        self.episode_steps = deque(maxlen=maxlen)
        self.terminal_reasons: dict[str, int] = {}
        self.timestamps = deque(maxlen=maxlen)
        self.last_update = time.time()

    def log_episode(self, reward: float, steps: int, terminal_reason: str):
        self.episode_rewards.append(float(reward))
        self.episode_steps.append(int(steps))
        terminal_reason = terminal_reason or "unknown"
        self.terminal_reasons[terminal_reason] = self.terminal_reasons.get(terminal_reason, 0) + 1
        self.timestamps.append(time.time())

    def rolling_avg(self, data) -> float:
        return sum(data) / len(data) if data else 0.0

    def render(self, screen: pygame.Surface, theme, pos=(24, 24)):
        font = theme.font(int(theme.font_size * theme.ui_scale))
        x, y = pos

        # Rolling averages
        avg_reward = self.rolling_avg(self.episode_rewards)
        avg_steps = self.rolling_avg(self.episode_steps)
        txt = f"Telemetry: AvgReward={avg_reward:.2f}  AvgSteps={avg_steps:.1f}  Episodes={len(self.episode_rewards)}"
        surf = font.render(txt, True, theme.palette.accent)
        screen.blit(surf, (x, y))

        # Terminal reasons
        y += 28
        for reason, count in self.terminal_reasons.items():
            rsurf = font.render(f"{reason}: {count}", True, theme.palette.muted)
            screen.blit(rsurf, (x, y))
            y += 22

        # Sparkline for rewards
        if len(self.episode_rewards) > 1:
            max_r = max(self.episode_rewards)
            min_r = min(self.episode_rewards)
            w = 160
            h = 36
            sx, sy = x, y + 8
            pygame.draw.rect(screen, theme.palette.panel, (sx, sy, w, h), border_radius=6)

            if max_r > min_r:
                points = [
                    (
                        sx + i * w // (len(self.episode_rewards) - 1),
                        sy + h - int((r - min_r) / (max_r - min_r + 1e-6) * (h - 6)) - 3,
                    )
                    for i, r in enumerate(self.episode_rewards)
                ]
                if len(points) > 1:
                    pygame.draw.lines(screen, theme.palette.accent, False, points, 2)
