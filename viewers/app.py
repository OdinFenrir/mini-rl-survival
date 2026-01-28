from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import pygame

from viewers.ui.theme import Theme, palette_for_mode
from viewers.ui.widgets import ToastManager

CONFIG_PATH = os.path.join('data', 'viewer_config.json')


@dataclass
class AppConfig:
    # env
    w: int = 10
    h: int = 10
    hazards: int = 8
    energy_start: int = 25
    energy_food: int = 18
    energy_step: int = 1
    seed: int = 0

    # agent
    alpha: float = 0.2
    gamma: float = 0.97
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 20000

    # view
    render_fps: int = 30
    sim_steps_per_frame: int = 1
    heatmap: bool = True
    policy: bool = False
    qhover: bool = True
    debug: bool = False
    color_mode: str = 'default'
    font_scale: float = 1.0
    reduced_motion: bool = False


def load_config(path: str = CONFIG_PATH) -> AppConfig:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = AppConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
    except Exception:
        return AppConfig()


def save_config(cfg: AppConfig, path: str = CONFIG_PATH) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, indent=2)


class App:
    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        self.cfg = cfg or load_config()
        self.theme = Theme(ui_scale=float(self.cfg.font_scale), reduced_motion=bool(self.cfg.reduced_motion))
        self.theme.palette = palette_for_mode(self.cfg.color_mode)
        self.toast = ToastManager()

        self._scene_stack: list[object] = []
        self._running = False

        pygame.init()
        pygame.display.set_caption('Mini RL Survival')
        self.screen = pygame.display.set_mode((1440, 960), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

    def push(self, scene: object) -> None:
        self._scene_stack.append(scene)

    def pop(self) -> None:
        if len(self._scene_stack) > 1:
            self._scene_stack.pop()

    def scene(self) -> object:
        return self._scene_stack[-1]

    def apply_theme_from_config(self) -> None:
        self.theme.ui_scale = float(self.cfg.font_scale)
        self.theme.reduced_motion = bool(self.cfg.reduced_motion)
        self.theme.palette = palette_for_mode(self.cfg.color_mode)

    def run(self) -> None:
        self._running = True
        while self._running:
            dt = self.clock.tick(int(self.cfg.render_fps)) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.scene().handle_event(self, event)

            self.scene().update(self, dt)
            self.toast.update(dt)

            self.scene().render(self, self.screen)
            self.toast.render(self.screen, self.theme)
            pygame.display.flip()

        pygame.quit()
        try:
            save_config(self.cfg)
        except Exception:
            pass





