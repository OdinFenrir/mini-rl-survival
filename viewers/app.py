from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import pygame

from viewers.ui.theme import Theme, palette_for_mode
from viewers.ui.widgets import ToastManager, set_sfx_manager
from viewers.ui.sfx import SfxManager

CONFIG_PATH = os.path.join('data', 'viewer_config.json')
BASE_FONT_SIZE = 22
BASE_TITLE_SIZE = 44


@dataclass
class AppConfig:
    # env
    w: int = 12
    h: int = 12
    hazards: int = 0
    energy_start: int = 25
    energy_food: int = 18
    energy_step: int = 1
    energy_max: int = 0  # 0 = unlimited
    seed: int = 0

    # agent
    alpha: float = 0.2
    gamma: float = 0.97
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 20000

    # view
    render_fps: int = 15
    sim_steps_per_frame: int = 1
    heatmap: bool = True
    policy: bool = False
    qhover: bool = True
    debug: bool = False
    color_mode: str = 'neo'
    font_scale: float = 1.0
    reduced_motion: bool = False
    sound_enabled: bool = True
    menu_background: str = "menu_background.png"

    # levels
    level_mode: str = "preset"
    level_index: int = 0
    level_cycle: bool = True
    n_walls: int = 18
    n_traps: int = 0
    food_enabled: bool = True
    placement_difficulty: str = "medium"

    # io defaults (persist last used paths so the viewer is usable without retyping)
    qtable_path: str = os.path.join('data', 'qtable_saved.pkl')
    env_snapshot_path: str = os.path.join('data', 'env_snapshot.json')

    # training
    train_episodes: int = 2000
    train_max_steps: int = 400
    train_eval_every: int = 200
    train_eval_episodes: int = 50
    train_speed: int = 5  # episodes per update
    train_autosave: bool = True
    train_curriculum: bool = True
    train_curriculum_start: int = 5
    train_curriculum_step: int = 5
    train_curriculum_window: int = 50
    train_curriculum_threshold: float = 0.8
    train_curriculum_eps_rewind: float = 0.5
    train_checkpoint_every: int = 500
    train_use_settings_for_play: bool = True

    # visualization
    viz_level_filter: int = -1  # -1 = all levels
    viz_color: str = "value"
    viz_size: str = "count"
    viz_min_visits: int = 2
    viz_max_points: int = 20000
    viz_action_figure: bool = True
    viz_level_feature: bool = False


def load_config(path: str = CONFIG_PATH) -> AppConfig:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = AppConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if getattr(cfg, "color_mode", None) in ("", None):
            cfg.color_mode = "neo"
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
        self.theme.ui_style = str(self.cfg.color_mode or "default").lower()
        font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "press-start-2p", "fonts", "PressStart2P.ttf"))
        if self.theme.ui_style == "pixel" and os.path.exists(font_path):
            self.theme.font_name = font_path
            self.theme.font_size = 18
            self.theme.font_size_title = 32
        elif self.theme.ui_style != "pixel":
            self.theme.font_name = None
            self.theme.font_size = BASE_FONT_SIZE
            self.theme.font_size_title = BASE_TITLE_SIZE
        self.toast = ToastManager()

        self._scene_stack: list[object] = []
        self._modal_stack: list[object] = []
        self._running = False

        pygame.init()
        pygame.display.set_caption('Mini RL Survival')
        self.screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.sfx = SfxManager(enabled=bool(self.cfg.sound_enabled))
        set_sfx_manager(self.sfx)

    def push_modal(self, modal: object) -> None:
        self._modal_stack.append(modal)

    def pop_modal(self) -> None:
        if self._modal_stack:
            self._modal_stack.pop()

    def modal(self) -> Optional[object]:
        return self._modal_stack[-1] if self._modal_stack else None

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
        self.theme.ui_style = str(self.cfg.color_mode or "default").lower()
        if self.theme.ui_style == "pixel":
            font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "press-start-2p", "fonts", "PressStart2P.ttf"))
            if os.path.exists(font_path):
                self.theme.font_name = font_path
            self.theme.font_size = 18
            self.theme.font_size_title = 32
        else:
            self.theme.font_name = None
            self.theme.font_size = BASE_FONT_SIZE
            self.theme.font_size_title = BASE_TITLE_SIZE
        if hasattr(self, "sfx"):
            self.sfx.set_enabled(bool(self.cfg.sound_enabled))

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
                # If a modal is open, only it handles events
                if self.modal():
                    self.modal().handle_event(event)
                    # Auto-pop modals that closed themselves
                    if hasattr(self.modal(), "active") and not self.modal().active:
                        self.pop_modal()
                else:
                    self.scene().handle_event(self, event)

            if self.modal():
                # Optionally update modal if needed
                pass
            else:
                self.scene().update(self, dt)
            self.toast.update(dt)

            self.scene().render(self, self.screen)
            if self.modal():
                self.modal().render(self.screen, self.theme)
            self.toast.render(self.screen, self.theme)
            pygame.display.flip()

        pygame.quit()
        try:
            save_config(self.cfg)
        except Exception:
            pass
