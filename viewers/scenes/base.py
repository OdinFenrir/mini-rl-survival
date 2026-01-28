
from __future__ import annotations

import pygame

class Scene:
    """Interface-style base for scenes."""

    def handle_event(self, app, event: pygame.event.Event) -> None:
        raise NotImplementedError

    def update(self, app, dt: float) -> None:
        raise NotImplementedError

    def render(self, app, screen: pygame.Surface) -> None:
        raise NotImplementedError
