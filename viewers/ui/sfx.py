from __future__ import annotations

import math

import numpy as np
import pygame


class SfxManager:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.available = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        if self.enabled:
            self._init_mixer()
            if self.available:
                self._build_sounds()

    def _init_mixer(self) -> None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.available = True
        except Exception:
            self.available = False

    def _tone(self, freq: float, duration: float, volume: float = 0.35, decay: float = 4.0) -> pygame.mixer.Sound:
        init = pygame.mixer.get_init()
        sample_rate = int(init[0]) if init else 44100
        channels = int(init[2]) if init else 1
        count = max(1, int(sample_rate * duration))
        t = np.linspace(0.0, duration, count, endpoint=False)
        envelope = np.exp(-decay * t)
        wave = np.sin(2.0 * math.pi * freq * t) * envelope
        audio = (wave * volume * 32767).astype(np.int16)
        if channels == 2:
            audio = np.repeat(audio[:, None], 2, axis=1)
        return pygame.sndarray.make_sound(audio)

    def _build_sounds(self) -> None:
        self._sounds = {
            "click": self._tone(720, 0.05, volume=0.28, decay=6.0),
            "confirm": self._tone(520, 0.08, volume=0.32, decay=5.0),
            "food": self._tone(980, 0.06, volume=0.30, decay=7.0),
            "hazard": self._tone(160, 0.14, volume=0.35, decay=3.5),
            "error": self._tone(260, 0.10, volume=0.33, decay=4.0),
        }

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.available = False
            self._sounds = {}
            try:
                pygame.mixer.stop()
            except Exception:
                pass
            return
        if not self.available:
            self._init_mixer()
        if self.available and not self._sounds:
            self._build_sounds()

    def play(self, name: str) -> None:
        if not self.enabled or not self.available:
            return
        sound = self._sounds.get(name)
        if sound:
            try:
                sound.play()
            except Exception:
                pass
