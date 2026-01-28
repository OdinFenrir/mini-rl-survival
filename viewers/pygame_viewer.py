from __future__ import annotations

import argparse

from viewers.app import App, AppConfig
from viewers.scenes.menu import MainMenuScene


def main() -> None:
    ap = argparse.ArgumentParser(description='Mini RL Survival - Modular Pygame Viewer')
    ap.add_argument('--reset-config', action='store_true')
    args = ap.parse_args()

    cfg = AppConfig() if args.reset_config else None
    app = App(cfg=cfg)
    app.push(MainMenuScene())
    app.run()


if __name__ == '__main__':
    main()
