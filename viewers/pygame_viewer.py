from __future__ import annotations

import argparse
import sys
import traceback

from viewers.app import App, AppConfig
from viewers.scenes.menu import MainMenuScene


def main() -> None:
    ap = argparse.ArgumentParser(description='Mini RL Survival - Modular Pygame Viewer')
    ap.add_argument('--reset-config', action='store_true')
    args = ap.parse_args()

    cfg = AppConfig() if args.reset_config else None
    app = App(cfg=cfg)
    app.push(MainMenuScene())
    try:
        app.run()
    except Exception as e:
        # Write crash dump
        import os, datetime
        dump_dir = 'data'
        os.makedirs(dump_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dump_path = os.path.join(dump_dir, f'crash_dump_{ts}.txt')
        with open(dump_path, 'w', encoding='utf-8') as f:
            f.write('Mini RL Survival Crash Dump\n')
            f.write(f'Time: {ts}\n')
            f.write(f'Exception: {e}\n')
            f.write('Traceback:\n')
            traceback.print_exc(file=f)
            f.write('\n')
            f.write('AppConfig:\n')
            f.write(str(app.cfg))
        # Show crash screen
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((800, 400))
        pygame.display.set_caption('Mini RL Survival - Crash')
        font = pygame.font.SysFont(None, 32)
        msg = f"A fatal error occurred.\nCrash dump written to {dump_path}\n\nPress any key to exit."
        screen.fill((30, 0, 0))
        y = 60
        for line in msg.split('\n'):
            surf = font.render(line, True, (255, 200, 200))
            screen.blit(surf, (40, y))
            y += 48
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    waiting = False
        pygame.quit()
        sys.exit(1)


if __name__ == '__main__':
    main()
