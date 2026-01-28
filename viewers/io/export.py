from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable

import pygame


def export_screenshot(screen: pygame.Surface, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pygame.image.save(screen, path)


def export_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def export_csv(rows: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    rows = list(rows)
    if not rows:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write('')
        return
    keys = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
