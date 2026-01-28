from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Iterable

from viewers.app import AppConfig

LOG_PATH = os.path.join('data', 'run_history.jsonl')


def append_entry(
    episode: int,
    steps: int,
    reward: float,
    terminal: str,
    cfg: AppConfig,
    timestamp: float | None = None,
) -> None:
    os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
    entry = {
        'timestamp': float(timestamp or 0.0),
        'episode': int(episode),
        'steps': int(steps),
        'reward': float(reward),
        'terminal': terminal,
        'config': asdict(cfg),
    }
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        json.dump(entry, f)
        f.write('\\n')


def read_entries(limit: int = 6) -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    entries = []
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries[-limit:]
