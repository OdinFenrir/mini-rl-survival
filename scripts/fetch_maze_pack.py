from __future__ import annotations

import json
import os
import random
import time
import urllib.parse
import urllib.request
from typing import Any


DATASET = "selimaktas/maze-curriculum-dataset"
API_URL = "https://datasets-server.huggingface.co/rows"
CONFIG = "default"
SPLIT = "train"

TARGET_COUNTS = {5: 20, 7: 15, 9: 15}
TRAPS_PER_SIZE = {5: 0, 7: 0, 9: 0}
FOOD_PER_LEVEL = 1
RNG_SEED = 42
PAGE_LEN = 100
MAX_REQUESTS = 200


def _fetch_rows(offset: int, length: int = PAGE_LEN) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{API_URL}?{params}", timeout=30) as resp:
        payload = json.load(resp)
    return payload.get("rows", [])


def _maze_to_layout(maze: str, rng: random.Random, grid_size: int) -> list[str]:
    lines = maze.splitlines()
    if not lines:
        return []
    max_len = max(len(line) for line in lines)
    # Pad with walls so jagged rows never create unintended openings.
    lines = [line.ljust(max_len, "#") for line in lines]

    grid = [list(line) for line in lines]
    start = None
    goal = None
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch == "S":
                start = (x, y)
            elif ch == "E":
                goal = (x, y)

    if start is None or goal is None:
        return []

    open_cells = []
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch != "#":
                open_cells.append((x, y))

    reserved = {start, goal}
    candidates = [pos for pos in open_cells if pos not in reserved]
    rng.shuffle(candidates)

    trap_count = TRAPS_PER_SIZE.get(grid_size, 0)
    traps = candidates[:trap_count]
    food_candidates = [pos for pos in candidates[trap_count:] if pos not in traps]
    foods = food_candidates[:FOOD_PER_LEVEL] if food_candidates else []

    # Apply goal/food/traps
    gx, gy = goal
    grid[gy][gx] = "G"
    sx, sy = start
    grid[sy][sx] = "S"
    for tx, ty in traps:
        grid[ty][tx] = "T"
    for fx, fy in foods:
        grid[fy][fx] = "F"

    return ["".join(row) for row in grid]


def _style_from_layout(layout: list[str]) -> str:
    if not layout:
        return "Unknown"
    w = max(len(line) for line in layout)
    h = len(layout)
    total = max(1, w * h)
    walls = sum(line.count("#") for line in layout)
    ratio = walls / total
    if ratio < 0.28:
        return "Open paths"
    if ratio < 0.42:
        return "Balanced maze"
    return "Tight corridors"


def main() -> None:
    rng = random.Random(RNG_SEED)
    counts = {k: 0 for k in TARGET_COUNTS}
    levels = []

    # Sample offsets across the dataset to mix sizes.
    offsets = [rng.randrange(0, 60000 - PAGE_LEN) for _ in range(MAX_REQUESTS)]
    for offset in offsets:
        rows = _fetch_rows(offset, PAGE_LEN)
        for entry in rows:
            row = entry.get("row", {})
            grid_size = int(row.get("grid_size", 0))
            if grid_size not in TARGET_COUNTS:
                continue
            if counts[grid_size] >= TARGET_COUNTS[grid_size]:
                continue
            maze = row.get("maze", "")
            layout = _maze_to_layout(maze, rng, grid_size)
            if not layout:
                continue
            style = _style_from_layout(layout)
            counts[grid_size] += 1
            idx = counts[grid_size]
            levels.append(
                {
                    "name": f"Maze {grid_size}x{grid_size} #{idx}",
                    "desc": f"Curated maze from the Maze Curriculum Dataset (grid {grid_size}).",
                    "source": "Maze Curriculum Dataset (selimaktas) - MIT License",
                    "layout": layout,
                    "style": style,
                    "grid_size": grid_size,
                }
            )
            if all(counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
                break
        if all(counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
            break

    if not all(counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
        raise SystemExit(f"Not enough mazes collected: {counts}")

    out_dir = os.path.join("assets", "levels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "maze_pack.json")
    payload = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": DATASET,
        "license": "MIT",
        "seed": RNG_SEED,
        "counts": counts,
        "levels": levels,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(levels)} levels to {out_path}")


if __name__ == "__main__":
    main()
