"""
#!/usr/bin/env python3
"""
Improved terminal visualization utilities for mini-rl-survival.

Features:
- Pretty box-drawn grid with configurable cell width
- Optional emoji or ASCII symbols
- Colored backgrounds for agent, food, hazards (uses colorama on Windows)
- Legend and status header
- Backwards-compatible alias `render_color(env)` for previous usage
- q_heatmap(agent, env, out) retained (uses matplotlib)
"""

from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt

# try to import colorama for Windows-friendly ANSI handling; fall back gracefully
try:
    import colorama
    colorama.init()
    _HAS_COLORAMA = True
except Exception:
    _HAS_COLORAMA = False

# ANSI color helpers
CSI = "\x1b["
RESET = CSI + "0m"
BOLD = CSI + "1m"

# Foreground colors
FG_BLACK = CSI + "30m"
FG_RED = CSI + "31m"
FG_GREEN = CSI + "32m"
FG_YELLOW = CSI + "33m"
FG_BLUE = CSI + "34m"
FG_MAGENTA = CSI + "35m"
FG_CYAN = CSI + "36m"
FG_WHITE = CSI + "37m"

# Background colors
BG_RED = CSI + "41m"
BG_GREEN = CSI + "42m"
BG_YELLOW = CSI + "43m"
BG_BLUE = CSI + "44m"
BG_MAGENTA = CSI + "45m"
BG_CYAN = CSI + "46m"
BG_WHITE = CSI + "47m"
BG_BLACK = CSI + "40m"

def _wrap(text: str, fg: str = "", bg: str = "") -> str:
    if not fg and not bg:
        return text
    return f"{{fg}}{{bg}}{{text}}{{RESET}}"

# Symbols (emoji preferred, fall back to ASCII if disabled)
SYMBOLS = {
    "agent": "🐺",
    "food": "🍎",
    "hazard": "💥",
    "empty": " ",
}

SYMBOLS_ASCII = {
    "agent": "A",
    "food": "F",
    "hazard": "X",
    "empty": ".",
}


def _cell_content(sym: str, width: int) -> str:
    s = str(sym)
    pad_total = max(0, width - len(s))
    left = pad_total // 2
    right = pad_total - left
    return " " * left + s + " " * right


def render_pretty(
    env,
    use_emoji: bool = True,
    cell_width: int = 3,
    show_legend: bool = True,
    color_bg: bool = True,
) -> str:
    """
    Return a pretty, boxed string representation of the env.

    Parameters:
    - env: GridSurvivalEnv instance
    - use_emoji: prefer emoji symbols; if False use ASCII fallbacks
    - cell_width: width of each cell in characters (recommended 1..5)
    - show_legend: append a legend below the grid
    - color_bg: use colored backgrounds for agent/food/hazard (ANSI); set False to disable colors
    """
    syms = SYMBOLS if use_emoji else SYMBOLS_ASCII

    w = env.width
    h = env.height

    # border pieces
    tl = "┌"
    tr = "┐"
    bl = "└"
    br = "┘"
    hor = "─"
    vert = "│"
    mid_top = "┬"
    mid_bottom = "┴"
    mid_left = "├"
    mid_right = "┤"
    mid = "┼"

    cell_span = hor * cell_width
    top = tl + (cell_span + mid_top) * (w - 1) + cell_span + tr

    rows = []
    hazards = set(getattr(env, "hazards", []))
    food = getattr(env, "food", (None, None))
    agent = getattr(env, "agent", (None, None))

    for y in range(h):
        row_cells = []
        for x in range(w):
            ch = syms["empty"]
            bg = ""
            fg = ""
            if (x, y) in hazards:
                ch = syms["hazard"]
                if color_bg:
                    bg = BG_RED
                    fg = FG_WHITE
            if (x, y) == food:
                ch = syms["food"]
                if color_bg:
                    bg = BG_GREEN
                    fg = FG_BLACK
            if (x, y) == agent:
                ch = syms["agent"]
                if color_bg:
                    bg = BG_BLUE
                    fg = FG_WHITE

            content = _cell_content(ch, cell_width)
            if (fg or bg) and (color_bg and _HAS_COLORAMA or not _HAS_COLORAMA):
                # if colorama present or ANSI likely ok, wrap; otherwise emit plain
                content = _wrap(content, fg=fg, bg=bg)
            row_cells.append(content)
        rows.append(vert + vert.join(row_cells) + vert)

    sep = mid_left + (cell_span + mid) * (w - 1) + cell_span + mid_right
    bottom = bl + (cell_span + mid_bottom) * (w - 1) + cell_span + br

    lines = []
    try:
        stats = f"energy={{env.energy:02d}} steps={{env.steps:03d}}"
    except Exception:
        stats = "energy=? steps=?"
    header = f"{{BOLD}}{{FG_CYAN}}{{stats}}{{RESET}}"
    lines.append(header)
    lines.append(top)
    for i, r in enumerate(rows):
        lines.append(r)
        if i < h - 1:
            lines.append(sep)
    lines.append(bottom)

    if show_legend:
        legend_items = []
        sym_agent = syms["agent"]
        sym_food = syms["food"]
        sym_hazard = syms["hazard"]
        if color_bg:
            legend_items.append(f"{{_wrap(sym_agent, fg=FG_WHITE, bg=BG_BLUE)}} agent")
            legend_items.append(f"{{_wrap(sym_food, fg=FG_BLACK, bg=BG_GREEN)}} food")
            legend_items.append(f"{{_wrap(sym_hazard, fg=FG_WHITE, bg=BG_RED)}} hazard")
        else:
            legend_items.append(f"{{sym_agent}} agent")
            legend_items.append(f"{{sym_food}} food")
            legend_items.append(f"{{sym_hazard}} hazard")
        lines.append("")
        lines.append("  ".join(legend_items))

    return "\n".join(lines)


# Compatibility alias
def render_color(env) -> str:
    return render_pretty(env, use_emoji=True, cell_width=3, show_legend=True, color_bg=True)


def q_heatmap(agent, env, out: str = "q_heatmap.png", vmax: float | None = None) -> None:
    """
    Build and save a heatmap of maxQ over the agent's (x,y) positions.
    We average max_a Q(state) across all states that share the same agent (x,y).
    """
    heat = np.zeros((env.height, env.width), dtype=float)
    counts = np.zeros((env.height, env.width), dtype=int)

    for s, qvals in agent.Q.items():
        try:
            ax, ay = int(s[0]), int(s[1])
        except Exception:
            continue
        val = max(qvals) if qvals else 0.0
        if 0 <= ay < env.height and 0 <= ax < env.width:
            heat[ay, ax] += val
            counts[ay, ax] += 1

    avg = np.zeros_like(heat)
    nonzero = counts > 0
    avg[nonzero] = heat[nonzero] / counts[nonzero]

    fig, ax = plt.subplots(figsize=(max(4, env.width / 2), max(4, env.height / 2)))
    im = ax.imshow(avg, origin="upper", cmap="viridis", vmax=vmax)
    ax.set_title("Average max-Q by agent position (agent_x, agent_y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xticklabels(range(env.width))
    ax.set_yticklabels(range(env.height))
    plt.colorbar(im, ax=ax, label="avg max Q")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)