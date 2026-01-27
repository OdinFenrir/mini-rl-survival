"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# ANSI color helpers (works in most terminals)
CSI = "\x1b["
RESET = CSI + "0m"
BOLD = CSI + "1m"
FG_RED = CSI + "31m"
FG_GREEN = CSI + "32m"
FG_YELLOW = CSI + "33m"
FG_BLUE = CSI + "34m"
BG_WHITE = CSI + "47m"

def _cell_str(ch: str, color: str = "") -> str:
    if color:
        return f"{color}{ch}{RESET}"
    return ch

def render_color(env) -> str:
    """
    Return a colored string representation of the env grid using emojis + ANSI.
    Designed to be printed to a terminal.
    """
    # Safe fallbacks (emoji may not render everywhere)
    AGENT = "🐺"  # OdinFenrir wolf — fallback to 'A' if terminal/console doesn't support emoji
    FOOD = "🍎"
    HAZARD = "💥"

    # Use single-char fallbacks for narrow terminals
    try:
        # construct grid
        grid = [["." for _ in range(env.width)] for __ in range(env.height)]
        for (hx, hy) in env.hazards:
            grid[hy][hx] = HAZARD
        fx, fy = env.food
        grid[fy][fx] = FOOD
        ax, ay = env.agent
        grid[ay][ax] = AGENT
    except Exception:
        return env.render()  # fallback to original ASCII if something weird

    lines = []
    for y, row in enumerate(grid):
        cols = []
        for x, ch in enumerate(row):
            if ch == AGENT:
                cols.append(_cell_str(ch, FG_BLUE + BOLD))
            elif ch == FOOD:
                cols.append(_cell_str(ch, FG_GREEN + BOLD))
            elif ch == HAZARD:
                cols.append(_cell_str(ch, FG_RED))
            else:
                cols.append(".")  # plain dot for empty
        lines.append(" ".join(cols))
    stats = f"energy={env.energy:02d} steps={env.steps:03d}"
    header = f"{BOLD}{FG_YELLOW}{stats}{RESET}"
    return header + "\n" + "\n".join(lines)

def q_heatmap(agent, env, out: str = "q_heatmap.png", vmax: float | None = None) -> None:
    """
    Build and save a heatmap of maxQ over the agent's (x,y) positions.
    We average max_a Q(state) across all states that share the same agent (x,y),
    since the full state includes food and energy.
    """
    # Prepare accumulation arrays
    heat = np.zeros((env.height, env.width), dtype=float)
    counts = np.zeros((env.height, env.width), dtype=int)

    # agent.Q is a dict mapping State -> list[float]
    # State is (ax, ay, fx, fy, energy)
    for s, qvals in agent.Q.items():
        try:
            ax, ay = s[0], s[1]
        except Exception:
            continue
        val = max(qvals) if qvals else 0.0
        # accumulate
        if 0 <= ay < env.height and 0 <= ax < env.width:
            heat[ay, ax] += val
            counts[ay, ax] += 1

    # avoid divide-by-zero
    avg = np.zeros_like(heat)
    nonzero = counts > 0
    avg[nonzero] = heat[nonzero] / counts[nonzero]

    # If there are empty cells the visualization will show them in gray
    fig, ax = plt.subplots(figsize=(max(4, env.width/2), max(4, env.height/2)))
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
"""