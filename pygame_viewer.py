#!/usr/bin/env python3
"""
Pygame interactive viewer demo for a grid survival environment.

Controls (when window focused):
- Space: Play / Pause
- Right Arrow: Step forward one frame
- Left Arrow: Step backward (replay from start to previous step)
- +/- : Faster / Slower playback
- H: Toggle heatmap overlay (requires a loaded qtable)
- S: Save an animated GIF of the current episode
- Q or ESC: Quit

Usage examples:
  python pygame_viewer.py                 # run demo environment (random agent)
  python pygame_viewer.py --load qtable.pkl  # run using saved Q-table (if you have one)

Dependencies: pygame, numpy, imageio, matplotlib (for heatmap generation)
"""
import argparse
import math
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy imports for optional features
try:
    import pygame
except Exception as e:
    raise SystemExit("pygame is required. Install with: pip install pygame")

try:
    import imageio
except Exception:
    imageio = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except Exception:
    plt = None


@dataclass
class StepResult:
    obs: Tuple[int, int, int, int, int]
    reward: float
    done: bool
    info: dict


class GridSurvivalEnv:
    """Simple deterministic grid survival environment for demo purposes."""

    def __init__(self, width=10, height=10, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        # Place hazards randomly, place food and agent
        self.hazards = set()
        for _ in range((self.width * self.height) // 10):
            x = self.rng.randint(0, self.width)
            y = self.rng.randint(0, self.height)
            self.hazards.add((x, y))
        # Ensure hazards don't fully crowd
        self.food = (self.rng.randint(self.width), self.rng.randint(self.height))
        # Place agent far from food if possible
        while True:
            ax = self.rng.randint(self.width)
            ay = self.rng.randint(self.height)
            if (ax, ay) != self.food and (ax, ay) not in self.hazards:
                break
        self.agent = (ax, ay)
        self.energy = 25
        self.steps = 0
        self.done = False
        return self._observe()

    def _observe(self):
        fx, fy = self.food
        ax, ay = self.agent
        return (ax, ay, fx, fy, self.energy)

    def step(self, action: int) -> StepResult:
        # actions: 0=up,1=right,2=down,3=left
        if self.done:
            return StepResult(self._observe(), 0.0, True, {})
        ax, ay = self.agent
        if action == 0:
            ay = max(0, ay - 1)
        elif action == 1:
            ax = min(self.width - 1, ax + 1)
        elif action == 2:
            ay = min(self.height - 1, ay + 1)
        elif action == 3:
            ax = max(0, ax - 1)
        self.agent = (ax, ay)
        self.steps += 1
        self.energy -= 1
        reward = -0.01
        if self.agent == self.food:
            reward += 1.0
            # respawn food elsewhere
            while True:
                fx = self.rng.randint(0, self.width)
                fy = self.rng.randint(0, self.height)
                if (fx, fy) not in self.hazards and (fx, fy) != self.agent:
                    break
            self.food = (fx, fy)
            self.energy = min(25, self.energy + 10)
        if self.agent in self.hazards:
            reward -= 1.0
            self.done = True
        if self.energy <= 0:
            self.done = True
        return StepResult(self._observe(), reward, self.done, {"steps": self.steps})


class QLearningAgent:
    """Simple Q-table wrapper used by the viewer if a qtable is provided."""

    def __init__(self, n_actions=4, qtable: Optional[Dict] = None):
        self.n_actions = n_actions
        self.Q = qtable or {}

    def act(self, obs, greedy=True):
        # obs: (ax,ay,fx,fy,energy)
        key = tuple(obs)
        if key in self.Q:
            qvals = self.Q[key]
            if greedy:
                return int(np.argmax(qvals))
            else:
                # epsilon-greedy fallback: pick argmax
                return int(np.argmax(qvals))
        # fallback random
        return int(np.random.randint(0, self.n_actions))

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(n_actions=4, qtable=data)


# Viewer implementation
class PygameViewer:
    CELL = 48
    MARGIN = 8

    def __init__(self, env: GridSurvivalEnv, agent: QLearningAgent, fps: float = 8.0):
        self.env = env
        self.agent = agent
        self.fps = fps
        self.playing = True
        self.frames: List[np.ndarray] = []
        self.frame_surfaces: List[pygame.Surface] = []
        self.heatmap_img = None

        pygame.init()
        grid_w = env.width
        grid_h = env.height
        window_w = grid_w * self.CELL + self.MARGIN * 2
        window_h = grid_h * self.CELL + self.MARGIN * 2 + 40
        self.screen = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption("Mini RL Survival - Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)

        # Replay buffer: store observations for replay step-back
        self.history: List[Tuple] = []
        self.reset_episode()

    def reset_episode(self):
        obs = self.env.reset()
        self.history = [obs]
        self.frames = [self._render_frame(obs)]
        self.current_step = 0
        self.playing = True
        self.heatmap_img = None

    def _render_frame(self, obs) -> np.ndarray:
        # create an RGB image array of the grid to allow saving frames
        surf = pygame.Surface((self.env.width * self.CELL, self.env.height * self.CELL))
        surf.fill((30, 30, 30))
        ax, ay, fx, fy, energy = obs
        for y in range(self.env.height):
            for x in range(self.env.width):
                rect = pygame.Rect(x * self.CELL, y * self.CELL, self.CELL, self.CELL)
                color = (220, 220, 220)
                if (x, y) in self.env.hazards:
                    color = (200, 60, 60)
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, (40, 40, 40), rect, 1)
        # draw food
        pygame.draw.circle(
            surf,
            (60, 200, 60),
            (fx * self.CELL + self.CELL // 2, fy * self.CELL + self.CELL // 2),
            self.CELL // 3,
        )
        # draw agent
        pygame.draw.circle(
            surf,
            (80, 160, 255),
            (ax * self.CELL + self.CELL // 2, ay * self.CELL + self.CELL // 2),
            self.CELL // 2 - 4,
        )
        # Blit to a string buffer
        arr = pygame.surfarray.array3d(surf)
        arr = np.transpose(arr, (1, 0, 2))  # convert to HxWx3
        return arr

    def compute_heatmap(self):
        if not self.agent or not getattr(self.agent, "Q", None):
            return
        heat = np.zeros((self.env.height, self.env.width), dtype=float)
        counts = np.zeros_like(heat, dtype=int)
        for s, qvals in self.agent.Q.items():
            try:
                ax, ay = int(s[0]), int(s[1])
            except Exception:
                continue
            if 0 <= ay < self.env.height and 0 <= ax < self.env.width:
                heat[ay, ax] += max(qvals) if qvals else 0.0
                counts[ay, ax] += 1
        avg = np.zeros_like(heat)
        nz = counts > 0
        avg[nz] = heat[nz] / counts[nz]
        # normalize for display
        mx = max(1e-6, float(np.nanmax(avg)))
        norm = avg / mx
        # render to pygame surface via matplotlib if available
        if plt is not None:
            fig = plt.figure(figsize=(self.env.width / 2, self.env.height / 2))
            ax = fig.add_subplot(111)
            ax.imshow(norm, origin='upper', cmap=cm.viridis)
            ax.axis("off")
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            self.heatmap_img = img
        else:
            # fallback simple heat as RGB
            h, w = avg.shape
            out = np.zeros((h * self.CELL, w * self.CELL, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    v = int(255 * (avg[y, x] / (mx + 1e-6)))
                    out[y * self.CELL : (y + 1) * self.CELL, x * self.CELL : (x + 1) * self.CELL, 0] = v
            self.heatmap_img = out

    def _blit_frame(self, arr: np.ndarray):
        # arr is HxWx3
        surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)).copy())
        # clear background
        self.screen.fill((16, 16, 16))
        # draw frame with margin
        self.screen.blit(surf, (self.MARGIN, self.MARGIN))
        # draw status bar
        status = f"t={self.current_step:03d} energy={self.env.energy:02d} steps={self.env.steps:03d}"
        txt = self.font.render(status, True, (200, 200, 200))
        self.screen.blit(txt, (self.MARGIN, self.env.height * self.CELL + self.MARGIN + 4))
        # heatmap overlay
        if self.heatmap_img is not None and self.show_heatmap:
            # convert heatmap_img (HxWx3) to surface and blit with alpha
            hm = pygame.surfarray.make_surface(np.transpose(self.heatmap_img, (1, 0, 2)).copy())
            hm = pygame.transform.scale(hm, (self.env.width * self.CELL, self.env.height * self.CELL))
            hm.set_alpha(150)
            self.screen.blit(hm, (self.MARGIN, self.MARGIN))
        pygame.display.flip()

    def save_gif(self, filename: str, fps: float = 8.0):
        if imageio is None:
            print("imageio not available; install imageio to enable GIF export")
            return
        print(f"Saving GIF to {filename} ...")
        imageio.mimsave(filename, self.frames, fps=fps)
        print("Saved GIF.")

    def run(self):
        running = True
        last_time = time.time()
        acc = 0.0
        self.show_heatmap = False
        if getattr(self.agent, "Q", None):
            self.compute_heatmap()
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_RIGHT:
                        self.step_forward()
                    elif event.key == pygame.K_LEFT:
                        self.step_back()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.fps = min(60.0, self.fps + 1.0)
                    elif event.key == pygame.K_MINUS:
                        self.fps = max(1.0, self.fps - 1.0)
                    elif event.key == pygame.K_h:
                        self.show_heatmap = not self.show_heatmap
                    elif event.key == pygame.K_s:
                        # save gif
                        ts = int(time.time())
                        self.save_gif(f"run_{ts}.gif", fps=self.fps)
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
            if self.playing:
                acc += dt
                frame_dt = 1.0 / max(1.0, self.fps)
                if acc >= frame_dt:
                    self.step_forward()
                    acc = 0.0
            # draw current frame
            arr = self.frames[self.current_step]
            self._blit_frame(arr)
        pygame.quit()

    def step_forward(self):
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            return
        if self.env.done:
            self.playing = False
            return
        action = self.agent.act(self.history[-1], greedy=True)
        res = self.env.step(action)
        self.history.append(res.obs)
        arr = self._render_frame(res.obs)
        self.frames.append(arr)
        self.current_step = len(self.frames) - 1

    def step_back(self):
        if self.current_step > 0:
            self.current_step -= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", help="optional qtable pickle file to load")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--fps", type=float, default=8.0)
    args = ap.parse_args()

    env = GridSurvivalEnv(width=args.w, height=args.h, seed=args.seed)
    agent = QLearningAgent(n_actions=4, qtable=None)
    if args.load:
        try:
            agent = QLearningAgent.load(args.load)
            print(f"Loaded qtable: {args.load} (states={len(agent.Q)})")
        except Exception as e:
            print("Failed to load qtable:", e)
    viewer = PygameViewer(env, agent, fps=args.fps)
    viewer.run()


if __name__ == "__main__":
    main()
