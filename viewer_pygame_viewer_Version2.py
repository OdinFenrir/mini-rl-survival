#!/usr/bin/env python3
"""
Pygame interactive viewer demo with live heatmap overlay and interactive debugging.
Drop this file into viewer/pygame_viewer.py (it replaces the moved file).
"""
import argparse
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
except Exception as e:
    raise SystemExit("pygame is required. Install with: pip install pygame")

try:
    import imageio
except Exception:
    imageio = None


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


def _colormap_viridis_like(v: float) -> Tuple[int, int, int]:
    v = float(np.clip(v, 0.0, 1.0))
    if v <= 0.5:
        t = v / 0.5
        r = int((1 - t) * 68 + t * 32)
        g = int((1 - t) * 1 + t * 144)
        b = int((1 - t) * 84 + t * 140)
    else:
        t = (v - 0.5) / 0.5
        r = int((1 - t) * 32 + t * 253)
        g = int((1 - t) * 144 + t * 231)
        b = int((1 - t) * 140 + t * 37)
    return (r, g, b)


class PygameViewer:
    CELL = 48
    MARGIN = 8

    def __init__(self, env: GridSurvivalEnv, agent: QLearningAgent, fps: float = 8.0, live_heatmap: bool = True, debug_console: bool = False):
        self.env = env
        self.agent = agent
        self.fps = fps
        self.playing = True
        self.frames: List[np.ndarray] = []
        self.heatmap_img: Optional[np.ndarray] = None  # HxWx3 uint8
        self.live_heatmap = live_heatmap
        self.debug_console = debug_console
        self.show_heatmap = live_heatmap  # default True for instant feedback
        self.show_numbers = False
        self.selected_cell = None  # (x,y) if clicked
        self.print_on_step = False

        pygame.init()
        grid_w = env.width
        grid_h = env.height
        window_w = grid_w * self.CELL + self.MARGIN * 2
        window_h = grid_h * self.CELL + self.MARGIN * 2 + 48
        self.screen = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption("Mini RL Survival - Viewer (live heatmap)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 18)
        self.small_font = pygame.font.SysFont(None, 14)

        # Replay buffer & initial frame
        self.history: List[Tuple] = []
        self.reset_episode()

    def reset_episode(self):
        obs = self.env.reset()
        self.history = [obs]
        self.frames = [self._render_frame(obs)]
        self.current_step = 0
        self.playing = True
        self.selected_cell = None
        if self.live_heatmap and getattr(self.agent, "Q", None):
            self.update_heatmap()

    def _render_frame(self, obs) -> np.ndarray:
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
        arr = pygame.surfarray.array3d(surf)
        arr = np.transpose(arr, (1, 0, 2))  # convert to HxWx3
        return arr

    def update_heatmap(self):
        """Compute avg max-Q per (x,y) and produce an HxWx3 uint8 image."""
        if not getattr(self.agent, "Q", None):
            self.heatmap_img = None
            return
        w, h = self.env.width, self.env.height
        heat = np.zeros((h, w), dtype=float)
        counts = np.zeros((h, w), dtype=int)
        for s, qvals in self.agent.Q.items():
            try:
                ax, ay = int(s[0]), int(s[1])
            except Exception:
                continue
            if 0 <= ay < h and 0 <= ax < w:
                heat[ay, ax] += max(qvals) if qvals else 0.0
                counts[ay, ax] += 1
        avg = np.zeros_like(heat)
        nz = counts > 0
        avg[nz] = heat[nz] / counts[nz]
        if not np.any(nz):
            self.heatmap_img = None
            return
        # normalize to [0,1]
        mx = float(np.nanmax(avg))
        norm = np.zeros_like(avg)
        if mx > 0:
            norm = avg / mx
        # produce pixel image sized (h*CELL, w*CELL, 3)
        img = np.zeros((h * self.CELL, w * self.CELL, 3), dtype=np.uint8)
        for yy in range(h):
            for xx in range(w):
                v = float(norm[yy, xx])
                color = _colormap_viridis_like(v)
                img[yy * self.CELL : (yy + 1) * self.CELL, xx * self.CELL : (xx + 1) * self.CELL] = color
        self.heatmap_img = img

    def _blit_frame(self, arr: np.ndarray):
        surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)).copy())
        self.screen.fill((16, 16, 16))
        self.screen.blit(surf, (self.MARGIN, self.MARGIN))
        # overlay heatmap if present
        if self.heatmap_img is not None and self.show_heatmap:
            hm = pygame.surfarray.make_surface(np.transpose(self.heatmap_img, (1, 0, 2)).copy())
            hm = pygame.transform.scale(hm, (self.env.width * self.CELL, self.env.height * self.CELL))
            hm.set_alpha(150)
            self.screen.blit(hm, (self.MARGIN, self.MARGIN))
        # draw grid cell numbers if requested
        if self.show_numbers:
            for y in range(self.env.height):
                for x in range(self.env.width):
                    txt = ""
                    if getattr(self.agent, "Q", None):
                        # compute avg for this pos
                        vals = []
                        for s, qvals in self.agent.Q.items():
                            try:
                                if int(s[0]) == x and int(s[1]) == y:
                                    vals.append(max(qvals) if qvals else 0.0)
                            except Exception:
                                pass
                        if vals:
                            txt = f"{sum(vals)/len(vals):.2f}"
                    if txt:
                        tx = self.small_font.render(txt, True, (0, 0, 0))
                        px = self.MARGIN + x * self.CELL + 4
                        py = self.MARGIN + y * self.CELL + 2
                        # draw small bg for readability
                        rect = pygame.Rect(px - 2, py - 1, tx.get_width() + 4, tx.get_height() + 2)
                        pygame.draw.rect(self.screen, (255, 255, 255, 200), rect)
                        self.screen.blit(tx, (px, py))
        # draw status and controls
        status = f"t={self.current_step:03d} energy={self.env.energy:02d} steps={self.env.steps:03d} fps={self.fps:.1f}"
        txt = self.font.render(status, True, (220, 220, 220))
        self.screen.blit(txt, (self.MARGIN, self.env.height * self.CELL + self.MARGIN + 4))
        ctrl = "Space:Play/Pause  Right:Step  Left:Back  H:Heatmap  N:Nums  D:Debug  Click:SelectCell  S:SaveGIF  Q:Quit"
        txt2 = self.small_font.render(ctrl, True, (160, 160, 160))
        self.screen.blit(txt2, (self.MARGIN, self.env.height * self.CELL + self.MARGIN + 24))
        # selected cell indicator and info
        if self.selected_cell is not None:
            sx, sy = self.selected_cell
            sel_rect = pygame.Rect(self.MARGIN + sx * self.CELL, self.MARGIN + sy * self.CELL, self.CELL, self.CELL)
            pygame.draw.rect(self.screen, (255, 255, 255), sel_rect, 2)
            info = f"Selected: ({sx},{sy})"
            self.screen.blit(self.small_font.render(info, True, (255, 255, 255)), (self.MARGIN + 6, self.MARGIN + 6))
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
        acc = 0.0
        if self.live_heatmap and getattr(self.agent, "Q", None):
            self.update_heatmap()
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    # translate to grid coords
                    gx = (mx - self.MARGIN) // self.CELL
                    gy = (my - self.MARGIN) // self.CELL
                    if 0 <= gx < self.env.width and 0 <= gy < self.env.height:
                        self.selected_cell = (int(gx), int(gy))
                        # print Q-values for selected cell (all states with agent at this pos)
                        if getattr(self.agent, "Q", None):
                            qvals_list = []
                            for s, qvals in self.agent.Q.items():
                                try:
                                    if int(s[0]) == gx and int(s[1]) == gy:
                                        qvals_list.append((s, qvals))
                                except Exception:
                                    pass
                            print(f"== Selected cell ({gx},{gy}) states: {len(qvals_list)} ==")
                            for s, qvs in qvals_list[:20]:
                                print(f"state={s} maxQ={max(qvs) if qvs else 0.0} q={qvs}")
                            if len(qvals_list) > 20:
                                print(f"... ({len(qvals_list)-20} more states)")
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
                    elif event.key == pygame.K_n:
                        self.show_numbers = not self.show_numbers
                    elif event.key == pygame.K_d:
                        self.debug_console = not self.debug_console
                        print("Console debug:", self.debug_console)
                    elif event.key == pygame.K_s:
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
            # live update heatmap each frame if enabled
            if self.live_heatmap and getattr(self.agent, "Q", None):
                self.update_heatmap()
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
        if self.debug_console:
            print(f"Step {self.current_step}: action={action} new_obs={res.obs} reward={res.reward} done={res.done}")

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
    ap.add_argument("--no-heatmap", dest="heatmap", action="store_false", help="disable live heatmap overlay")
    ap.add_argument("--debug", action="store_true", help="enable console debug prints")
    args = ap.parse_args()

    env = GridSurvivalEnv(width=args.w, height=args.h, seed=args.seed)
    agent = QLearningAgent(n_actions=4, qtable=None)
    if args.load:
        try:
            agent = QLearningAgent.load(args.load)
            print(f"Loaded qtable: {args.load} (states={len(agent.Q)})")
        except Exception as e:
            print("Failed to load qtable:", e)
    viewer = PygameViewer(env, agent, fps=args.fps, live_heatmap=args.heatmap, debug_console=args.debug)
    viewer.run()


if __name__ == '__main__':
    main()