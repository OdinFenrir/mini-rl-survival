import importlib
import os
from typing import Callable, Optional

import pygame
from .widgets import Button, Slider, Toggle, FocusManager

class Modal:
    def __init__(self, rect: pygame.Rect, title: str, on_close: Optional[Callable]=None):
        self.rect = rect
        self.title = title
        self.on_close = on_close
        self.active = True

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.close()

    def render(self, screen: pygame.Surface, theme):
        # Dim background so the modal stands out and the user sees it's blocking input.
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        pygame.draw.rect(screen, theme.palette.panel, self.rect, border_radius=8)
        font = theme.font(int(theme.font_size * theme.ui_scale))
        title_surf = font.render(self.title, True, theme.palette.fg)
        screen.blit(title_surf, (self.rect.x + 16, self.rect.y + 16))

    def close(self):
        self.active = False
        if self.on_close:
            self.on_close()

class FileDialogModal(Modal):
    def __init__(
        self,
        rect,
        title,
        on_confirm,
        on_cancel,
        initial_path: str = "",
        recent_files=None,
        must_exist: bool = False,
        ext_filter: str | None = None,
    ):
        super().__init__(rect, title, on_close=on_cancel)
        self.on_confirm = on_confirm
        self.path = initial_path
        self.recent_files = recent_files or []
        self.must_exist = bool(must_exist)
        self.ext_filter = (ext_filter or os.path.splitext(initial_path)[1] or "").lower()
        self.browse_dir = os.path.dirname(initial_path) or "data"
        self.input_active = True
        self.error = ""
        self.files: list[str] = []
        self.selected_index = 0
        self._layout_cache: dict[str, pygame.Rect | int | float] = {}
        self._last_scale = 1.0
        self._refresh_files()

    def _layout(self, scale: float) -> dict[str, pygame.Rect | int | float]:
        scale = float(scale or 1.0)
        pad = int(16 * scale)
        input_h = int(36 * scale)
        button_w = int(96 * scale)
        button_h = int(36 * scale)
        button_gap = int(12 * scale)
        list_gap = int(12 * scale)
        header_h = int(24 * scale)
        row_h = int(24 * scale)

        input_rect = pygame.Rect(self.rect.x + pad, self.rect.y + pad + int(24 * scale), self.rect.width - 2 * pad, input_h)
        button_y = self.rect.bottom - pad - button_h
        ok_rect = pygame.Rect(self.rect.right - pad - (2 * button_w + button_gap), button_y, button_w, button_h)
        cancel_rect = pygame.Rect(self.rect.right - pad - button_w, button_y, button_w, button_h)

        list_top = input_rect.bottom + list_gap
        list_bottom = button_y - list_gap
        list_h = max(0, list_bottom - list_top)
        list_rect = pygame.Rect(self.rect.x + pad, list_top, self.rect.width - 2 * pad, list_h)

        layout = {
            "pad": pad,
            "input_rect": input_rect,
            "list_rect": list_rect,
            "ok_rect": ok_rect,
            "cancel_rect": cancel_rect,
            "header_h": header_h,
            "row_h": row_h,
        }
        self._layout_cache = layout
        return layout

    def _refresh_files(self) -> None:
        self.files = []
        try:
            if not os.path.isdir(self.browse_dir):
                return
            for name in os.listdir(self.browse_dir):
                full = os.path.join(self.browse_dir, name)
                if not os.path.isfile(full):
                    continue
                if self.ext_filter and os.path.splitext(name)[1].lower() != self.ext_filter:
                    continue
                self.files.append(name)
            self.files.sort(key=lambda n: os.path.getmtime(os.path.join(self.browse_dir, n)), reverse=True)
        except Exception:
            self.files = []
        self.selected_index = max(0, min(self.selected_index, max(0, len(self.files) - 1)))

    def _normalized_path(self) -> str:
        p = (self.path or "").strip()
        if len(p) >= 2 and ((p[0] == '"' and p[-1] == '"') or (p[0] == "'" and p[-1] == "'")):
            p = p[1:-1].strip()
        return p

    @staticmethod
    def _ellipsize_left(text: str, font: pygame.font.Font, max_w: int) -> str:
        if max_w <= 0 or not text:
            return ""
        if font.size(text)[0] <= max_w:
            return text
        ell = "..."
        ell_w = font.size(ell)[0]
        if ell_w >= max_w:
            return ell
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            if font.size(text[mid:])[0] + ell_w <= max_w:
                hi = mid
            else:
                lo = mid + 1
        return ell + text[lo:]

    def handle_event(self, event):
        layout = self._layout_cache or self._layout(self._last_scale)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            input_rect = layout["input_rect"]
            list_rect = layout["list_rect"]
            ok_rect = layout["ok_rect"]
            cancel_rect = layout["cancel_rect"]
            if ok_rect.collidepoint(mx, my):
                self._confirm()
                return
            if cancel_rect.collidepoint(mx, my):
                self.close()
                return
            if input_rect.collidepoint(mx, my):
                self.input_active = True
                return
            if list_rect.collidepoint(mx, my) and self.files:
                row_h = int(layout["row_h"])
                header_h = int(layout["header_h"])
                items_y0 = list_rect.y + header_h + int(6 * (layout["row_h"] / 24))
                if my < items_y0:
                    return
                idx = (my - items_y0) // row_h
                max_rows = max(0, (list_rect.h - (items_y0 - list_rect.y) - 4) // row_h)
                if 0 <= idx < min(len(self.files), max_rows):
                    self.selected_index = int(idx)
                    self.path = os.path.join(self.browse_dir, self.files[self.selected_index])
                    return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.close()
                return

            if event.key == pygame.K_RETURN:
                self._confirm()
                return

            if event.key == pygame.K_UP and self.files:
                self.selected_index = max(0, self.selected_index - 1)
                self.path = os.path.join(self.browse_dir, self.files[self.selected_index])
                return

            if event.key == pygame.K_DOWN and self.files:
                self.selected_index = min(len(self.files) - 1, self.selected_index + 1)
                self.path = os.path.join(self.browse_dir, self.files[self.selected_index])
                return

            if event.key == pygame.K_v and (event.mod & pygame.KMOD_CTRL):
                # Best-effort clipboard paste; silently ignore if unavailable.
                try:
                    scrap = importlib.import_module("pygame.scrap")
                    scrap.init()
                    clip = scrap.get(pygame.SCRAP_TEXT) or scrap.get("text/plain;charset=utf-8")
                    if clip:
                        if isinstance(clip, bytes):
                            clip = clip.decode("utf-8", errors="ignore")
                        self.path += str(clip).strip()
                except Exception:
                    pass
                return

            if event.key == pygame.K_BACKSPACE:
                self.path = self.path[:-1]
                return

            if self.input_active and event.unicode and event.unicode.isprintable():
                self.path += event.unicode
                return

    def _confirm(self) -> None:
        p = self._normalized_path()
        if not p:
            self.error = "Path required"
            return
        if self.must_exist and not os.path.exists(p):
            self.error = "File does not exist"
            return
        try:
            self.on_confirm(p)
            self.close()
        except Exception as exc:
            self.error = str(exc)

    def render(self, screen, theme):
        self._last_scale = float(theme.ui_scale or 1.0)
        layout = self._layout(self._last_scale)
        super().render(screen, theme)
        font = theme.font(int(theme.font_size * theme.ui_scale))
        input_rect = layout["input_rect"]
        pygame.draw.rect(screen, theme.palette.bg, input_rect, border_radius=6)
        pygame.draw.rect(screen, theme.palette.fg, input_rect, 2, border_radius=6)
        display_path = self._ellipsize_left(self.path, font, input_rect.w - 16)
        input_surf = font.render(display_path, True, theme.palette.fg)
        screen.blit(input_surf, (input_rect.x + 8, input_rect.y + 6))
        if self.error:
            err_surf = font.render(self.error, True, theme.palette.danger)
            screen.blit(err_surf, (input_rect.x, input_rect.y + 40))

        # File list (auto-populated from browse_dir)
        list_rect = layout["list_rect"]
        pygame.draw.rect(screen, theme.palette.grid0, list_rect, border_radius=6)
        pygame.draw.rect(screen, theme.palette.grid_line, list_rect, 1, border_radius=6)

        small = theme.font(max(14, int(theme.font_size * 0.8 * theme.ui_scale)))
        header = f"{self.browse_dir}  ({self.ext_filter or '*'} files)"
        header_surf = small.render(header, True, theme.palette.muted)
        screen.blit(header_surf, (list_rect.x + 8, list_rect.y + 6))

        y = list_rect.y + int(layout["header_h"]) + int(6 * (layout["row_h"] / 24))
        row_h = int(layout["row_h"])
        max_rows = max(0, (list_rect.h - (y - list_rect.y) - 4) // row_h)
        for i, name in enumerate(self.files[:max_rows]):
            row_rect = pygame.Rect(list_rect.x + 6, y, list_rect.w - 12, row_h - 2)
            if i == self.selected_index:
                pygame.draw.rect(screen, theme.palette.grid1, row_rect, border_radius=4)
            f_surf = small.render(name, True, theme.palette.fg if i == self.selected_index else theme.palette.muted)
            screen.blit(f_surf, (row_rect.x + 6, row_rect.y + 3))
            y += row_h

        # Buttons (draw last so they stay above the list)
        ok_rect = layout["ok_rect"]
        cancel_rect = layout["cancel_rect"]
        pygame.draw.rect(screen, theme.palette.accent, ok_rect, border_radius=6)
        pygame.draw.rect(screen, theme.palette.grid0, cancel_rect, border_radius=6)
        ok_surf = font.render("OK", True, theme.palette.fg)
        cancel_surf = font.render("Cancel", True, theme.palette.fg)
        screen.blit(ok_surf, (ok_rect.centerx - ok_surf.get_width() // 2, ok_rect.centery - ok_surf.get_height() // 2))
        screen.blit(cancel_surf, (cancel_rect.centerx - cancel_surf.get_width() // 2, cancel_rect.centery - cancel_surf.get_height() // 2))

class ConfirmDialog(Modal):
    def __init__(self, rect, title, message, on_confirm, on_cancel):
        super().__init__(rect, title, on_close=on_cancel)
        self.message = message
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.focus = 0  # 0: confirm, 1: cancel

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            btn_w = 100
            btn_h = 36
            gap = 24
            x0 = self.rect.x + 16
            y0 = self.rect.y + self.rect.height - btn_h - 16
            confirm_rect = pygame.Rect(x0, y0, btn_w, btn_h)
            cancel_rect = pygame.Rect(x0 + btn_w + gap, y0, btn_w, btn_h)
            if confirm_rect.collidepoint(mx, my):
                self.on_confirm()
                self.close()
                return
            if cancel_rect.collidepoint(mx, my):
                self.close()
                return
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_TAB):
                self.focus = 1 - self.focus
            elif event.key == pygame.K_RETURN:
                if self.focus == 0:
                    self.on_confirm()
                self.close()
            elif event.key == pygame.K_ESCAPE:
                self.close()

    def render(self, screen, theme):
        super().render(screen, theme)
        font = theme.font(int(theme.font_size * theme.ui_scale))
        msg_surf = font.render(self.message, True, theme.palette.fg)
        screen.blit(msg_surf, (self.rect.x + 16, self.rect.y + 56))
        # Buttons
        btn_w = 100
        btn_h = 36
        gap = 24
        x0 = self.rect.x + 16
        y0 = self.rect.y + self.rect.height - btn_h - 16
        for i, label in enumerate(["Confirm", "Cancel"]):
            btn_rect = pygame.Rect(x0 + i * (btn_w + gap), y0, btn_w, btn_h)
            color = theme.palette.accent if self.focus == i else theme.palette.panel
            pygame.draw.rect(screen, color, btn_rect, border_radius=6)
            lbl_surf = font.render(label, True, theme.palette.fg)
            screen.blit(lbl_surf, (btn_rect.x + 16, btn_rect.y + 8))


class TrainingSetupWizard(Modal):
    def __init__(self, rect, cfg, on_apply, on_cancel):
        super().__init__(rect, "New Q-table (guided)", on_close=on_cancel)
        self.cfg = cfg
        self.on_apply = on_apply
        self._step = 0
        self._steps = ["intro", "env", "train", "curriculum", "summary"]
        self._widgets: list = []
        self._lines: list[str] = []
        self._focus = FocusManager()
        self._dirty = True
        self._last_scale = 1.0
        self._show_warnings = False
        self._preset_mode = "advanced"

    def _lines_for_step(self) -> list[str]:
        if self._steps[self._step] == "intro":
            return [
                "This wizard helps you create a fresh Q-table.",
                "We will configure environment + training settings step by step.",
                "Use Tab/Up/Down to navigate. Esc cancels.",
            ]
        if self._steps[self._step] == "env":
            return [
                "Environment settings: make the maze and rewards.",
                "Tip: Easy = closer key/door, Hard = farther spacing.",
            ]
        if self._steps[self._step] == "train":
            return [
                "Training settings: how long to train and how often to evaluate.",
            ]
        if self._steps[self._step] == "curriculum":
            return [
                "Curriculum gradually unlocks more maps as the agent improves.",
                "Turn it ON to train across all preset maps faster.",
            ]
        energy_max = getattr(self.cfg, "energy_max", 0)
        energy_max_label = "∞" if not energy_max else str(int(energy_max))
        return [
            "Summary: these settings will be applied.",
            f"Grid: {int(self.cfg.w)}x{int(self.cfg.h)}  Walls: {int(getattr(self.cfg, 'n_walls', 0))}",
            f"Energy: start {int(self.cfg.energy_start)}  max {energy_max_label}  food +{int(self.cfg.energy_food)}",
            f"Train: {int(self.cfg.train_episodes)} eps  max steps {int(self.cfg.train_max_steps)}  eval {int(self.cfg.train_eval_every)}",
            f"Curriculum: {'ON' if bool(getattr(self.cfg, 'train_curriculum', False)) else 'OFF'}  Difficulty: {getattr(self.cfg, 'placement_difficulty', 'medium')}",
            "Click Create to reset training and start with a fresh Q-table.",
        ]

    def _warnings(self) -> list[str]:
        warnings: list[str] = []
        energy_max = int(getattr(self.cfg, "energy_max", 0) or 0)
        if energy_max and energy_max < int(self.cfg.energy_start):
            warnings.append("Energy max is lower than energy start; it will clamp.")
        if int(self.cfg.energy_food) < int(self.cfg.energy_step):
            warnings.append("Food gain is lower than step cost; food may feel weak.")
        min_energy_hint = max(6, int((int(self.cfg.w) + int(self.cfg.h)) * 0.5))
        if int(self.cfg.energy_start) < min_energy_hint:
            warnings.append("Energy start is low for this maze size; may time out.")
        if int(getattr(self.cfg, "train_episodes", 0)) < 500:
            warnings.append("Very few episodes; learning may look random.")
        return warnings

    def _apply_beginner_preset(self) -> None:
        self._preset_mode = "beginner"
        self.cfg.train_curriculum = True
        self.cfg.train_curriculum_start = 5
        self.cfg.train_curriculum_step = 5
        self.cfg.train_curriculum_window = 50
        self.cfg.train_curriculum_threshold = 0.8
        self.cfg.train_curriculum_eps_rewind = 0.5
        self.cfg.energy_start = 60
        self.cfg.energy_max = 80
        self.cfg.energy_food = 30
        self.cfg.train_eval_every = 0
        self.cfg.train_speed = 20
        self.cfg.train_episodes = 20000
        self.cfg.placement_difficulty = "medium"
        self._dirty = True

    def _apply_advanced_preset(self) -> None:
        self._preset_mode = "advanced"
        self._dirty = True

    def _layout(self, theme) -> None:
        scale = float(theme.ui_scale or 1.0)
        if not self._dirty and abs(scale - self._last_scale) < 1e-3:
            return
        self._last_scale = scale
        self._dirty = False
        pad = int(18 * scale)
        content_left = self.rect.x + pad
        content_top = self.rect.y + int(56 * scale)
        content_w = self.rect.w - 2 * pad
        row_h = int(46 * scale)
        gap = int(10 * scale)
        font = theme.font(int(theme.font_size * theme.ui_scale))
        line_h = font.get_height()

        self._lines = self._lines_for_step()
        text_h = len(self._lines) * (line_h + int(4 * scale)) + int(8 * scale)
        y0 = content_top + text_h

        def rect_row(i: int) -> pygame.Rect:
            return pygame.Rect(content_left, y0 + i * (row_h + gap), content_w, row_h)

        self._widgets = []
        i = 0
        step = self._steps[self._step]

        if step == "intro":
            self._widgets.append(Button(rect_row(i), "Beginner preset (recommended)", self._apply_beginner_preset)); i += 1
            self._widgets.append(Button(rect_row(i), "Advanced (manual settings)", self._apply_advanced_preset)); i += 1

        if step == "env":
            def cycle_diff() -> None:
                modes = ["easy", "medium", "hard"]
                cur = str(getattr(self.cfg, "placement_difficulty", "medium")).lower()
                if cur not in modes:
                    cur = "medium"
                self.cfg.placement_difficulty = modes[(modes.index(cur) + 1) % len(modes)]
                self._dirty = True

            self._widgets.append(Button(rect_row(i), f"Difficulty: {getattr(self.cfg, 'placement_difficulty', 'medium')}", cycle_diff)); i += 1
            self._widgets.append(Slider(rect_row(i), "Grid width", 4, 40, 1,
                                        lambda: float(self.cfg.w), lambda v: setattr(self.cfg, "w", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Grid height", 4, 40, 1,
                                        lambda: float(self.cfg.h), lambda v: setattr(self.cfg, "h", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Random walls", 0, 120, 1,
                                        lambda: float(getattr(self.cfg, "n_walls", 18)),
                                        lambda v: setattr(self.cfg, "n_walls", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Toggle(rect_row(i), "Bonus food",
                                        lambda: bool(getattr(self.cfg, "food_enabled", True)),
                                        lambda b: setattr(self.cfg, "food_enabled", bool(b)))); i += 1
            self._widgets.append(Slider(rect_row(i), "Energy start", 1, 120, 1,
                                        lambda: float(self.cfg.energy_start), lambda v: setattr(self.cfg, "energy_start", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Energy max (0=unlimited)", 0, 200, 1,
                                        lambda: float(getattr(self.cfg, "energy_max", 0)),
                                        lambda v: setattr(self.cfg, "energy_max", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Energy food gain", 1, 120, 1,
                                        lambda: float(self.cfg.energy_food), lambda v: setattr(self.cfg, "energy_food", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Energy step cost", 1, 20, 1,
                                        lambda: float(self.cfg.energy_step), lambda v: setattr(self.cfg, "energy_step", int(v)), fmt="{:.0f}")); i += 1

        if step == "train":
            self._widgets.append(Slider(rect_row(i), "Episodes", 200, 50000, 200,
                                        lambda: float(self.cfg.train_episodes), lambda v: setattr(self.cfg, "train_episodes", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Max steps/ep", 50, 800, 10,
                                        lambda: float(self.cfg.train_max_steps), lambda v: setattr(self.cfg, "train_max_steps", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Eval every", 0, 2000, 50,
                                        lambda: float(self.cfg.train_eval_every), lambda v: setattr(self.cfg, "train_eval_every", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Eval episodes", 10, 200, 10,
                                        lambda: float(self.cfg.train_eval_episodes), lambda v: setattr(self.cfg, "train_eval_episodes", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Speed (eps/update)", 1, 50, 1,
                                        lambda: float(self.cfg.train_speed), lambda v: setattr(self.cfg, "train_speed", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Toggle(rect_row(i), "Autosave on finish",
                                        lambda: bool(getattr(self.cfg, "train_autosave", True)),
                                        lambda b: setattr(self.cfg, "train_autosave", bool(b)))); i += 1

        if step == "curriculum":
            self._widgets.append(Toggle(rect_row(i), "Curriculum mode",
                                        lambda: bool(getattr(self.cfg, "train_curriculum", False)),
                                        lambda b: setattr(self.cfg, "train_curriculum", bool(b)))); i += 1
            total_levels = max(1, getattr(importlib.import_module("core.env"), "GridSurvivalEnv").preset_level_count())
            self._widgets.append(Slider(rect_row(i), "Start levels", 1, total_levels, 1,
                                        lambda: float(getattr(self.cfg, "train_curriculum_start", 5)),
                                        lambda v: setattr(self.cfg, "train_curriculum_start", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Add levels", 1, total_levels, 1,
                                        lambda: float(getattr(self.cfg, "train_curriculum_step", 5)),
                                        lambda v: setattr(self.cfg, "train_curriculum_step", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Success threshold", 0.5, 1.0, 0.05,
                                        lambda: float(getattr(self.cfg, "train_curriculum_threshold", 0.8)),
                                        lambda v: setattr(self.cfg, "train_curriculum_threshold", float(v)), fmt="{:.2f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Window (episodes)", 10, 200, 10,
                                        lambda: float(getattr(self.cfg, "train_curriculum_window", 50)),
                                        lambda v: setattr(self.cfg, "train_curriculum_window", int(v)), fmt="{:.0f}")); i += 1
            self._widgets.append(Slider(rect_row(i), "Eps rewind", 0.0, 1.0, 0.05,
                                        lambda: float(getattr(self.cfg, "train_curriculum_eps_rewind", 0.5)),
                                        lambda v: setattr(self.cfg, "train_curriculum_eps_rewind", float(v)), fmt="{:.2f}")); i += 1

        btn_h = int(38 * scale)
        btn_w = int(140 * scale)
        btn_gap = int(12 * scale)
        btn_y = self.rect.bottom - pad - btn_h
        cancel_btn = Button(pygame.Rect(self.rect.x + pad, btn_y, btn_w, btn_h), "Cancel", self.close)
        back_btn = Button(pygame.Rect(self.rect.right - pad - (2 * btn_w + btn_gap), btn_y, btn_w, btn_h), "Back", self._back)
        next_label = "Create Q-table" if self._step == len(self._steps) - 1 else "Next"
        next_btn = Button(pygame.Rect(self.rect.right - pad - btn_w, btn_y, btn_w, btn_h), next_label, self._next_or_apply)
        back_btn.enabled = self._step > 0
        self._widgets += [cancel_btn, back_btn, next_btn]
        self._focus.set(self._widgets)

    def _back(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._dirty = True

    def _next_or_apply(self) -> None:
        self._show_warnings = True
        if self._step < len(self._steps) - 1:
            self._step += 1
            self._dirty = True
        else:
            self.on_apply()
            self.close()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.close()
            return
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_TAB, pygame.K_DOWN):
                self._focus.next()
            elif event.key == pygame.K_UP:
                self._focus.prev()
        focused = self._focus.focused()
        for w in self._widgets:
            w.handle_event(event, focused=(w is focused))

    def render(self, screen, theme):
        self._layout(theme)
        super().render(screen, theme)
        small = theme.font(max(14, int(theme.font_size * 0.85 * theme.ui_scale)))
        pad = int(18 * theme.ui_scale)
        x = self.rect.x + pad
        y = self.rect.y + int(56 * theme.ui_scale)
        for line in self._lines:
            surf = small.render(line, True, theme.palette.muted)
            screen.blit(surf, (x, y))
            y += surf.get_height() + int(4 * theme.ui_scale)

        warnings = self._warnings() if self._show_warnings else []
        if warnings:
            wy = self.rect.bottom - int(86 * theme.ui_scale)
            for warn in warnings[:2]:
                warn_surf = small.render(f"Warning: {warn}", True, theme.palette.warn)
                screen.blit(warn_surf, (x, wy))
                wy += warn_surf.get_height() + int(4 * theme.ui_scale)

        focused = self._focus.focused()
        for w in self._widgets:
            w.draw(screen, theme, focused=(w is focused))

