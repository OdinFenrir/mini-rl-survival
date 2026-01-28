from __future__ import annotations

import os
import pygame

from core.env import GridSurvivalEnv

from viewers.ui.modals import ConfirmDialog
from viewers.ui.widgets import Button, Slider, Toggle, FocusManager, Label


class SettingsScene:
    def __init__(self) -> None:
        self.focus = FocusManager()

        # Scrolling state
        self.scroll_y = 0
        self.max_scroll = 0
        self._content_top = 0
        self._content_bottom = 0

        # Split into scrollable content + fixed footer
        self.content_widgets: list = []
        self.footer_widgets: list = []
        self.widgets: list = []

    def _layout(self, app) -> None:
        w, h = app.screen.get_size()
        scale = float(app.theme.ui_scale)

        x = int(60 * scale)
        y0 = int(90 * scale)
        ww = w - 2 * x

        font = app.theme.font(int(app.theme.font_size * app.theme.ui_scale))
        label_h = font.get_height()
        row_h = max(int(64 * scale), int(label_h * 2 + 20 * scale))
        gap = int(14 * scale)

        # Footer (fixed buttons)
        footer_h = int(90 * scale)
        footer_y = h - footer_h
        footer_pad = int(14 * scale)

        # Scrollable viewport bounds
        content_top = y0
        content_bottom = footer_y - footer_pad
        viewport_h = max(1, content_bottom - content_top)

        def rect(i: int) -> pygame.Rect:
            return pygame.Rect(x, y0 + i * (row_h + gap), ww, row_h)

        cfg = app.cfg
        items: list = []
        i = 0

        items.append(Label(rect(i), "Environment")); i += 1
        items.append(Slider(rect(i), "Grid width", 4, 40, 1,
                            lambda: float(cfg.w), lambda v: setattr(cfg, "w", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Grid height", 4, 40, 1,
                            lambda: float(cfg.h), lambda v: setattr(cfg, "h", int(v)), fmt="{:.0f}")); i += 1
        def set_energy_start(v: float) -> None:
            v = int(v)
            setattr(cfg, "energy_start", v)
            if int(getattr(cfg, "energy_max", 0)) > 0 and v > int(getattr(cfg, "energy_max", 0)):
                setattr(cfg, "energy_max", v)

        items.append(Slider(rect(i), "Energy start", 1, 120, 1,
                            lambda: float(cfg.energy_start), set_energy_start, fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Energy max (0=unlimited)", 0, 200, 1,
                            lambda: float(getattr(cfg, "energy_max", 0)),
                            lambda v: setattr(cfg, "energy_max", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Energy food gain", 1, 120, 1,
                            lambda: float(cfg.energy_food), lambda v: setattr(cfg, "energy_food", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Energy step cost", 1, 20, 1,
                            lambda: float(cfg.energy_step), lambda v: setattr(cfg, "energy_step", int(v)), fmt="{:.0f}")); i += 1

        level_count = GridSurvivalEnv.preset_level_count()
        level_template = GridSurvivalEnv.get_level_template(int(cfg.level_index) % max(1, level_count)) if level_count else {}
        level_name = level_template.get("name", "Preset level")
        level_desc = level_template.get("desc", "")
        level_source = level_template.get("source", "")
        level_layout = level_template.get("layout", []) or []
        if level_layout:
            level_w = max(len(line) for line in level_layout)
            level_h = len(level_layout)
            walls = sum(line.count("#") for line in level_layout)
            total = max(1, level_w * level_h)
            ratio = walls / total
            if ratio < 0.28:
                level_style = "Open paths"
            elif ratio < 0.42:
                level_style = "Balanced maze"
            else:
                level_style = "Tight corridors"
        else:
            level_w, level_h = cfg.w, cfg.h
            level_style = ""

        items.append(Label(rect(i), "Levels")); i += 1
        items.append(Label(rect(i), f"Preset: {level_name}")); i += 1
        items.append(Label(rect(i), f"Size: {level_w}x{level_h}")); i += 1
        if level_style:
            items.append(Label(rect(i), f"Style: {level_style}")); i += 1
        if level_desc:
            items.append(Label(rect(i), level_desc)); i += 1
        if level_source:
            items.append(Label(rect(i), f"Source: {level_source}")); i += 1

        def cycle_level_mode():
            modes = ["preset", "random"]
            cur = cfg.level_mode if cfg.level_mode in modes else "preset"
            cfg.level_mode = modes[(modes.index(cur) + 1) % len(modes)]
            app.toast.push(f"Level mode: {cfg.level_mode}")

        items.append(Button(rect(i), f"Level mode: {cfg.level_mode} (click to cycle)", cycle_level_mode)); i += 1
        max_level = max(0, level_count - 1)
        items.append(Slider(rect(i), "Level index", 0, max(0, max_level), 1,
                            lambda: float(cfg.level_index),
                            lambda v: setattr(cfg, "level_index", int(v)), fmt="{:.0f}")); i += 1
        items.append(Toggle(rect(i), "Cycle levels",
                            lambda: bool(getattr(cfg, "level_cycle", True)),
                            lambda b: setattr(cfg, "level_cycle", bool(b)))); i += 1
        items.append(Slider(rect(i), "Random walls", 0, 120, 1,
                            lambda: float(getattr(cfg, "n_walls", 18)),
                            lambda v: setattr(cfg, "n_walls", int(v)), fmt="{:.0f}")); i += 1
        items.append(Toggle(rect(i), "Bonus food",
                            lambda: bool(getattr(cfg, "food_enabled", True)),
                            lambda b: setattr(cfg, "food_enabled", bool(b)))); i += 1

        items.append(Label(rect(i), "Agent")); i += 1
        items.append(Slider(rect(i), "Alpha", 0.01, 1.0, 0.01,
                            lambda: float(cfg.alpha), lambda v: setattr(cfg, "alpha", float(v)))); i += 1
        items.append(Slider(rect(i), "Gamma", 0.50, 0.999, 0.001,
                            lambda: float(cfg.gamma), lambda v: setattr(cfg, "gamma", float(v)))); i += 1
        items.append(Slider(rect(i), "Eps start", 0.0, 1.0, 0.01,
                            lambda: float(cfg.eps_start), lambda v: setattr(cfg, "eps_start", float(v)))); i += 1
        items.append(Slider(rect(i), "Eps end", 0.0, 1.0, 0.01,
                            lambda: float(cfg.eps_end), lambda v: setattr(cfg, "eps_end", float(v)))); i += 1
        items.append(Slider(rect(i), "Eps decay steps", 100, 300000, 100,
                            lambda: float(cfg.eps_decay), lambda v: setattr(cfg, "eps_decay", int(v)), fmt="{:.0f}")); i += 1

        items.append(Label(rect(i), "View")); i += 1
        items.append(Slider(rect(i), "Render FPS", 10, 240, 1,
                            lambda: float(cfg.render_fps), lambda v: setattr(cfg, "render_fps", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Sim steps/frame", 1, 60, 1,
                            lambda: float(cfg.sim_steps_per_frame), lambda v: setattr(cfg, "sim_steps_per_frame", int(v)), fmt="{:.0f}")); i += 1
        items.append(Slider(rect(i), "Font scale", 0.8, 1.8, 0.05,
                            lambda: float(cfg.font_scale), lambda v: setattr(cfg, "font_scale", float(v)))); i += 1
        items.append(Toggle(rect(i), "Sound effects",
                            lambda: bool(getattr(cfg, "sound_enabled", True)),
                            lambda b: setattr(cfg, "sound_enabled", bool(b)))); i += 1
        items.append(Toggle(rect(i), "Reduced motion",
                            lambda: bool(cfg.reduced_motion), lambda b: setattr(cfg, "reduced_motion", bool(b)))); i += 1

        def cycle_color():
            modes = ["neo", "pixel", "default", "colorblind", "high_contrast"]
            cur = cfg.color_mode if cfg.color_mode in modes else "default"
            cfg.color_mode = modes[(modes.index(cur) + 1) % len(modes)]
            app.apply_theme_from_config()
            app.toast.push(f"Theme: {cfg.color_mode}")

        items.append(Button(rect(i), f"Theme preset: {cfg.color_mode} (click to cycle)", cycle_color)); i += 1
        items.append(Slider(rect(i), "Heatmap opacity", 0.1, 1.0, 0.05,
                            lambda: float(getattr(cfg, "heatmap_opacity", 0.7)),
                            lambda v: setattr(cfg, "heatmap_opacity", float(v)), fmt="{:.2f}")); i += 1

        items.append(Label(rect(i), "Data")); i += 1

        def clear_run_history() -> None:
            path = os.path.join("data", "run_history.jsonl")
            try:
                if os.path.exists(path):
                    os.remove(path)
                app.toast.push("Run history cleared")
            except Exception as exc:
                app.toast.push(f"Clear failed: {exc}")

        def confirm_clear_run_history() -> None:
            rect = pygame.Rect(0, 0, int(420 * scale), int(200 * scale))
            rect.center = app.screen.get_rect().center
            modal = ConfirmDialog(
                rect,
                "Clear run history",
                "This will delete data/run_history.jsonl",
                on_confirm=clear_run_history,
                on_cancel=lambda: None,
            )
            app.push_modal(modal)

        items.append(Button(rect(i), "Clear run history", confirm_clear_run_history)); i += 1

        # Footer buttons (fixed)
        bw = int(220 * scale)
        bh = int(56 * scale)
        gap2 = int(14 * scale)
        bx = (w - (3 * bw + 2 * gap2)) // 2
        by = footer_y + (footer_h - bh) // 2

        def apply():
            prev_scale = float(app.theme.ui_scale)
            prev_mode = str(app.theme.ui_style)
            app.apply_theme_from_config()
            app.toast.push("Settings applied")
            if float(app.theme.ui_scale) != prev_scale or str(app.theme.ui_style) != prev_mode:
                self._layout(app)

        def defaults():
            from viewers.app import AppConfig
            app.cfg = AppConfig()
            app.apply_theme_from_config()
            app.toast.push("Restored defaults")
            self._layout(app)  # rebuild with defaults

        footer = [
            Button(pygame.Rect(bx + 0 * (bw + gap2), by, bw, bh), "Apply", apply),
            Button(pygame.Rect(bx + 1 * (bw + gap2), by, bw, bh), "Defaults", defaults),
            Button(pygame.Rect(bx + 2 * (bw + gap2), by, bw, bh), "Back", lambda: app.pop()),
        ]

        # Store + compute scroll bounds
        self.content_widgets = items
        self.footer_widgets = footer
        self.widgets = self.content_widgets + self.footer_widgets
        self.focus.set(self.widgets)

        self._content_top = content_top
        self._content_bottom = content_bottom

        if self.content_widgets:
            content_end = self.content_widgets[-1].rect.bottom
        else:
            content_end = content_top

        content_h = max(0, content_end - content_top)
        self.max_scroll = max(0, content_h - viewport_h)

        # Clamp scroll after resize/layout
        self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))

    def _scroll_by(self, dy: int) -> None:
        self.scroll_y = max(0, min(self.scroll_y + dy, self.max_scroll))

    def _scroll_to_widget(self, w) -> None:
        """Ensure focused widget is visible inside the scroll viewport."""
        if w is None or w in self.footer_widgets:
            return

        top = self._content_top
        bottom = self._content_bottom
        view_h = max(1, bottom - top)

        # widget rect is in "content space"; draw position is rect.y - scroll_y
        wy_top = w.rect.top - self.scroll_y
        wy_bottom = w.rect.bottom - self.scroll_y

        if wy_top < top:
            # scroll up
            self.scroll_y = max(0, w.rect.top - top)
        elif wy_bottom > bottom:
            # scroll down
            self.scroll_y = min(self.max_scroll, w.rect.bottom - bottom)

    def _event_pos_to_content_space(self, event: pygame.event.Event) -> pygame.event.Event:
        """Adjust mouse events so widgets with content-space rects get correct hit-testing."""
        if not hasattr(event, "pos"):
            return event

        x, y = event.pos
        # Only translate inside scrollable content area
        if self._content_top <= y <= self._content_bottom:
            y = y + self.scroll_y
            # Create a shallow copy-like event with updated pos
            # pygame events are not always mutable, so we recreate the dict
            data = event.dict.copy()
            data["pos"] = (x, y)
            return pygame.event.Event(event.type, data)
        return event

    def handle_event(self, app, event: pygame.event.Event) -> None:
        if not self.widgets:
            self._layout(app)

        scale = float(app.theme.ui_scale)
        wheel_step = int(60 * scale)

        # Scrolling inputs
        if event.type == pygame.MOUSEWHEEL:
            # wheel.y: +1 up, -1 down (usually)
            self._scroll_by(int(-event.y * wheel_step))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # old wheel events: 4 up, 5 down
            if event.button == 4:
                self._scroll_by(-wheel_step)
            elif event.button == 5:
                self._scroll_by(+wheel_step)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                app.pop()
                return

            # Focus navigation
            if event.key in (pygame.K_TAB, pygame.K_DOWN):
                self.focus.next()
                self._scroll_to_widget(self.focus.focused())
            elif event.key == pygame.K_UP:
                self.focus.prev()
                self._scroll_to_widget(self.focus.focused())

            # Keyboard scrolling
            elif event.key == pygame.K_PAGEDOWN:
                self._scroll_by(+int((self._content_bottom - self._content_top) * 0.85))
            elif event.key == pygame.K_PAGEUP:
                self._scroll_by(-int((self._content_bottom - self._content_top) * 0.85))
            elif event.key == pygame.K_HOME:
                self.scroll_y = 0
            elif event.key == pygame.K_END:
                self.scroll_y = self.max_scroll

        # Translate mouse event coordinates into content space before passing to widgets
        ev = self._event_pos_to_content_space(event)

        focused = self.focus.focused()
        for w in self.widgets:
            # For footer widgets, use their fixed rects: don't translate
            use_event = event if w in self.footer_widgets else ev

            # If the widget is scrollable, ignore interaction when it's outside viewport
            if w in self.content_widgets:
                draw_rect = w.rect.move(0, -self.scroll_y)
                if draw_rect.bottom < self._content_top or draw_rect.top > self._content_bottom:
                    continue

            w.handle_event(use_event, focused=(w is focused))

    def update(self, app, dt: float) -> None:
        pass

    def render(self, app, screen: pygame.Surface) -> None:
        if not self.widgets:
            self._layout(app)

        screen.fill(app.theme.palette.bg)

        title_font = app.theme.font(int(app.theme.font_size_title * 0.55 * app.theme.ui_scale))
        title = title_font.render("Settings", True, app.theme.palette.fg)
        screen.blit(title, (int(60 * app.theme.ui_scale), int(30 * app.theme.ui_scale)))

        # Draw scroll area clip
        clip_rect = pygame.Rect(0, self._content_top, screen.get_width(), max(1, self._content_bottom - self._content_top))
        old_clip = screen.get_clip()
        screen.set_clip(clip_rect)

        focused = self.focus.focused()

        # Draw content widgets with scroll offset
        for w in self.content_widgets:
            shifted = w.rect.move(0, -self.scroll_y)
            if shifted.bottom < self._content_top or shifted.top > self._content_bottom:
                continue

            # Temporarily draw with shifted rect without permanently mutating layout
            old = w.rect
            w.rect = shifted
            try:
                w.draw(screen, app.theme, focused=(w is focused))
            finally:
                w.rect = old

        screen.set_clip(old_clip)

        # Simple scrollbar (right side)
        if self.max_scroll > 0:
            vw = clip_rect.width
            vh = clip_rect.height
            bar_w = int(10 * app.theme.ui_scale)
            track = pygame.Rect(vw - bar_w - 6, clip_rect.y + 6, bar_w, vh - 12)
            pygame.draw.rect(screen, app.theme.palette.grid_line, track, 1, border_radius=6)

            thumb_h = max(int(30 * app.theme.ui_scale), int(track.h * (vh / (vh + self.max_scroll))))
            t = 0.0 if self.max_scroll == 0 else (self.scroll_y / self.max_scroll)
            thumb_y = int(track.y + t * (track.h - thumb_h))
            thumb = pygame.Rect(track.x + 1, thumb_y, track.w - 2, thumb_h)
            pygame.draw.rect(screen, app.theme.palette.muted, thumb, 0, border_radius=6)

        # Draw footer (fixed buttons)
        focused = self.focus.focused()
        for w in self.footer_widgets:
            w.draw(screen, app.theme, focused=(w is focused))
        # Hint (placed under title so it never overlaps footer buttons)
        font = app.theme.font(int(app.theme.font_size * app.theme.ui_scale))
        hint = font.render("Scroll: mouse wheel / PgUp-PgDn | Tab to navigate | Esc to go back", True, app.theme.palette.muted)
        hint_y = int(30 * app.theme.ui_scale) + title.get_height() + int(10 * app.theme.ui_scale)
        screen.blit(hint, (int(60 * app.theme.ui_scale), hint_y))
