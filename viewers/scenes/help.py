from __future__ import annotations

import pygame

from viewers.keymap import get_keymap


class HelpScene:
    def __init__(self) -> None:
        self.scroll_y = 0
        self.max_scroll = 0

    def handle_event(self, app, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
            app.pop()
            return
        scale = float(app.theme.ui_scale)
        wheel_step = int(60 * scale)
        if event.type == pygame.MOUSEWHEEL:
            self.scroll_y = max(0, min(self.max_scroll, self.scroll_y - int(event.y * wheel_step)))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                self.scroll_y = max(0, self.scroll_y - wheel_step)
            elif event.button == 5:
                self.scroll_y = min(self.max_scroll, self.scroll_y + wheel_step)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PAGEDOWN:
                self.scroll_y = min(self.max_scroll, self.scroll_y + int(360 * scale))
            elif event.key == pygame.K_PAGEUP:
                self.scroll_y = max(0, self.scroll_y - int(360 * scale))
            elif event.key == pygame.K_HOME:
                self.scroll_y = 0
            elif event.key == pygame.K_END:
                self.scroll_y = self.max_scroll

    def update(self, app, dt: float) -> None:
        pass

    def render(self, app, screen: pygame.Surface) -> None:
        screen.fill(app.theme.palette.bg)
        title_font = app.theme.font(int(app.theme.font_size_title * 0.65 * app.theme.ui_scale))
        font = app.theme.font(int(app.theme.font_size * app.theme.ui_scale))
        small = app.theme.font(int(app.theme.font_size * 0.9 * app.theme.ui_scale))
        sections: list[tuple[str, list[str]]] = [
            ("Goal", ["Collect the food first, then the exit (goal) unlocks."]),
            ("Quick start", [
                "1) Start / Continue: explore the environment with the current agent.",
                "2) Load Q-table & Start: watch a trained agent play immediately.",
                "3) Use Settings to tweak grid size and learning params.",
            ]),
            ("Levels", [
                "Preset mode cycles through curated levels with clear goals.",
                "Random mode generates new walls each reset for variety.",
            ]),
            ("What you are seeing", [
                "Agent (blue) moves on the grid; food (green) adds energy.",
                "Goal (gold) appears after you collect food.",
                "Walls (dark) block movement.",
                "Heatmap (H): higher values indicate better long-term reward.",
                "Policy arrows (P): best action per cell from the Q-table.",
                "Q-hover (Q): shows per-action Q-values for the hovered cell.",
            ]),
            ("How learning works", [
                "Q-learning updates a table of action values per state.",
                "Epsilon controls exploration vs exploitation.",
                "More episodes usually improve policy quality over time.",
            ]),
            ("View & overlays", [
                "Telemetry (Ctrl+T): episode reward/steps trend.",
                "Run history (Ctrl+K): recent episode outcomes.",
                "Debug (D): shows internal details for troubleshooting.",
            ]),
        ]
        controls = [f"{entry['keys']}: {entry['action']} - {entry['desc']}" for entry in get_keymap("Simulation")]
        sections += [
            ("Controls (simulation)", controls),
            ("Files & outputs", [
                "Save/Load Q-table: keep trained agents between sessions.",
                "Env snapshot: save exact state to resume or share.",
                "Export stats: CSV + JSON summaries of runs.",
                "Screenshot: capture the current view.",
            ]),
            ("File dialog tips", [
                "Click a file to select it, Enter or OK to confirm, Esc to cancel.",
                "Ctrl+V pastes a path into the input box.",
            ]),
            ("Exit", ["Press Esc to return."]),
        ]

        pad = int(60 * app.theme.ui_scale)
        inner_pad = int(40 * app.theme.ui_scale)
        col_gap = int(36 * app.theme.ui_scale)
        row_gap = int(14 * app.theme.ui_scale)
        line_h = int(22 * app.theme.ui_scale)
        title_h = int(60 * app.theme.ui_scale)
        view_h = screen.get_height() - 2 * pad
        avail_h = max(1, view_h - title_h)

        def wrap_line(text: str, use_font: pygame.font.Font, max_w: int) -> list[str]:
            if not text:
                return [""]
            if use_font.size(text)[0] <= max_w:
                return [text]
            words = text.split(" ")
            out: list[str] = []
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if use_font.size(test)[0] <= max_w:
                    cur = test
                else:
                    if cur:
                        out.append(cur)
                    cur = w
            if cur:
                out.append(cur)
            return out

        panel_rect = pygame.Rect(pad, pad, screen.get_width() - 2 * pad, screen.get_height() - 2 * pad)
        inner_w = max(1, panel_rect.w - 2 * inner_pad)
        max_w = max(240, inner_w)
        cols = 3 if max_w >= 1200 else (2 if max_w >= 860 else 1)
        col_w = max(260, (max_w - (cols - 1) * col_gap) // cols)

        def section_height(items: list[str]) -> int:
            h = line_h  # header
            for line in items:
                wrapped = wrap_line(line, small, col_w)
                h += len(wrapped) * line_h
            return h + row_gap

        # Balance sections across columns for a cleaner layout.
        columns: list[list[tuple[str, list[str]]]] = [[] for _ in range(cols)]
        heights = [0 for _ in range(cols)]
        for title, items in sections:
            idx = heights.index(min(heights))
            columns[idx].append((title, items))
            heights[idx] += section_height(items)

        content_h = title_h + max(heights)
        self.max_scroll = max(0, content_h - view_h)
        self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))

        start_y = pad - self.scroll_y
        if content_h <= view_h:
            start_y = pad + (view_h - content_h) // 2

        # Panel background for a cleaner, professional look.
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel.fill((*app.theme.palette.panel, app.theme.palette.panel_alpha))
        screen.blit(panel, panel_rect.topleft)
        pygame.draw.rect(screen, app.theme.palette.grid_line, panel_rect, 1, border_radius=12)

        # Draw title centered
        title_text = "Mini RL Survival - Help"
        t = title_font.render(title_text, True, app.theme.palette.fg)
        screen.blit(t, ((screen.get_width() - t.get_width()) // 2, start_y))

        # Draw sectioned columns (left-aligned)
        total_w = cols * col_w + (cols - 1) * col_gap
        start_x = panel_rect.x + inner_pad + max(0, (inner_w - total_w) // 2)
        body_y0 = start_y + title_h
        for col in range(cols):
            col_x = start_x + col * (col_w + col_gap)
            y = body_y0
            for header, items in columns[col]:
                header_surf = font.render(header + ":", True, app.theme.palette.fg)
                screen.blit(header_surf, (col_x, y))
                y += line_h
                for line in items:
                    for part in wrap_line(line, small, col_w):
                        line_surf = small.render(part, True, app.theme.palette.fg)
                        screen.blit(line_surf, (col_x, y))
                        y += line_h
                y += row_gap
