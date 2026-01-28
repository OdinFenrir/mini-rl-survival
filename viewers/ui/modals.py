import importlib
import os
from typing import Callable, Optional

import pygame

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
                self.on_cancel()
                self.close()
                return
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_TAB):
                self.focus = 1 - self.focus
            elif event.key == pygame.K_RETURN:
                if self.focus == 0:
                    self.on_confirm()
                else:
                    self.on_cancel()
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

