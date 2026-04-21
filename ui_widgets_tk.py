# -*- coding: utf-8 -*-
"""
Переиспользуемые UI-компоненты:
  Tooltip            — всплывающая подсказка
  ScrollableFrame    — прокручиваемый фрейм
  GradientBackground — полноэкранный градиентный фон (Apple Dark)
  RoundedButton      — кнопка со скруглёнными углами (Apple-style)
  RoundedCard        — карточка со скруглёнными углами (Canvas-based)
  StatusPill         — цветной бейдж-статус
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from ui_theme import _best_font
from app_logger import get_logger

_log = get_logger(__name__)

# PIL/Pillow — опционально, нужен для масштабирования фонового изображения.
try:
    from PIL import Image as _PILImage
    from PIL import ImageTk as _PILImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ─────────────────────────────────────────────────────────────────────────────
# Tooltip
# ─────────────────────────────────────────────────────────────────────────────

class Tooltip:
    """
    Всплывающая подсказка для любого tkinter-виджета.
    Появляется через delay_ms мс после наведения мыши, скрывается при уходе.
    """

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        delay_ms: int = 350,
        font_size: Optional[int] = None,
    ) -> None:
        self.widget = widget
        self.text = text or ""
        self.delay = delay_ms
        self._font_size = int(font_size or self._recommended_font_size())
        self._after: Optional[str] = None
        self._tip: Optional[tk.Toplevel] = None

        widget.bind("<Enter>",       self._schedule, add="+")
        widget.bind("<Leave>",       self._hide,     add="+")
        widget.bind("<ButtonPress>", self._hide,     add="+")
        widget.bind("<Destroy>",     self._hide,     add="+")

    def _recommended_font_size(self) -> int:
        try:
            scaling = float(self.widget.tk.call("tk", "scaling"))
        except (tk.TclError, RuntimeError, ValueError, TypeError) as _e:
            _log.debug("Tooltip scaling lookup failed: %s", _e)
            scaling = 1.0
        return max(10, int(round(9 * scaling)))

    def _schedule(self, _e: object = None) -> None:
        if not self.text:
            return
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except (tk.TclError, RuntimeError) as _e:
                _log.debug("Tooltip.after_cancel failed: %s", _e)
            self._after = None
        self._after = self.widget.after(self.delay, self._show)

    def _show(self) -> None:
        if self._tip or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        except (tk.TclError, RuntimeError) as _e:
            _log.debug("Tooltip geometry lookup failed: %s", _e)
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.attributes("-topmost", True)
        self._tip.wm_geometry(f"+{x}+{y}")

        # Outer border — paper-dark border colour
        outer = tk.Frame(self._tip, bg="#2d2f3a", padx=1, pady=1)
        outer.pack(fill="both", expand=True)

        # Inner background
        inner = tk.Frame(outer, bg="#1c1d23", padx=0, pady=0)
        inner.pack(fill="both", expand=True)

        # Thin accent bar at top — orange
        accent_bar = tk.Frame(inner, bg="#f97316", height=2)
        accent_bar.pack(fill="x")

        lbl = tk.Label(
            inner,
            text=self.text,
            justify="left",
            bg="#1c1d23",
            fg="#e8e9f0",
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=10,
            font=(_best_font(), self._font_size),
            wraplength=520,
        )
        lbl.pack(fill="both", expand=True)

    def _hide(self, _e: object = None) -> None:
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except (tk.TclError, RuntimeError) as _e:
                _log.debug("Tooltip.after_cancel on hide failed: %s", _e)
            self._after = None
        if self._tip:
            try:
                self._tip.destroy()
            except (tk.TclError, RuntimeError) as _e:
                _log.debug("Tooltip destroy failed: %s", _e)
            self._tip = None


# ─────────────────────────────────────────────────────────────────────────────
# ScrollableFrame
# ─────────────────────────────────────────────────────────────────────────────

class ScrollableFrame(ttk.Frame):
    """
    ttk.Frame с вертикальной прокруткой через Canvas.
    Используй self.inner как контейнер для дочерних виджетов.
    Поддерживает прокрутку колесом мыши (Windows / Linux).
    """

    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>",  self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Локальные bind (вместо bind_all), чтобы несколько ScrollableFrame
        # не накапливали глобальные обработчики колеса мыши.
        for _w in (self.canvas, self.inner):
            _w.bind("<MouseWheel>", self._on_mousewheel, add="+")
            _w.bind("<Button-4>", self._on_mousewheel, add="+")
            _w.bind("<Button-5>", self._on_mousewheel, add="+")
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._wheel_enabled = True

    def _on_destroy(self, _event: object = None) -> None:
        try:
            for _w in (self.canvas, self.inner):
                _w.unbind("<MouseWheel>")
                _w.unbind("<Button-4>")
                _w.unbind("<Button-5>")
        except (tk.TclError, RuntimeError) as _e:
            _log.debug("ScrollableFrame unbind failed: %s", _e)

    def _on_configure(self, _e: object = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if not self._wheel_enabled:
            return
        # Не перехватываем скролл у виджетов с собственной прокруткой (Treeview, Text).
        # Это позволяет прокручивать Treeview / Text внутри ScrollableFrame по-прежнему,
        # а сам фрейм — только когда курсор над canvas или inner frame.
        if isinstance(event.widget, (tk.Text, ttk.Treeview)):
            return
        try:
            x, y = self.canvas.winfo_pointerxy()
            cx1 = self.canvas.winfo_rootx()
            cy1 = self.canvas.winfo_rooty()
            cx2 = cx1 + self.canvas.winfo_width()
            cy2 = cy1 + self.canvas.winfo_height()
            if not (cx1 <= x <= cx2 and cy1 <= y <= cy2):
                return
        except (tk.TclError, RuntimeError) as _e:
            _log.debug("ScrollableFrame pointer bounds lookup failed: %s", _e)
            return

        if hasattr(event, "delta") and event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")


def _parse_hex_color(hex_color: str) -> tuple:
    """Parse #RRGGBB hex string to (r, g, b) tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ─────────────────────────────────────────────────────────────────────────────
# GradientBackground
# ─────────────────────────────────────────────────────────────────────────────

class GradientBackground(tk.Canvas):
    """
    Полноэкранный градиентный фон в стиле Dark Teal.
    Помещается первым в корневом окне и опускается за все виджеты через .lower().
    """

    _DEBOUNCE_MS = 80
    _BAND_PX     = 3

    def __init__(
        self,
        master: tk.Widget,
        color_top: str    = "#050f0c",
        color_bottom: str = "#030a08",
    ) -> None:
        super().__init__(master, highlightthickness=0, bd=0)
        self._ct = _parse_hex_color(color_top)
        self._cb = _parse_hex_color(color_bottom)
        self._after_id: Optional[str] = None
        self.bind("<Configure>", self._schedule)

    def _schedule(self, _e: object = None) -> None:
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(self._DEBOUNCE_MS, self._draw)

    def _draw(self) -> None:
        self._after_id = None
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return

        r1, g1, b1 = self._ct
        r2, g2, b2 = self._cb
        s = self._BAND_PX

        for y in range(0, h, s):
            t = y / h
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            self.create_rectangle(0, y, w, y + s + 1,
                                  fill=f"#{r:02x}{g:02x}{b:02x}", outline="")


# ─────────────────────────────────────────────────────────────────────────────
# ImageBackground
# ─────────────────────────────────────────────────────────────────────────────

class ImageBackground(tk.Canvas):
    """
    Полноэкранный фон на основе изображения (PNG/JPG и др.).
    Если файл не найден или не загружается — рисует градиент Apple Dark.
    """

    _DEBOUNCE_MS = 60
    _BAND_PX     = 3

    def __init__(
        self,
        master: tk.Widget,
        image_path=None,
        color_top:      str = "#050f0c",
        color_bottom:   str = "#030a08",
        overlay_color:  str = "#030e0a",
        overlay_stipple: str = "gray50",
        overlay_alpha:  int = 90,
    ) -> None:
        super().__init__(master, highlightthickness=0, bd=0)
        self._ct             = _parse_hex_color(color_top)
        self._cb             = _parse_hex_color(color_bottom)
        self._overlay_color  = overlay_color
        self._overlay_stipple = overlay_stipple
        self._overlay_alpha  = max(0, min(255, overlay_alpha))

        self._pil_source: Optional[object] = None
        self._photo:      Optional[object] = None
        self._last_size:  tuple            = (0, 0)
        self._after_id:   Optional[str]   = None

        if image_path is not None:
            if _HAS_PIL:
                try:
                    self._pil_source = _PILImage.open(str(image_path)).convert("RGBA")
                except (FileNotFoundError, OSError, ValueError) as _e:
                    _log.debug("ImageBackground PIL source load failed: %s", _e)
                    self._pil_source = None
            else:
                try:
                    self._photo = tk.PhotoImage(file=str(image_path))
                except (tk.TclError, RuntimeError, OSError) as _e:
                    _log.debug("ImageBackground tk.PhotoImage load failed: %s", _e)
                    self._photo = None

        self.bind("<Configure>", self._schedule)

    def _schedule(self, _e: object = None) -> None:
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(self._DEBOUNCE_MS, self._draw)

    def _draw(self) -> None:
        self._after_id = None
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return

        if self._pil_source is not None and _HAS_PIL:
            if (w, h) != self._last_size:
                img_w, img_h = self._pil_source.size
                scale = max(w / img_w, h / img_h)
                new_w = max(1, int(img_w * scale))
                new_h = max(1, int(img_h * scale))
                scaled = self._pil_source.resize((new_w, new_h), _PILImage.LANCZOS)
                left = (new_w - w) // 2
                top  = (new_h - h) // 2
                cropped = scaled.crop((left, top, left + w, top + h))
                oc = _parse_hex_color(self._overlay_color)
                overlay = _PILImage.new(
                    "RGBA", (w, h),
                    (oc[0], oc[1], oc[2], self._overlay_alpha),
                )
                result = _PILImage.alpha_composite(cropped, overlay)
                self._photo = _PILImageTk.PhotoImage(result)
                self._last_size = (w, h)
            self.create_image(0, 0, image=self._photo, anchor="nw")

        elif self._photo is not None:
            img_w = self._photo.width()
            img_h = self._photo.height()
            for tx in range(0, w, img_w):
                for ty in range(0, h, img_h):
                    self.create_image(tx, ty, image=self._photo, anchor="nw")
            self.create_rectangle(
                0, 0, w, h,
                fill=self._overlay_color,
                stipple=self._overlay_stipple,
                outline="",
            )

        else:
            r1, g1, b1 = self._ct
            r2, g2, b2 = self._cb
            s = self._BAND_PX
            for y in range(0, h, s):
                t = y / h
                r = int(r1 + (r2 - r1) * t)
                g = int(g1 + (g2 - g1) * t)
                b = int(b1 + (b2 - b1) * t)
                self.create_rectangle(
                    0, y, w, y + s + 1,
                    fill=f"#{r:02x}{g:02x}{b:02x}", outline=""
                )


# ─────────────────────────────────────────────────────────────────────────────
# RoundedButton
# ─────────────────────────────────────────────────────────────────────────────

class RoundedButton(tk.Canvas):
    """
    Кнопка со скруглёнными углами, нарисованная на Canvas (Apple-style).
    Поддерживает hover / press / disabled состояния.
    """

    def __init__(
        self,
        master: tk.Widget,
        *,
        text: str,
        command: Optional[Callable] = None,
        radius: int = 14,
        bg_normal:   str = "#00c896",
        bg_hover:    str = "#1de9b6",
        bg_press:    str = "#009b75",
        bg_disabled: str = "#102418",
        fg_normal:   str = "#ffffff",
        fg_disabled: str = "#3e7a62",
        font: Optional[tuple] = None,
        padx: int = 24,
        pady: int = 10,
        **kw,
    ) -> None:
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("bd", 0)
        kw.setdefault("cursor", "hand2")

        _font = font or ("Segoe UI", 11, "bold")
        if "width" not in kw:
            kw["width"] = self._measure_text(text, _font) + 2 * padx
        if "height" not in kw:
            kw["height"] = self._font_height(_font) + 2 * pady

        super().__init__(master, **kw)

        self._text    = text
        self._command = command
        self._radius  = radius
        self._bg = dict(n=bg_normal, h=bg_hover, p=bg_press, d=bg_disabled)
        self._fg = dict(n=fg_normal, d=fg_disabled)
        self._font    = _font
        self._state   = "normal"
        self._hover   = False
        self._pressed = False

        self.bind("<Configure>",       lambda _e: self._draw())
        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ── Public API ────────────────────────────────────────────────────────────

    def configure(self, **kw) -> None:
        if "state" in kw:
            self._state = kw.pop("state")
        if "text" in kw:
            self._text = kw.pop("text")
        if kw:
            super().configure(**kw)
        self._draw()

    config = configure

    # ── Internals ──────────────────────────────────────────────────────────────

    def _bg_now(self) -> str:
        if self._state == "disabled":
            return self._bg["d"]
        if self._pressed:
            return self._bg["p"]
        if self._hover:
            return self._bg["h"]
        return self._bg["n"]

    def _fg_now(self) -> str:
        return self._fg["d"] if self._state == "disabled" else self._fg["n"]

    def _draw(self) -> None:
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return
        r = min(self._radius, h // 2, w // 2)
        bg = self._bg_now()

        # Основная скруглённая форма
        self.create_polygon(
            r,     0,
            w - r, 0,
            w,     0,
            w,     r,
            w,     h - r,
            w,     h,
            w - r, h,
            r,     h,
            0,     h,
            0,     h - r,
            0,     r,
            0,     0,
            smooth=True,
            fill=bg,
            outline="",
        )

        # Subtle highlight line at top (Apple-style inner glow)
        if self._state != "disabled" and not self._pressed:
            highlight = self._lighten(bg, 30)
            self.create_line(
                r + 2, 1, w - r - 2, 1,
                fill=highlight, width=1,
            )

        self.create_text(
            w // 2, h // 2,
            text=self._text,
            fill=self._fg_now(),
            font=self._font,
        )

    def _lighten(self, hex_color: str, amount: int) -> str:
        """Слегка осветляет цвет для inner glow."""
        try:
            h = hex_color.lstrip("#")
            r = min(255, int(h[0:2], 16) + amount)
            g = min(255, int(h[2:4], 16) + amount)
            b = min(255, int(h[4:6], 16) + amount)
            return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, TypeError) as _e:
            _log.debug("RoundedButton._lighten fallback for %r: %s", hex_color, _e)
            return hex_color

    def _on_enter(self, _e: object) -> None:
        if self._state != "disabled":
            self._hover = True
            self._draw()

    def _on_leave(self, _e: object) -> None:
        self._hover   = False
        self._pressed = False
        self._draw()

    def _on_press(self, _e: object) -> None:
        if self._state != "disabled":
            self._pressed = True
            self._draw()

    def _on_release(self, _e: object) -> None:
        if self._state != "disabled":
            was = self._pressed
            self._pressed = False
            self._draw()
            if was and self._command:
                self._command()

    @staticmethod
    def _measure_text(text: str, font: tuple) -> int:
        try:
            import tkinter.font as tkfont
            f = tkfont.Font(family=font[0], size=font[1])
            return f.measure(text)
        except (tk.TclError, RuntimeError, TypeError, ValueError) as _e:
            _log.debug("RoundedButton._measure_text fallback for %r: %s", text, _e)
            size = font[1] if len(font) > 1 else 11
            return int(len(text) * size * 0.68)

    @staticmethod
    def _font_height(font: tuple) -> int:
        try:
            import tkinter.font as tkfont
            f = tkfont.Font(family=font[0], size=font[1])
            return f.metrics("linespace")
        except (tk.TclError, RuntimeError, TypeError, ValueError) as _e:
            _log.debug("RoundedButton._font_height fallback: %s", _e)
            size = font[1] if len(font) > 1 else 11
            return int(size * 1.6)


# ─────────────────────────────────────────────────────────────────────────────
# RoundedCard
# ─────────────────────────────────────────────────────────────────────────────

class RoundedCard(tk.Canvas):
    """
    Карточка с реально скруглёнными углами (Canvas-based).

    Использование:
        card = RoundedCard(parent, title="Секция", radius=14)
        card.pack(fill="x", pady=(0, 10))
        # Помещай дочерние виджеты в card.inner:
        ttk.Label(card.inner, text="Содержимое").pack(anchor="w")

    Параметры:
        title       — заголовок карточки (пустая строка = без заголовка)
        radius      — радиус скругления углов
        bg_card     — цвет фона карточки
        bg_outer    — цвет внешнего фона (родительский)
        border      — цвет рамки
        padding     — внутренний отступ
        accent_top  — цвет акцентной полосы сверху (пустая строка = нет)
    """

    def __init__(
        self,
        master: tk.Widget,
        *,
        title: str = "",
        radius: int = 14,
        bg_card: str = "#0b1c16",
        bg_outer: str = "#050f0c",
        border: str = "#1c4a35",
        padding: int = 12,
        accent_top: str = "",
        title_color: str = "#1de9b6",
        title_font: Optional[tuple] = None,
        **kw,
    ) -> None:
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("bd", 0)
        kw["bg"] = bg_outer
        super().__init__(master, **kw)

        self._radius      = radius
        self._bg_card     = bg_card
        self._bg_outer    = bg_outer
        self._border      = border
        self._padding     = padding
        self._accent_top  = accent_top
        self._title       = title
        self._title_color = title_color
        self._title_font  = title_font or ("Segoe UI", 10, "bold")

        # Вычисляем отступ для заголовка
        self._title_h = 0
        if title:
            self._title_h = self._measure_title_height()

        # Внутренний Frame с таким же фоном
        self.inner = tk.Frame(self, bg=bg_card, padx=padding, pady=padding)
        self._win_id = self.create_window(
            0, self._title_h,
            window=self.inner,
            anchor="nw",
        )

        self.bind("<Configure>",      self._on_canvas_configure)
        self.inner.bind("<Configure>", self._on_inner_configure)

    def _measure_title_height(self) -> int:
        try:
            import tkinter.font as tkfont
            f = tkfont.Font(family=self._title_font[0], size=self._title_font[1])
            return f.metrics("linespace") + 10
        except (tk.TclError, RuntimeError, TypeError, ValueError) as _e:
            _log.debug("RoundedCard._measure_title_height fallback: %s", _e)
            return 24

    def _on_canvas_configure(self, event: tk.Event) -> None:
        w = event.width
        h = event.height
        self.itemconfigure(self._win_id, width=w)
        self._draw(w, h)

    def _on_inner_configure(self, event: tk.Event) -> None:
        needed_h = event.height + self._title_h
        if abs(needed_h - self.winfo_height()) > 2:
            self.configure(height=needed_h)
        self._draw()

    def _draw(self, w: int = None, h: int = None) -> None:
        w = w or self.winfo_width()
        h = h or self.winfo_height()
        if w < 4 or h < 4:
            return

        self.delete("card_bg")
        r = min(self._radius, w // 2, h // 2)

        # Основная карточка
        self.create_polygon(
            r,     0,
            w - r, 0,
            w,     0,
            w,     r,
            w,     h - r,
            w,     h,
            w - r, h,
            r,     h,
            0,     h,
            0,     h - r,
            0,     r,
            0,     0,
            smooth=True,
            fill=self._bg_card,
            outline=self._border,
            width=1,
            tags="card_bg",
        )

        # Акцентная полоса сверху
        if self._accent_top:
            accent_h = 3
            ar = min(r, accent_h)
            self.create_polygon(
                ar, 0,
                w - ar, 0,
                w, ar,
                w, accent_h,
                0, accent_h,
                0, ar,
                smooth=True,
                fill=self._accent_top,
                outline="",
                tags="card_bg",
            )

        # Заголовок
        if self._title and self._title_h > 0:
            # Подложка заголовка
            self.create_rectangle(
                1, 0, w - 1, self._title_h,
                fill=self._bg_card,
                outline="",
                tags="card_bg",
            )
            # Разделитель
            self.create_line(
                self._padding, self._title_h - 1,
                w - self._padding, self._title_h - 1,
                fill=self._border,
                width=1,
                tags="card_bg",
            )
            # Текст заголовка
            self.create_text(
                self._padding + 4, self._title_h // 2,
                text=self._title,
                fill=self._title_color,
                font=self._title_font,
                anchor="w",
                tags="card_bg",
            )

        self.tag_lower("card_bg")

    def set_title(self, title: str) -> None:
        """Обновить заголовок карточки."""
        self._title = title
        self._title_h = self._measure_title_height() if title else 0
        self.coords(self._win_id, 0, self._title_h)
        self._draw()


# ─────────────────────────────────────────────────────────────────────────────
# StatusPill
# ─────────────────────────────────────────────────────────────────────────────

class CollapsibleSection(ttk.Frame):
    """
    Сворачиваемый блок с кнопкой-заголовком.
    Дочерние виджеты добавляй в .body (ttk.Frame).

    Использование:
        sec = CollapsibleSection(parent, title="Расширенные настройки")
        sec.pack(fill="x", pady=(0, 6))
        ttk.Label(sec.body, text="Содержимое").pack(anchor="w")
    """

    def __init__(self, master, title: str = "Расширенные настройки",
                 collapsed: bool = True, **kw) -> None:
        super().__init__(master, **kw)
        self._collapsed = collapsed
        self._title = title

        hdr = ttk.Frame(self)
        hdr.pack(fill="x")
        self._btn = ttk.Button(hdr, text=self._label(), command=self._toggle)
        self._btn.pack(fill="x")

        self.body = ttk.Frame(self)
        if not collapsed:
            self.body.pack(fill="x", pady=(4, 0))

    def _label(self) -> str:
        arrow = "▶" if self._collapsed else "▼"
        return f"  {arrow}   {self._title}"

    def _toggle(self) -> None:
        if self._collapsed:
            self.body.pack(fill="x", pady=(4, 0))
            self._collapsed = False
        else:
            self.body.pack_forget()
            self._collapsed = True
        self._btn.configure(text=self._label())


# ─────────────────────────────────────────────────────────────────────────────
# ToggleSwitch
# ─────────────────────────────────────────────────────────────────────────────

class ToggleSwitch(tk.Canvas):
    """
    iOS-style toggle switch backed by a tk.BooleanVar.

    Использование:
        ts = ToggleSwitch(parent, variable=self.use_sbert, bg=PANEL)
        ts.pack(side="left", padx=4)
    """

    _W = 44   # track width
    _H = 24   # track height
    _R = 9    # knob radius

    def __init__(
        self,
        master: tk.Widget,
        variable: "tk.BooleanVar",
        *,
        on_color:  str = "#00c896",
        off_color: str = "#1c4a35",
        command: Optional[Callable] = None,
        bg: str = "#0b1c16",
        **kw,
    ) -> None:
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("bd", 0)
        kw.setdefault("cursor", "hand2")
        kw["width"]  = self._W
        kw["height"] = self._H
        kw["bg"]     = bg
        super().__init__(master, **kw)

        self._var     = variable
        self._on_c    = on_color
        self._off_c   = off_color
        self._command = command

        self.bind("<ButtonPress-1>", self._on_click)
        self._var.trace_add("write", lambda *_: self.after_idle(self._draw))
        self.bind("<Configure>", lambda _: self._draw())

    # ── helpers ────────────────────────────────────────────────────────────────

    def _knob_x(self) -> int:
        return (self._W - self._R - 3) if self._var.get() else (self._R + 3)

    def _draw(self) -> None:
        val = bool(self._var.get())
        bg  = self._on_c if val else self._off_c
        kx  = self._knob_x()
        w, h = self._W, self._H
        r    = h // 2

        self.delete("all")

        # Track (pill shape)
        self.create_polygon(
            r, 0,  w - r, 0,  w, 0,  w, r,
            w, h - r,  w, h,  w - r, h,
            r, h,  0, h,  0, h - r,  0, r,  0, 0,
            smooth=True, fill=bg, outline="",
        )

        # Knob
        pad = 2
        self.create_oval(
            kx - self._R + pad, pad,
            kx + self._R - pad, h - pad,
            fill="#ffffff", outline="",
        )

    def _on_click(self, _=None) -> None:
        self._var.set(not bool(self._var.get()))
        if self._command:
            self._command()


# ─────────────────────────────────────────────────────────────────────────────
# PillTabBar
# ─────────────────────────────────────────────────────────────────────────────

class PillTabBar(tk.Frame):
    """
    Горизонтальная навигационная панель с pill-индикатором активной вкладки.
    Заменяет ttk.Notebook header.

    Использование:
        bar = PillTabBar(parent, tabs=["Обучение", "Классификация"],
                         on_change=lambda i: switch_to(i))
        bar.pack(fill="x")
    """

    _ACTIVE_FG     = "#1de9b6"
    _INACTIVE_FG   = "#6db39a"
    _INDICATOR_COL = "#00c896"
    _BG            = "#050f0c"

    def __init__(
        self,
        master: tk.Widget,
        tabs: list,
        *,
        on_change: Optional[Callable] = None,
        bg: str = _BG,
        active_fg:     str = _ACTIVE_FG,
        inactive_fg:   str = _INACTIVE_FG,
        indicator_col: str = _INDICATOR_COL,
        font_size: int = 10,
        **kw,
    ) -> None:
        super().__init__(master, bg=bg, **kw)
        self._tabs          = tabs
        self._on_change     = on_change
        self._active        = 0
        self._bg            = bg
        self._active_fg     = active_fg
        self._inactive_fg   = inactive_fg
        self._indicator_col = indicator_col
        self._font_size     = font_size

        self._cells:  list = []
        self._labels: list = []
        self._dots:   list = []

        self._build()

    def _build(self) -> None:
        for i, text in enumerate(self._tabs):
            cell = tk.Frame(self, bg=self._bg, cursor="hand2", padx=2)
            cell.pack(side="left", pady=4, padx=2)
            self._cells.append(cell)

            dot = tk.Canvas(cell, width=8, height=8,
                            highlightthickness=0, bd=0, bg=self._bg)
            dot.pack()
            self._dots.append(dot)

            lbl = tk.Label(
                cell, text=text, bg=self._bg,
                fg=self._inactive_fg, cursor="hand2",
                font=("Segoe UI", self._font_size),
                padx=14, pady=5,
            )
            lbl.pack()
            self._labels.append(lbl)

            for w in (cell, dot, lbl):
                w.bind("<Button-1>", lambda _e, idx=i: self._user_select(idx))

        self._update_styles()

    # ── Public API ─────────────────────────────────────────────────────────────

    def select(self, index: int) -> None:
        """Переключить активную вкладку (без вызова on_change callback)."""
        self._active = index
        self._update_styles()

    def _user_select(self, index: int) -> None:
        """Select tab on user click — fires on_change callback."""
        self._active = index
        self._update_styles()
        if self._on_change:
            self._on_change(index)

    def set_font_size(self, size: int) -> None:
        """Обновить размер шрифта (вызывается из _apply_zoom)."""
        self._font_size = size
        self._update_styles()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update_styles(self) -> None:
        fs = self._font_size
        for i, (lbl, dot) in enumerate(zip(self._labels, self._dots)):
            dot.delete("all")
            if i == self._active:
                lbl.configure(fg=self._active_fg,
                              font=("Segoe UI", fs, "bold"))
                dot.create_oval(1, 1, 7, 7,
                                fill=self._indicator_col, outline="")
            else:
                lbl.configure(fg=self._inactive_fg,
                              font=("Segoe UI", fs))


class StatusPill(tk.Canvas):
    """
    Цветной бейдж-пилюля для отображения статусов (OK, Warning, Error, etc.).

    Использование:
        pill = StatusPill(parent, text="Готово", color="#30d158")
        pill.pack(side="left", padx=4)
    """

    def __init__(
        self,
        master: tk.Widget,
        *,
        text: str = "",
        color: str = "#30d158",
        fg: str = "#ffffff",
        bg: str = "#0d1b2a",
        font: Optional[tuple] = None,
        padx: int = 10,
        pady: int = 4,
        **kw,
    ) -> None:
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("bd", 0)
        kw["bg"] = bg

        _font = font or ("Segoe UI", 9, "bold")
        w = self._measure(text, _font) + 2 * padx
        h = self._line_h(_font) + 2 * pady
        kw.setdefault("width", max(w, 40))
        kw.setdefault("height", h)

        super().__init__(master, **kw)
        self._text  = text
        self._color = color
        self._fg    = fg
        self._font  = _font
        self._padx  = padx

        self.bind("<Configure>", lambda _e: self._draw())

    def configure(self, **kw) -> None:
        if "text" in kw:
            self._text = kw.pop("text")
        if "color" in kw:
            self._color = kw.pop("color")
        if kw:
            super().configure(**kw)
        self._draw()

    config = configure

    def _draw(self) -> None:
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return
        r = h // 2
        self.create_polygon(
            r, 0, w - r, 0, w, r, w - r, h, r, h, 0, r,
            smooth=True, fill=self._color, outline="",
        )
        self.create_text(
            w // 2, h // 2,
            text=self._text, fill=self._fg, font=self._font,
        )

    @staticmethod
    def _measure(text: str, font: tuple) -> int:
        try:
            import tkinter.font as tkfont
            return tkfont.Font(family=font[0], size=font[1]).measure(text)
        except (tk.TclError, RuntimeError, TypeError, ValueError) as _e:
            _log.debug("StatusPill._measure fallback for %r: %s", text, _e)
            return len(text) * 7

    @staticmethod
    def _line_h(font: tuple) -> int:
        try:
            import tkinter.font as tkfont
            return tkfont.Font(family=font[0], size=font[1]).metrics("linespace")
        except (tk.TclError, RuntimeError, TypeError, ValueError) as _e:
            _log.debug("StatusPill._line_h fallback: %s", _e)
            return 14
