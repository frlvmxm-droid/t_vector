# -*- coding: utf-8 -*-
"""
Тема Dark Teal / Matrix Green для Tkinter / ttk.
Палитра соответствует референсу: тёмный зелёно-бирюзовый фон,
frosted-glass карточки, яркие teal-акценты.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

# ── Палитра Dark Teal ─────────────────────────────────────────────────────────
BG       = "#050f0c"   # основной фон — почти чёрный зелёный
PANEL    = "#0b1c16"   # карточки первого уровня
PANEL2   = "#102418"   # вложенные карточки
FG       = "#e4f4ee"   # основной текст — тёплый белый с зелёным оттенком
MUTED    = "#6db39a"   # второстепенный текст
MUTED2   = "#3e7a62"   # третичный текст
ENTRY_BG = "#040d0a"   # поля ввода
BORDER   = "#1c4a35"   # рамки карточек
BORDER2  = "#112d20"   # тонкие внутренние разделители
ACCENT   = "#00c896"   # основной teal-акцент
ACCENT2  = "#1de9b6"   # hover — светлее
ACCENT3  = "#009b75"   # pressed — темнее
SELECT   = "#0a3d2a"   # выделение в списках
SUCCESS  = "#1de9b6"   # успех — светлый teal
SUCCESS2 = "#0a6b4a"   # успех фон
WARNING  = "#f0b429"   # предупреждение — янтарный
ERROR    = "#ff5252"   # ошибка
HOVER    = "#0d2b1f"   # hover фон строк / кнопок


_FONT_CACHE: Optional[str] = None


def _best_font() -> str:
    """Возвращает лучший доступный шрифт (SF Pro → Segoe UI → Helvetica)."""
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE
    try:
        from tkinter import font as tkfont
        available = set(tkfont.families())
        for name in (
            "SF Pro Display", "SF Pro Text", ".SF NS Display",
            "Segoe UI Variable Display", "Segoe UI",
            "Helvetica Neue", "Helvetica", "Arial",
        ):
            if name in available:
                _FONT_CACHE = name
                return _FONT_CACHE
    except Exception:
        pass
    _FONT_CACHE = "Segoe UI"
    return _FONT_CACHE


def apply_dark_theme(root: tk.Tk, font_size: int = 10) -> None:
    """
    Применяет Dark Teal тему ко всем ttk-виджетам.
    """
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    FF  = _best_font()
    F   = (FF, font_size)
    FB  = (FF, font_size, "bold")
    FSB = (FF, font_size + 1, "bold")

    root.configure(bg=BG)

    # ── Базовые ─────────────────────────────────────────────────────────────
    style.configure(".",
        background=BG, foreground=FG,
        font=F, bordercolor=BORDER,
        focuscolor=ACCENT,
    )
    style.configure("TFrame",       background=BG)
    style.configure("Card.TFrame",  background=PANEL)
    style.configure("Card2.TFrame", background=PANEL2)
    style.configure("Inner.TFrame", background=ENTRY_BG)

    # ── Label ────────────────────────────────────────────────────────────────
    style.configure("TLabel",                   background=BG,     foreground=FG)
    style.configure("Muted.TLabel",             background=BG,     foreground=MUTED)
    style.configure("Muted2.TLabel",            background=BG,     foreground=MUTED2)
    style.configure("Card.TLabel",              background=PANEL,  foreground=FG)
    style.configure("Card.Muted.TLabel",        background=PANEL,  foreground=MUTED)
    style.configure("Card2.TLabel",             background=PANEL2, foreground=FG)
    style.configure("Card2.Muted.TLabel",       background=PANEL2, foreground=MUTED)
    style.configure("Accent.TLabel",            background=BG,     foreground=ACCENT2, font=FB)
    style.configure("Header.TLabel",            background=BG,     foreground=FG,      font=FSB)
    style.configure("Card.Header.TLabel",       background=PANEL,  foreground=FG,      font=FSB)
    style.configure("Card.Accent.TLabel",       background=PANEL,  foreground=ACCENT2, font=FB)
    style.configure("Success.TLabel",           background=BG,     foreground=SUCCESS, font=FB)
    style.configure("Warning.TLabel",           background=BG,     foreground=WARNING)
    style.configure("Error.TLabel",             background=BG,     foreground=ERROR)
    style.configure("Card.Success.TLabel",      background=PANEL,  foreground=SUCCESS)
    style.configure("Small.Muted.TLabel",       background=BG,     foreground=MUTED,   font=(FF, font_size - 1))
    style.configure("Card.Small.Muted.TLabel",  background=PANEL,  foreground=MUTED,   font=(FF, font_size - 1))

    # ── LabelFrame — frosted-glass секция ────────────────────────────────────
    style.configure("TLabelframe",
        background=BG,
        foreground=MUTED,
        bordercolor=BG,
        lightcolor=BG,
        darkcolor=BG,
        relief="flat",
        borderwidth=0,
        padding=(14, 10, 14, 12),
    )
    style.configure("TLabelframe.Label",
        background=BG,
        foreground=ACCENT2,
        font=FB,
        padding=(4, 2, 8, 2),
    )

    # Step LabelFrame
    style.configure("Step.TLabelframe",
        background=BG,
        bordercolor=BG,
        lightcolor=BG,
        darkcolor=BG,
        relief="flat",
        borderwidth=0,
        padding=(14, 10, 14, 12),
    )
    style.configure("Step.TLabelframe.Label",
        background=BG,
        foreground=ACCENT2,
        font=FSB,
        padding=(6, 2, 8, 2),
    )

    # Warn LabelFrame
    style.configure("Warn.TLabelframe",
        background=BG,
        bordercolor=BG,
        lightcolor=BG,
        darkcolor=BG,
        relief="flat",
        borderwidth=0,
        padding=(14, 10, 14, 12),
    )
    style.configure("Warn.TLabelframe.Label",
        background=BG,
        foreground=WARNING,
        font=FB,
        padding=(4, 2, 8, 2),
    )

    # ── Button ───────────────────────────────────────────────────────────────
    style.configure("TButton",
        background=PANEL2,
        foreground=FG,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        padding=(10, 6),
        relief="flat",
        font=F,
        focuscolor=ACCENT,
    )
    style.map("TButton",
        background=[("active", HOVER), ("pressed", SELECT), ("disabled", BG)],
        foreground=[("disabled", MUTED2)],
        bordercolor=[("active", ACCENT), ("focus", ACCENT)],
    )

    # Accent.TButton — teal главная кнопка
    style.configure("Accent.TButton",
        background=ACCENT,
        foreground="#ffffff",
        bordercolor=ACCENT2,
        lightcolor=ACCENT2,
        darkcolor=ACCENT3,
        padding=(16, 8),
        relief="flat",
        font=FB,
    )
    style.map("Accent.TButton",
        background=[("active", ACCENT2), ("pressed", ACCENT3), ("disabled", PANEL2)],
        foreground=[("disabled", MUTED)],
        bordercolor=[("active", ACCENT2)],
    )

    # Secondary.TButton
    style.configure("Secondary.TButton",
        background=BG,
        foreground=MUTED,
        bordercolor=BORDER2,
        lightcolor=BORDER2,
        darkcolor=BORDER2,
        padding=(8, 5),
        relief="flat",
        font=F,
    )
    style.map("Secondary.TButton",
        background=[("active", PANEL), ("pressed", PANEL2), ("disabled", BG)],
        foreground=[("active", FG), ("disabled", MUTED2)],
        bordercolor=[("active", BORDER)],
    )

    # Danger.TButton
    style.configure("Danger.TButton",
        background=BG,
        foreground=ERROR,
        bordercolor=ERROR,
        padding=(10, 6),
        relief="flat",
        font=F,
    )
    style.map("Danger.TButton",
        background=[("active", "#2d1010"), ("pressed", "#1a0808")],
        foreground=[("active", "#ff7070")],
    )

    # ── Entry / Combobox / Spinbox ───────────────────────────────────────────
    style.configure("TEntry",
        background=PANEL,
        fieldbackground=ENTRY_BG,
        foreground=FG,
        bordercolor=BORDER2,
        lightcolor=BORDER2,
        darkcolor=BORDER2,
        insertcolor=ACCENT2,
        padding=(8, 5),
        relief="flat",
    )
    style.map("TEntry",
        bordercolor=[("focus", ACCENT), ("hover", BORDER)],
        lightcolor=[("focus", ACCENT)],
        darkcolor=[("focus", ACCENT)],
    )

    style.configure("TCombobox",
        background=PANEL,
        fieldbackground=ENTRY_BG,
        foreground=FG,
        bordercolor=BORDER2,
        lightcolor=BORDER2,
        darkcolor=BORDER2,
        arrowcolor=MUTED,
        padding=(8, 5),
        relief="flat",
    )
    style.map("TCombobox",
        fieldbackground=[("readonly", ENTRY_BG)],
        foreground=[("readonly", FG)],
        bordercolor=[("focus", ACCENT), ("hover", BORDER)],
        lightcolor=[("focus", ACCENT)],
        arrowcolor=[("active", ACCENT2)],
    )

    style.configure("TSpinbox",
        background=PANEL,
        fieldbackground=ENTRY_BG,
        foreground=FG,
        bordercolor=BORDER2,
        lightcolor=BORDER2,
        darkcolor=BORDER2,
        arrowcolor=MUTED,
        padding=(6, 4),
        relief="flat",
    )
    style.map("TSpinbox",
        bordercolor=[("focus", ACCENT)],
        lightcolor=[("focus", ACCENT)],
        arrowcolor=[("active", ACCENT2)],
    )

    # ── Checkbutton / Radiobutton ────────────────────────────────────────────
    style.configure("TCheckbutton",
        background=BG,
        foreground=FG,
        indicatorcolor=ENTRY_BG,
        indicatormargin=6,
        padding=(4, 3),
    )
    style.map("TCheckbutton",
        background=[("active", BG), ("hover", BG)],
        indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT2)],
        foreground=[("disabled", MUTED2), ("active", FG)],
    )

    style.configure("Card.TCheckbutton",
        background=PANEL,
        foreground=FG,
        indicatorcolor=ENTRY_BG,
        indicatormargin=6,
        padding=(4, 3),
    )
    style.map("Card.TCheckbutton",
        background=[("active", PANEL)],
        indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT2)],
        foreground=[("disabled", MUTED2)],
    )

    style.configure("Card2.TCheckbutton",
        background=PANEL2,
        foreground=FG,
        indicatorcolor=ENTRY_BG,
        indicatormargin=6,
        padding=(4, 3),
    )
    style.map("Card2.TCheckbutton",
        background=[("active", PANEL2)],
        indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT2)],
        foreground=[("disabled", MUTED2)],
    )

    style.configure("TRadiobutton",
        background=BG,
        foreground=FG,
        indicatorcolor=ENTRY_BG,
        indicatormargin=6,
        padding=(4, 3),
    )
    style.map("TRadiobutton",
        background=[("active", BG)],
        indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT2)],
        foreground=[("disabled", MUTED2)],
    )

    style.configure("Card.TRadiobutton",
        background=PANEL,
        foreground=FG,
        indicatorcolor=ENTRY_BG,
        indicatormargin=6,
        padding=(4, 3),
    )
    style.map("Card.TRadiobutton",
        background=[("active", PANEL)],
        indicatorcolor=[("selected", ACCENT), ("pressed", ACCENT2)],
        foreground=[("disabled", MUTED2)],
    )

    # ── Notebook — pill-style tab bar ────────────────────────────────────────
    style.configure("TNotebook",
        background=BG,
        bordercolor=BORDER,
        tabmargins=(4, 6, 4, 0),
    )
    style.configure("TNotebook.Tab",
        background=PANEL2,
        foreground=MUTED,
        padding=(20, 9),
        font=F,
        bordercolor=BORDER2,
        lightcolor=BORDER2,
        darkcolor=BORDER2,
    )
    style.map("TNotebook.Tab",
        background=[("selected", PANEL)],
        foreground=[("selected", ACCENT2)],
        font=[("selected", FB)],
        expand=[("selected", (0, 2, 0, 0))],
        bordercolor=[("selected", ACCENT)],
        lightcolor=[("selected", ACCENT)],
        darkcolor=[("selected", ACCENT3)],
    )

    # ── Progressbar — teal ──────────────────────────────────────────────────
    style.configure("TProgressbar",
        background=ACCENT,
        troughcolor=ENTRY_BG,
        bordercolor=BORDER2,
        lightcolor=ACCENT2,
        darkcolor=ACCENT,
        thickness=10,
        relief="flat",
    )
    style.configure("Success.TProgressbar",
        background=SUCCESS,
        troughcolor=ENTRY_BG,
        thickness=10,
        relief="flat",
    )

    # ── Treeview ─────────────────────────────────────────────────────────────
    style.configure("Treeview",
        background=ENTRY_BG,
        fieldbackground=ENTRY_BG,
        foreground=FG,
        rowheight=30,
        bordercolor=BORDER,
        relief="flat",
        font=F,
    )
    style.configure("Treeview.Heading",
        background=PANEL,
        foreground=ACCENT2,
        relief="flat",
        bordercolor=BORDER,
        font=FB,
        padding=(10, 7),
    )
    style.map("Treeview",
        background=[("selected", SELECT)],
        foreground=[("selected", "#ffffff")],
    )
    style.map("Treeview.Heading",
        background=[("active", HOVER)],
        foreground=[("active", FG)],
    )

    # ── Scrollbar — тонкая ───────────────────────────────────────────────────
    style.configure("TScrollbar",
        background=PANEL,
        troughcolor=BG,
        bordercolor=BG,
        arrowcolor=BORDER,
        relief="flat",
        width=7,
        arrowsize=7,
    )
    style.map("TScrollbar",
        background=[("active", BORDER), ("pressed", ACCENT)],
        arrowcolor=[("active", MUTED)],
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    style.configure("TScale",
        background=BG,
        troughcolor=PANEL2,
        bordercolor=BORDER2,
        lightcolor=ACCENT,
        darkcolor=ACCENT,
        sliderlength=18,
        sliderrelief="flat",
    )
    style.map("TScale", background=[("active", BG)])
    style.configure("Card.TScale",
        background=PANEL,
        troughcolor=PANEL2,
        bordercolor=BORDER2,
    )

    # ── Separator ─────────────────────────────────────────────────────────────
    style.configure("TSeparator",        background=BORDER2)
    style.configure("Accent.TSeparator", background=ACCENT)

    # ── Panedwindow / Sash ────────────────────────────────────────────────────
    style.configure("TPanedwindow", background=BG)
    style.configure("Sash",
        sashthickness=5,
        gripcount=0,
        background=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
    )
