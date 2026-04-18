# -*- coding: utf-8 -*-
"""
ui_theme_ctk.py — палитры и темы для CustomTkinter.

Использование:
    import customtkinter as ctk
    from ui_theme_ctk import apply_theme, COLORS

    ctk.set_appearance_mode("dark")          # или "light"
    apply_theme("dark-teal")                  # "dark-teal" | "paper" | "amber-crt"
    root = ctk.CTk()

Палитра соответствует прототипу BankReasonTrainer.html.
"""
from __future__ import annotations

import json
import os
import tempfile
import customtkinter as ctk

# ── Палитры (1-в-1 из styles.css прототипа) ─────────────────────────
PALETTES = {
    "dark-teal": {
        "bg":      "#050f0c",
        "panel":   "#0b1c16",
        "panel2":  "#102418",
        "entry":   "#040d0a",
        "fg":      "#e4f4ee",
        "muted":   "#6db39a",
        "border":  "#1c4a35",
        "border2": "#112d20",
        "accent":  "#00c896",
        "accent2": "#1de9b6",
        "accent3": "#009b75",
        "hover":   "#0d2b1f",
        "success": "#1de9b6",
        "warning": "#f0b429",
        "error":   "#ff5252",
        "select":  "#0a3d2a",
    },
    "paper": {
        "bg":      "#f5f3ee",
        "panel":   "#ffffff",
        "panel2":  "#faf7f1",
        "entry":   "#ffffff",
        "fg":      "#1a1a1a",
        "muted":   "#6b6356",
        "border":  "#d8d2c4",
        "border2": "#e8e3d6",
        "accent":  "#b85c2c",
        "accent2": "#d27040",
        "accent3": "#8c4520",
        "hover":   "#f0ece4",
        "success": "#4a7d4a",
        "warning": "#b8862c",
        "error":   "#b8392c",
        "select":  "#f3e6dc",
    },
    "amber-crt": {
        "bg":      "#160a02",
        "panel":   "#1f1004",
        "panel2":  "#2a1707",
        "entry":   "#100600",
        "fg":      "#ffd9a3",
        "muted":   "#c89060",
        "border":  "#4a2a10",
        "border2": "#2e1908",
        "accent":  "#ff8a1f",
        "accent2": "#ffb060",
        "accent3": "#cc6a10",
        "hover":   "#2a1707",
        "success": "#ffb060",
        "warning": "#ffd060",
        "error":   "#ff5a3a",
        "select":  "#3a1f0a",
    },
}

COLORS: dict = PALETTES["dark-teal"]  # текущая активная палитра


def _font_family() -> str:
    """Лучший доступный шрифт в системе."""
    try:
        from tkinter import font as tkfont
        avail = set(tkfont.families())
        for name in ("SF Pro Text", "Segoe UI Variable Text", "Segoe UI",
                     "Helvetica Neue", "Helvetica", "Arial"):
            if name in avail:
                return name
    except Exception:
        pass
    return "Segoe UI"


def _build_ctk_theme_json(p: dict) -> dict:
    """
    Собирает CustomTkinter-совместимый theme dict.
    CTk требует пары (light, dark) для каждого цвета — для одной темы дублируем.
    """
    pair = lambda a, b=None: [a, b or a]
    return {
        "CTk": {
            "fg_color": pair(p["bg"]),
        },
        "CTkToplevel": {
            "fg_color": pair(p["bg"]),
        },
        "CTkFrame": {
            "corner_radius": 10,
            "border_width": 1,
            "fg_color": pair(p["panel"]),
            "top_fg_color": pair(p["bg"]),
            "border_color": pair(p["border2"]),
        },
        "CTkButton": {
            "corner_radius": 7,
            "border_width": 0,
            "fg_color": pair(p["panel2"]),
            "hover_color": pair(p["hover"]),
            "border_color": pair(p["border"]),
            "text_color": pair(p["fg"]),
            "text_color_disabled": pair(p["muted"]),
        },
        "CTkLabel": {
            "corner_radius": 0,
            "fg_color": pair("transparent"),
            "text_color": pair(p["fg"]),
        },
        "CTkEntry": {
            "corner_radius": 7,
            "border_width": 1,
            "fg_color": pair(p["entry"]),
            "border_color": pair(p["border2"]),
            "text_color": pair(p["fg"]),
            "placeholder_text_color": pair(p["muted"]),
        },
        "CTkCheckbox": {
            "corner_radius": 4,
            "border_width": 1,
            "fg_color": pair(p["accent"]),
            "border_color": pair(p["border"]),
            "hover_color": pair(p["accent2"]),
            "checkmark_color": pair(p["bg"]),
            "text_color": pair(p["fg"]),
        },
        "CTkRadiobutton": {
            "corner_radius": 1000,
            "border_width_checked": 5,
            "border_width_unchecked": 1,
            "fg_color": pair(p["accent"]),
            "border_color": pair(p["border"]),
            "hover_color": pair(p["accent2"]),
            "text_color": pair(p["fg"]),
        },
        "CTkProgressBar": {
            "corner_radius": 1000,
            "border_width": 0,
            "fg_color": pair(p["entry"]),
            "progress_color": pair(p["accent"]),
            "border_color": pair(p["border2"]),
        },
        "CTkSlider": {
            "corner_radius": 1000,
            "button_corner_radius": 1000,
            "border_width": 0,
            "button_length": 0,
            "fg_color": pair(p["entry"]),
            "progress_color": pair(p["accent"]),
            "button_color": pair(p["accent2"]),
            "button_hover_color": pair(p["accent"]),
        },
        "CTkOptionMenu": {
            "corner_radius": 7,
            "fg_color": pair(p["entry"]),
            "button_color": pair(p["panel2"]),
            "button_hover_color": pair(p["hover"]),
            "text_color": pair(p["fg"]),
            "text_color_disabled": pair(p["muted"]),
        },
        "CTkComboBox": {
            "corner_radius": 7,
            "border_width": 1,
            "fg_color": pair(p["entry"]),
            "border_color": pair(p["border2"]),
            "button_color": pair(p["panel2"]),
            "button_hover_color": pair(p["hover"]),
            "text_color": pair(p["fg"]),
            "text_color_disabled": pair(p["muted"]),
        },
        "CTkScrollbar": {
            "corner_radius": 1000,
            "border_spacing": 4,
            "fg_color": pair("transparent"),
            "button_color": pair(p["border"]),
            "button_hover_color": pair(p["accent3"]),
        },
        "CTkSegmentedButton": {
            "corner_radius": 8,
            "border_width": 1,
            "fg_color": pair(p["entry"]),
            "selected_color": pair(p["accent"]),
            "selected_hover_color": pair(p["accent2"]),
            "unselected_color": pair(p["entry"]),
            "unselected_hover_color": pair(p["hover"]),
            "text_color": pair(p["fg"]),
            "text_color_disabled": pair(p["muted"]),
        },
        "CTkTextbox": {
            "corner_radius": 8,
            "border_width": 1,
            "fg_color": pair(p["entry"]),
            "border_color": pair(p["border2"]),
            "text_color": pair(p["fg"]),
            "scrollbar_button_color": pair(p["border"]),
            "scrollbar_button_hover_color": pair(p["accent3"]),
        },
        "CTkScrollableFrame": {
            "label_fg_color": pair(p["panel2"]),
        },
        "CTkFont": {
            "macOS":   {"family": _font_family(), "size": 13, "weight": "normal"},
            "Windows": {"family": _font_family(), "size": 13, "weight": "normal"},
            "Linux":   {"family": _font_family(), "size": 13, "weight": "normal"},
        },
    }


def apply_theme(name: str = "dark-teal") -> dict:
    """
    Применяет тему. Должен быть вызван ДО создания ctk.CTk().
    Возвращает текущую палитру (dict).
    """
    global COLORS
    if name not in PALETTES:
        raise ValueError(f"Unknown theme: {name}. Choose: {list(PALETTES)}")
    COLORS = PALETTES[name]

    # appearance — light для paper, dark для остального
    ctk.set_appearance_mode("light" if name == "paper" else "dark")

    # CTk загружает темы из файла → пишем во временный JSON
    theme_dict = _build_ctk_theme_json(COLORS)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="ctk_theme_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(theme_dict, f, ensure_ascii=False)
    ctk.set_default_color_theme(path)
    return COLORS


# ── Хелперы для типографики ──────────────────────────────────────────
def font_xs():        return ctk.CTkFont(family=_font_family(), size=11)
def font_sm():        return ctk.CTkFont(family=_font_family(), size=12)
def font_base():      return ctk.CTkFont(family=_font_family(), size=13)
def font_md_bold():   return ctk.CTkFont(family=_font_family(), size=14, weight="bold")
def font_lg_bold():   return ctk.CTkFont(family=_font_family(), size=18, weight="bold")
def font_xl_bold():   return ctk.CTkFont(family=_font_family(), size=22, weight="bold")
def font_label():     return ctk.CTkFont(family=_font_family(), size=11, weight="bold")
def font_mono():      return ctk.CTkFont(family="JetBrains Mono", size=11)
