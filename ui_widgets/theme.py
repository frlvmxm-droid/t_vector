# -*- coding: utf-8 -*-
"""Dark-Teal theme for the Voilà dashboard.

Palette and card aesthetics mirror ``ui_theme.py`` (CTk / Tkinter path)
so the web UI reads as the same product. Everything is pure CSS + HTML
strings — we don't introduce custom JS or Voilà templates.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


# ── palette (mirrors ui_theme.py) ───────────────────────────────────────
BG       = "#050f0c"
PANEL    = "#0b1c16"
PANEL2   = "#102418"
FG       = "#e4f4ee"
MUTED    = "#6db39a"
MUTED2   = "#3e7a62"
ENTRY_BG = "#040d0a"
BORDER   = "#1c4a35"
BORDER2  = "#112d20"
ACCENT   = "#00c896"
ACCENT2  = "#1de9b6"
ACCENT3  = "#009b75"
SUCCESS  = "#1de9b6"
WARNING  = "#f0b429"
ERROR    = "#ff5252"


_CSS = f"""
<style>
  /* ── page-level dark background ─────────────────────────────────── */
  body, .jp-Notebook, .jp-Cell, .jp-OutputArea, #rendered_cells,
  .vl-OutputArea, .voila-container {{
    background: {BG} !important;
    color: {FG} !important;
  }}
  .jp-Cell {{ padding: 0 !important; }}
  .jp-OutputArea-output pre {{
    background: {ENTRY_BG} !important;
    color: {FG} !important;
  }}

  /* ── widget labels / text ──────────────────────────────────────── */
  .widget-label, .widget-inline-hbox > .widget-label,
  .widget-html-content, .jupyter-widgets label {{
    color: {FG} !important;
  }}
  .widget-html-content code {{
    background: {PANEL2};
    color: {ACCENT2};
    padding: 1px 5px;
    border-radius: 3px;
  }}

  /* ── text inputs / textarea ────────────────────────────────────── */
  .widget-text input, .widget-textarea textarea {{
    background: {ENTRY_BG} !important;
    color: {FG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
  }}
  .widget-text input:focus, .widget-textarea textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 1px {ACCENT2}33 !important;
  }}

  /* ── dropdown ──────────────────────────────────────────────────── */
  .widget-dropdown > select {{
    background: {ENTRY_BG} !important;
    color: {FG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
  }}

  /* ── sliders ───────────────────────────────────────────────────── */
  .ui-slider {{
    background: {PANEL2} !important;
    border: 1px solid {BORDER} !important;
  }}
  .ui-slider .ui-slider-range {{
    background: {ACCENT} !important;
  }}
  .ui-slider .ui-slider-handle {{
    background: {ACCENT2} !important;
    border: 1px solid {ACCENT3} !important;
  }}

  /* ── buttons ───────────────────────────────────────────────────── */
  .jupyter-widgets button.jupyter-button {{
    background: {PANEL2} !important;
    color: {FG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    font-weight: 600;
  }}
  .jupyter-widgets button.jupyter-button:hover {{
    background: {BORDER} !important;
    border-color: {ACCENT3} !important;
  }}
  .jupyter-widgets button.mod-primary {{
    background: {ACCENT3} !important;
    color: #04140e !important;
    border-color: {ACCENT} !important;
  }}
  .jupyter-widgets button.mod-primary:hover {{
    background: {ACCENT} !important;
  }}

  /* ── progress bar ──────────────────────────────────────────────── */
  .progress {{ background: {PANEL2} !important; }}
  .progress .progress-bar {{ background: {ACCENT} !important; }}
  .progress-bar.bg-success {{ background: {SUCCESS} !important; }}
  .progress-bar.bg-danger  {{ background: {ERROR}   !important; }}

  /* ── checkbox ──────────────────────────────────────────────────── */
  .widget-checkbox input[type="checkbox"] {{
    accent-color: {ACCENT};
  }}

  /* ── FileUpload button ─────────────────────────────────────────── */
  .widget-upload > .jupyter-button {{
    background: {PANEL2} !important;
    color: {FG} !important;
    border: 1px dashed {BORDER} !important;
  }}

  /* ── custom helpers used by Python code ────────────────────────── */
  .brt-app {{
    font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: {FG};
  }}
  .brt-sidebar {{
    background: {PANEL};
    border-right: 1px solid {BORDER2};
    padding: 14px 10px;
    min-width: 220px;
  }}
  .brt-brand {{
    display: flex; align-items: center; gap: 10px;
    padding: 4px 6px 14px 6px;
    border-bottom: 1px solid {BORDER2};
    margin-bottom: 10px;
  }}
  .brt-brand-badge {{
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient(135deg, {ACCENT3}, {ACCENT2});
    color: #04140e; font-weight: 800; font-size: 13px;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: 0.5px;
  }}
  .brt-brand-name {{ font-size: 14px; font-weight: 700; color: {FG}; line-height: 1.15; }}
  .brt-brand-sub  {{ font-size: 10px; color: {MUTED}; letter-spacing: 1.5px; }}

  .brt-nav-section {{
    color: {MUTED2}; font-size: 10px; letter-spacing: 1.5px;
    font-weight: 700; padding: 14px 6px 6px 6px;
  }}

  .brt-hw-card {{
    margin-top: 16px;
    background: {PANEL2};
    border: 1px solid {BORDER2};
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 11px; color: {MUTED};
    line-height: 1.7;
  }}
  .brt-hw-card b {{ color: {FG}; }}

  .brt-footer {{
    margin-top: auto; padding: 12px 6px 4px 6px;
    color: {MUTED2}; font-size: 10px;
    border-top: 1px solid {BORDER2};
  }}

  .brt-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px 12px 18px;
    background: {BG};
    border-bottom: 1px solid {BORDER2};
  }}
  .brt-header-title {{
    font-size: 13px; color: {FG}; font-weight: 600;
  }}
  .brt-header-title .muted {{ color: {MUTED}; font-weight: 400; }}
  .brt-status {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: {MUTED}; text-transform: lowercase;
  }}
  .brt-status::before {{
    content: ""; width: 8px; height: 8px; border-radius: 50%;
    background: {ACCENT};
  }}

  .brt-card {{
    background: {PANEL};
    border: 1px solid {BORDER2};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 14px;
  }}
  .brt-card-title {{
    color: {ACCENT2}; font-size: 11px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
  }}
  .brt-card-sub {{
    color: {MUTED}; font-size: 12px; margin-top: 2px;
  }}

  .brt-chip {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    border: 1px solid {BORDER};
    background: {PANEL2};
    color: {FG};
    margin-right: 6px; margin-top: 4px;
  }}
  .brt-chip.ok    {{ color: {SUCCESS}; border-color: {ACCENT3}; }}
  .brt-chip.warn  {{ color: {WARNING}; border-color: #7a5a15; }}
  .brt-chip.err   {{ color: {ERROR};   border-color: #7a1f1f; }}
  .brt-chip.info  {{ color: {ACCENT2}; border-color: {ACCENT3}; }}

  .brt-metric {{
    flex: 1 1 0;
    background: {PANEL2};
    border: 1px solid {BORDER2};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 0 6px;
  }}
  .brt-metric-label {{
    color: {MUTED}; font-size: 11px;
    letter-spacing: 1.5px; text-transform: uppercase;
    font-weight: 700;
  }}
  .brt-metric-value {{
    color: {FG}; font-size: 28px; font-weight: 700;
    margin-top: 6px;
  }}
  .brt-metric-sub {{
    color: {MUTED}; font-size: 11px; margin-top: 4px;
  }}

  /* Fix widgets default light borders/backgrounds */
  .widget-container, .widget-hbox, .widget-vbox {{
    background: transparent !important;
  }}
  .jupyter-widgets-output-area {{
    background: {ENTRY_BG} !important;
    color: {FG} !important;
    border-radius: 4px;
  }}
</style>
"""


def inject_css() -> Any:
    """Return an ``ipywidgets.HTML`` widget carrying the global dark-teal CSS.

    Include this once at the top of ``build_app()``.
    """
    import ipywidgets as w
    return w.HTML(value=_CSS)


def section_header(title: str, subtitle: str = "") -> Any:
    """Uppercase teal section title + optional muted subtitle."""
    import ipywidgets as w
    sub_html = f"<div class='brt-card-sub'>{subtitle}</div>" if subtitle else ""
    return w.HTML(
        f"<div class='brt-card-title'>{title}</div>{sub_html}"
    )


def chip(text: str, kind: str = "default") -> str:
    """Return an HTML snippet for a single chip badge.

    ``kind`` ∈ ``{'default', 'ok', 'warn', 'err', 'info'}``.
    """
    cls = "brt-chip" if kind == "default" else f"brt-chip {kind}"
    return f"<span class='{cls}'>{text}</span>"


def chips_row(items: Iterable[str]) -> Any:
    """Wrap pre-built chip snippets in a paragraph."""
    import ipywidgets as w
    return w.HTML("<div>" + "".join(items) + "</div>")


def metric_card(label: str, value: str, sub: str = "") -> str:
    """HTML for a single metric card (big number + label + sub)."""
    sub_html = f"<div class='brt-metric-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='brt-metric'>"
        f"<div class='brt-metric-label'>{label}</div>"
        f"<div class='brt-metric-value'>{value}</div>"
        f"{sub_html}"
        f"</div>"
    )


def metric_row(cards_html: Iterable[str]) -> Any:
    """Return an HTML widget with metric cards aligned in a flex row."""
    import ipywidgets as w
    inner = "".join(cards_html)
    return w.HTML(
        f"<div style='display:flex; gap:12px; margin:0 8px;'>{inner}</div>"
    )


def card_layout() -> Any:
    """``ipywidgets.Layout`` mimicking the ``.brt-card`` style on VBox/HBox."""
    import ipywidgets as w
    return w.Layout(
        border=f"1px solid {BORDER2}",
        padding="10px 14px",
        margin="8px 14px",
    )


def section_card(title: str, children: Iterable[Any], subtitle: str = "") -> Any:
    """VBox with a teal section header + body children, styled as a card."""
    import ipywidgets as w
    head = section_header(title, subtitle)
    return w.VBox([head, *children], layout=card_layout())


def status_badge(text: str = "idle") -> str:
    return f"<span class='brt-status'>{text}</span>"
