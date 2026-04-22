# -*- coding: utf-8 -*-
"""Theme layer for the Voilà dashboard.

The UI ships three palettes — ``dark-teal`` (default), ``paper`` (light)
and ``amber-crt`` (retro) — mirroring the ``ctk_migration_pack/
BankReasonTrainer.html`` prototype. Everything is pure CSS + HTML
strings; we don't introduce custom JS or Voilà templates.

Public API:

- ``PALETTES``              — dict of palette-name → colour dict
- ``apply_theme(name)``     — switch the active palette (no redraw)
- ``rebuild_css() -> str``  — regenerate CSS for the active palette
- ``inject_css()``          — return an ``ipywidgets.HTML`` carrying
                              the active CSS; call once at build time
- ``get_active_theme()``    — current palette name (for persistence)

The module-level constants ``BG``, ``PANEL``, ``FG``, ``ACCENT`` etc.
are **default-only aliases** from ``PALETTES["dark-teal"]``. They are
kept for backwards compatibility with the pre-Sprint-2 call-sites that
inline-style HTML snippets. Fresh code should prefer CSS classes so
theme-switching works without a page reload.
"""
from __future__ import annotations

from typing import Any, Iterable


# ── Palettes (source: ctk_migration_pack/ui_theme_ctk.py) ─────────────
# 17 keys per palette. `muted2` and `entry_bg` are aliases added on top
# of the CTk palette to match the existing CSS variable names.
PALETTES: dict[str, dict[str, str]] = {
    "dark-teal": {
        "bg":       "#050f0c",
        "panel":    "#0b1c16",
        "panel2":   "#102418",
        "entry_bg": "#040d0a",
        "fg":       "#e4f4ee",
        "muted":    "#6db39a",
        "muted2":   "#3e7a62",
        "border":   "#1c4a35",
        "border2":  "#112d20",
        "accent":   "#00c896",
        "accent2":  "#1de9b6",
        "accent3":  "#009b75",
        "hover":    "#0d2b1f",
        "success":  "#1de9b6",
        "warning":  "#f0b429",
        "error":    "#ff5252",
        "select":   "#0a3d2a",
    },
    "paper": {
        "bg":       "#f5f3ee",
        "panel":    "#ffffff",
        "panel2":   "#faf7f1",
        "entry_bg": "#ffffff",
        "fg":       "#1a1a1a",
        "muted":    "#6b6356",
        "muted2":   "#4a4238",
        "border":   "#d8d2c4",
        "border2":  "#e8e3d6",
        "accent":   "#b85c2c",
        "accent2":  "#d27040",
        "accent3":  "#8c4520",
        "hover":    "#f0ece4",
        "success":  "#4a7d4a",
        "warning":  "#b8862c",
        "error":    "#b8392c",
        "select":   "#f3e6dc",
    },
    "amber-crt": {
        "bg":       "#160a02",
        "panel":    "#1f1004",
        "panel2":   "#2a1707",
        "entry_bg": "#100600",
        "fg":       "#ffd9a3",
        "muted":    "#c89060",
        "muted2":   "#8c6545",
        "border":   "#4a2a10",
        "border2":  "#2e1908",
        "accent":   "#ff8a1f",
        "accent2":  "#ffb060",
        "accent3":  "#cc6a10",
        "hover":    "#2a1707",
        "success":  "#ffb060",
        "warning":  "#ffd060",
        "error":    "#ff5a3a",
        "select":   "#3a1f0a",
    },
}

_DEFAULT_THEME = "dark-teal"
_ACTIVE_THEME: str = _DEFAULT_THEME


# ── Default-only aliases (see module docstring) ───────────────────────
# Kept so pre-Sprint-2 imports like ``from ui_widgets.theme import BG``
# keep working. These are frozen to the default palette; call-sites that
# need live theme-aware colours must use ``rebuild_css()`` + CSS classes.
_DEFAULT = PALETTES[_DEFAULT_THEME]
BG       = _DEFAULT["bg"]
PANEL    = _DEFAULT["panel"]
PANEL2   = _DEFAULT["panel2"]
FG       = _DEFAULT["fg"]
MUTED    = _DEFAULT["muted"]
MUTED2   = _DEFAULT["muted2"]
ENTRY_BG = _DEFAULT["entry_bg"]
BORDER   = _DEFAULT["border"]
BORDER2  = _DEFAULT["border2"]
ACCENT   = _DEFAULT["accent"]
ACCENT2  = _DEFAULT["accent2"]
ACCENT3  = _DEFAULT["accent3"]
SUCCESS  = _DEFAULT["success"]
WARNING  = _DEFAULT["warning"]
ERROR    = _DEFAULT["error"]


def get_active_theme() -> str:
    """Return the currently active palette name (for persistence)."""
    return _ACTIVE_THEME


def apply_theme(name: str) -> None:
    """Switch the active palette. No UI redraw — caller must refresh CSS.

    Typical usage in ``notebook_app.build_app``:

    >>> apply_theme("paper")
    >>> css_widget.value = rebuild_css()

    Raises ``ValueError`` if ``name`` is not a known palette.
    """
    global _ACTIVE_THEME
    if name not in PALETTES:
        raise ValueError(
            f"Unknown theme {name!r}. Available: {sorted(PALETTES)}"
        )
    _ACTIVE_THEME = name


def rebuild_css() -> str:
    """Return a ``<style>...</style>`` block for the active palette."""
    return _build_css(PALETTES[_ACTIVE_THEME])


def _build_css(p: dict[str, str]) -> str:
    """Render the CSS stylesheet for a given palette dict."""
    BG       = p["bg"]
    PANEL    = p["panel"]
    PANEL2   = p["panel2"]
    ENTRY_BG = p["entry_bg"]
    FG       = p["fg"]
    MUTED    = p["muted"]
    MUTED2   = p["muted2"]
    BORDER   = p["border"]
    BORDER2  = p["border2"]
    ACCENT   = p["accent"]
    ACCENT2  = p["accent2"]
    ACCENT3  = p["accent3"]
    SUCCESS  = p["success"]
    WARNING  = p["warning"]
    ERROR    = p["error"]

    return f"""
<style>
  /* ── page-level background ─────────────────────────────────────── */
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
    color: {BG} !important;
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
    color: {BG}; font-weight: 800; font-size: 13px;
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
  .brt-header-title .brand {{ color: {ACCENT2}; font-weight: 800; }}
  .brt-header-title .muted {{ color: {MUTED}; font-weight: 400; }}
  .brt-header-title .sub   {{ color: {MUTED}; }}

  .brt-theme-switcher {{
    display: flex;
    gap: 4px;
    margin: 10px 0 0 0;
    padding: 6px 4px 0 4px;
    border-top: 1px solid {BORDER2};
  }}
  .brt-theme-switcher .jupyter-button {{
    flex: 1 1 0;
    padding: 4px 0 !important;
    font-size: 10px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
  }}
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
  .brt-chip.ok      {{ color: {SUCCESS}; border-color: {ACCENT3}; }}
  .brt-chip.warn    {{ color: {WARNING}; border-color: #7a5a15; }}
  .brt-chip.err     {{ color: {ERROR};   border-color: #7a1f1f; }}
  .brt-chip.info    {{ color: {ACCENT2}; border-color: {ACCENT3}; }}
  .brt-chip.accent  {{ color: {ACCENT};  border-color: {ACCENT2}; font-weight: 600; }}

  .brt-field-label {{
    display: block;
    color: {MUTED};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 4px 0 2px 0;
  }}

  .brt-sep {{
    height: 0;
    border-top: 1px solid {BORDER2};
    margin: 10px 0;
  }}

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

  /* ── Phase 7: header-chip row, predictions table, badges, accordion ── */
  .brt-header-chip-row {{
    display: flex; flex-wrap: wrap; gap: 6px;
    padding: 0 14px 8px 18px;
    background: {BG};
    border-bottom: 1px solid {BORDER2};
  }}
  .brt-header-chip-row .brt-chip {{ margin: 0; }}

  .brt-header-actions {{
    display: flex; align-items: center; gap: 8px;
  }}

  .brt-overlay {{
    background: {PANEL};
    border: 1px solid {BORDER2};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 14px;
    min-height: 480px;
  }}
  .brt-overlay-title {{
    color: {ACCENT2}; font-weight: 700; font-size: 13px;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 10px;
  }}

  .brt-pred-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    color: {FG};
    background: {PANEL2};
    border-radius: 4px;
    overflow: hidden;
    margin-top: 6px;
  }}
  .brt-pred-table thead th {{
    background: {ENTRY_BG};
    color: {MUTED};
    font-weight: 700; text-transform: uppercase;
    font-size: 10px; letter-spacing: 1px;
    padding: 8px 10px;
    border-bottom: 1px solid {BORDER2};
    text-align: left;
  }}
  .brt-pred-table tbody td {{
    padding: 7px 10px;
    border-bottom: 1px solid {BORDER2};
    vertical-align: middle;
  }}
  .brt-pred-table tbody tr:hover {{
    background: {BORDER2};
  }}
  .brt-pred-text {{
    max-width: 520px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    color: {FG};
  }}

  .brt-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid {BORDER};
    background: {ENTRY_BG};
    color: {FG};
  }}
  .brt-badge-ok     {{ color: {SUCCESS}; border-color: {ACCENT3}; background: rgba(29,233,182,0.08); }}
  .brt-badge-warn   {{ color: {WARNING}; border-color: #7a5a15; background: rgba(240,180,41,0.08); }}
  .brt-badge-err    {{ color: {ERROR};   border-color: #7a1f1f; background: rgba(255,82,82,0.08); }}
  .brt-badge-info   {{ color: {ACCENT2}; border-color: {ACCENT3}; background: rgba(29,233,182,0.06); }}
  .brt-badge-accent {{ color: {ACCENT};  border-color: {ACCENT2}; background: rgba(0,200,150,0.10); font-weight: 800; }}

  .brt-filter-tabs {{
    display: flex; gap: 6px; margin: 6px 0 4px 0;
    border-bottom: 1px solid {BORDER2}; padding-bottom: 6px;
  }}

  /* override jupyter accordion to match active palette */
  .jupyter-widgets.widget-accordion .p-Collapse-header,
  .jupyter-widgets.widget-accordion .lm-Collapse-header {{
    background: {PANEL2} !important;
    color: {ACCENT2} !important;
    border: 1px solid {BORDER2} !important;
    font-weight: 700; letter-spacing: 1.0px; text-transform: uppercase;
    font-size: 11px;
  }}
  .jupyter-widgets.widget-accordion .p-Collapse-contents,
  .jupyter-widgets.widget-accordion .lm-Collapse-contents {{
    background: {PANEL} !important;
    color: {FG} !important;
    border: 1px solid {BORDER2} !important;
    border-top: none !important;
  }}
</style>
"""


def inject_css() -> Any:
    """Return an ``ipywidgets.HTML`` widget carrying the active CSS.

    Include this once at the top of ``build_app()``. When the user
    switches themes at runtime, assign ``rebuild_css()`` to the widget's
    ``value`` to re-render without reloading the page.
    """
    import ipywidgets as w
    return w.HTML(value=rebuild_css())


def section_header(title: str, subtitle: str = "") -> Any:
    """Uppercase accent section title + optional muted subtitle."""
    import ipywidgets as w
    sub_html = f"<div class='brt-card-sub'>{subtitle}</div>" if subtitle else ""
    return w.HTML(
        f"<div class='brt-card-title'>{title}</div>{sub_html}"
    )


_CHIP_KINDS: frozenset[str] = frozenset(
    {"default", "accent", "ok", "warn", "err", "info"}
)


def chip(text: str, kind: str = "default") -> str:
    """Return an HTML snippet for a single chip badge.

    ``kind`` ∈ ``{'default', 'accent', 'ok', 'warn', 'err', 'info'}``.
    Unknown kinds degrade silently to ``'default'`` so stale call-sites
    never crash the UI.
    """
    if kind not in _CHIP_KINDS:
        kind = "default"
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


def section_card(
    title: str,
    children: Iterable[Any],
    subtitle: str = "",
    right: Any = None,
) -> Any:
    """VBox with an accent section header + body children, styled as a card.

    When ``right`` is a widget (typically a ``Button`` or ``HBox``), it is
    rendered on the header's right edge via an HBox; title/subtitle stay
    left-aligned and expand to fill available width.
    """
    import ipywidgets as w
    head = section_header(title, subtitle)
    if right is not None:
        head = w.HBox(
            [head, right],
            layout=w.Layout(
                justify_content="space-between",
                align_items="center",
                width="100%",
            ),
        )
    return w.VBox([head, *children], layout=card_layout())


def field_label(text: str) -> Any:
    """Uppercase muted label for a form field (CSS class ``brt-field-label``)."""
    import ipywidgets as w
    return w.HTML(f"<span class='brt-field-label'>{text}</span>")


def separator() -> Any:
    """Thin horizontal divider (CSS class ``brt-sep``)."""
    import ipywidgets as w
    return w.HTML("<div class='brt-sep'></div>")


def status_badge(text: str = "idle") -> str:
    return f"<span class='brt-status'>{text}</span>"


_BADGE_KINDS: frozenset[str] = frozenset(
    {"default", "accent", "ok", "warn", "err", "info"}
)


def badge(text: str, kind: str = "ok") -> str:
    """HTML snippet for a small pill badge.

    ``kind`` ∈ ``{'default', 'accent', 'ok', 'warn', 'err', 'info'}``.
    Unknown kinds degrade silently to ``'default'``.
    """
    if kind not in _BADGE_KINDS:
        kind = "default"
    cls = "brt-badge" if kind == "default" else f"brt-badge brt-badge-{kind}"
    return f"<span class='{cls}'>{text}</span>"


def header_chip_row(chip_html: Iterable[str]) -> Any:
    """Container for a horizontal row of chip badges (header strip)."""
    import ipywidgets as w
    inner = "".join(chip_html)
    return w.HTML(f"<div class='brt-header-chip-row'>{inner}</div>")


def overlay_card(title: str, body_html: str) -> Any:
    """Full-width card used for sidebar context dialogs (history/artifacts/settings)."""
    import ipywidgets as w
    return w.HTML(
        f"<div class='brt-overlay'>"
        f"<div class='brt-overlay-title'>{title}</div>"
        f"{body_html}"
        f"</div>"
    )
