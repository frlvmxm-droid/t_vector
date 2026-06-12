# -*- coding: utf-8 -*-
"""Three-theme palette + CSS for the Voilà dashboard.

Mirrors ``ctk_migration_pack/ui_theme_ctk.py`` (Dark Teal / Paper /
Amber-CRT) — pure CSS, no custom JS or Voilà templates. The active
palette can be swapped at runtime via :func:`apply_theme`; the CSS is
regenerated and pushed to the shared HTML widget returned by
:func:`inject_css`, so the UI re-themes in-place without a kernel
restart.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


# ── Palettes (1-to-1 with ctk_migration_pack/ui_theme_ctk.py) ──────────
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
        "success":  "#1de9b6",
        "warning":  "#f0b429",
        "error":    "#ff5252",
        "warn_border": "#7a5a15",
        "err_border":  "#7a1f1f",
        "primary_text": "#04140e",
    },
    "paper": {
        "bg":       "#f5f3ee",
        "panel":    "#ffffff",
        "panel2":   "#faf7f1",
        "entry_bg": "#ffffff",
        "fg":       "#1a1a1a",
        "muted":    "#6b6356",
        "muted2":   "#8c8275",
        "border":   "#d8d2c4",
        "border2":  "#e8e3d6",
        "accent":   "#b85c2c",
        "accent2":  "#d27040",
        "accent3":  "#8c4520",
        "success":  "#4a7d4a",
        "warning":  "#b8862c",
        "error":    "#b8392c",
        "warn_border": "#cfa55a",
        "err_border":  "#c97a6f",
        "primary_text": "#ffffff",
    },
    "amber-crt": {
        "bg":       "#160a02",
        "panel":    "#1f1004",
        "panel2":   "#2a1707",
        "entry_bg": "#100600",
        "fg":       "#ffd9a3",
        "muted":    "#c89060",
        "muted2":   "#8a6038",
        "border":   "#4a2a10",
        "border2":  "#2e1908",
        "accent":   "#ff8a1f",
        "accent2":  "#ffb060",
        "accent3":  "#cc6a10",
        "success":  "#ffb060",
        "warning":  "#ffd060",
        "error":    "#ff5a3a",
        "warn_border": "#8a6028",
        "err_border":  "#8a3320",
        "primary_text": "#160a02",
    },
}

DEFAULT_THEME = "dark-teal"

# Module-level live palette — mutated by ``apply_theme``. Code that
# reads colors at request time (e.g. dynamic HTML in notebook_app)
# should read ``COLORS["accent2"]`` rather than caching a constant.
COLORS: dict[str, str] = dict(PALETTES[DEFAULT_THEME])

# Track the active theme name for session save/restore.
_CURRENT_THEME: str = DEFAULT_THEME

# Reference to the live CSS widget returned by ``inject_css``. Updated
# in-place by ``apply_theme`` so users see the new palette without a
# page reload.
_css_widget_ref: Optional[Any] = None


def _build_css(p: dict[str, str]) -> str:
    """Render the global stylesheet for a single palette dict."""
    return f"""
<style>
  body, .jp-Notebook, .jp-Cell, .jp-OutputArea, #rendered_cells,
  .vl-OutputArea, .voila-container {{
    background: {p['bg']} !important;
    color: {p['fg']} !important;
  }}
  .jp-Cell {{ padding: 0 !important; }}
  .jp-OutputArea-output pre {{
    background: {p['entry_bg']} !important;
    color: {p['fg']} !important;
  }}

  .widget-label, .widget-inline-hbox > .widget-label,
  .widget-html-content, .jupyter-widgets label {{
    color: {p['fg']} !important;
  }}
  .widget-html-content code {{
    background: {p['panel2']};
    color: {p['accent2']};
    padding: 1px 5px;
    border-radius: 3px;
  }}

  .widget-text input, .widget-textarea textarea {{
    background: {p['entry_bg']} !important;
    color: {p['fg']} !important;
    border: 1px solid {p['border']} !important;
    border-radius: 4px !important;
  }}
  .widget-text input:focus, .widget-textarea textarea:focus {{
    border-color: {p['accent']} !important;
    box-shadow: 0 0 0 1px {p['accent2']}33 !important;
  }}

  .widget-dropdown > select {{
    background: {p['entry_bg']} !important;
    color: {p['fg']} !important;
    border: 1px solid {p['border']} !important;
    border-radius: 4px !important;
  }}

  .ui-slider {{
    background: {p['panel2']} !important;
    border: 1px solid {p['border']} !important;
  }}
  .ui-slider .ui-slider-range {{
    background: {p['accent']} !important;
  }}
  .ui-slider .ui-slider-handle {{
    background: {p['accent2']} !important;
    border: 1px solid {p['accent3']} !important;
  }}

  .jupyter-widgets button.jupyter-button {{
    background: {p['panel2']} !important;
    color: {p['fg']} !important;
    border: 1px solid {p['border']} !important;
    border-radius: 4px !important;
    font-weight: 600;
  }}
  .jupyter-widgets button.jupyter-button:hover {{
    background: {p['border']} !important;
    border-color: {p['accent3']} !important;
  }}
  .jupyter-widgets button.mod-primary {{
    background: {p['accent3']} !important;
    color: {p['primary_text']} !important;
    border-color: {p['accent']} !important;
  }}
  .jupyter-widgets button.mod-primary:hover {{
    background: {p['accent']} !important;
  }}

  .progress {{ background: {p['panel2']} !important; }}
  .progress .progress-bar {{ background: {p['accent']} !important; }}
  .progress-bar.bg-success {{ background: {p['success']} !important; }}
  .progress-bar.bg-danger  {{ background: {p['error']}   !important; }}

  .widget-checkbox input[type="checkbox"] {{
    accent-color: {p['accent']};
  }}

  .widget-upload > .jupyter-button {{
    background: {p['panel2']} !important;
    color: {p['fg']} !important;
    border: 1px dashed {p['border']} !important;
  }}

  .brt-app {{
    font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: {p['fg']};
  }}
  .brt-sidebar {{
    background: {p['panel']};
    border-right: 1px solid {p['border2']};
    padding: 14px 10px;
    min-width: 220px;
  }}
  .brt-brand {{
    display: flex; align-items: center; gap: 10px;
    padding: 4px 6px 14px 6px;
    border-bottom: 1px solid {p['border2']};
    margin-bottom: 10px;
  }}
  .brt-brand-badge {{
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient(135deg, {p['accent3']}, {p['accent2']});
    color: {p['primary_text']}; font-weight: 800; font-size: 13px;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: 0.5px;
  }}
  .brt-brand-name {{ font-size: 14px; font-weight: 700; color: {p['fg']}; line-height: 1.15; }}
  .brt-brand-sub  {{ font-size: 10px; color: {p['muted']}; letter-spacing: 1.5px; }}

  .brt-nav-section {{
    color: {p['muted2']}; font-size: 10px; letter-spacing: 1.5px;
    font-weight: 700; padding: 14px 6px 6px 6px;
  }}

  .brt-hw-card {{
    margin-top: 16px;
    background: {p['panel2']};
    border: 1px solid {p['border2']};
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 11px; color: {p['muted']};
    line-height: 1.7;
  }}
  .brt-hw-card b {{ color: {p['fg']}; }}

  .brt-theme-switch {{
    margin-top: 12px;
    padding: 8px 4px 6px 4px;
    border-top: 1px solid {p['border2']};
  }}
  .brt-theme-switch-label {{
    color: {p['muted2']}; font-size: 9px; letter-spacing: 1.5px;
    font-weight: 700; text-transform: uppercase;
    padding: 0 6px 6px 6px;
  }}

  .brt-footer {{
    margin-top: auto; padding: 12px 6px 4px 6px;
    color: {p['muted2']}; font-size: 10px;
    border-top: 1px solid {p['border2']};
  }}

  .brt-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px 12px 18px;
    background: {p['bg']};
    border-bottom: 1px solid {p['border2']};
  }}
  .brt-header-title {{
    font-size: 13px; color: {p['fg']}; font-weight: 600;
  }}
  .brt-header-title .muted {{ color: {p['muted']}; font-weight: 400; }}
  .brt-status {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: {p['muted']}; text-transform: lowercase;
  }}
  .brt-status::before {{
    content: ""; width: 8px; height: 8px; border-radius: 50%;
    background: {p['accent']};
  }}

  .brt-card {{
    background: {p['panel']};
    border: 1px solid {p['border2']};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 14px;
  }}
  .brt-card-header {{
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 10px;
  }}
  .brt-card-title {{
    color: {p['accent2']}; font-size: 11px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
  }}
  .brt-card-sub {{
    color: {p['muted']}; font-size: 12px; margin-top: 2px;
  }}

  .brt-field-label {{
    color: {p['muted']}; font-size: 10px;
    letter-spacing: 1.4px; text-transform: uppercase;
    font-weight: 700;
    margin: 8px 0 4px 0;
  }}

  .brt-separator {{
    border-bottom: 1px solid {p['border2']};
    margin: 14px 0;
    height: 0;
  }}

  .brt-chip {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    border: 1px solid {p['border']};
    background: {p['panel2']};
    color: {p['fg']};
    margin-right: 6px; margin-top: 4px;
  }}
  .brt-chip.ok    {{ color: {p['success']}; border-color: {p['accent3']}; }}
  .brt-chip.warn  {{ color: {p['warning']}; border-color: {p['warn_border']}; }}
  .brt-chip.err   {{ color: {p['error']};   border-color: {p['err_border']}; }}
  .brt-chip.info  {{ color: {p['accent2']}; border-color: {p['accent3']}; }}
  .brt-chip.accent {{ color: {p['warning']}; border-color: {p['warning']};
                      background: {p['warning']}1f; font-weight: 700; }}

  .brt-metric {{
    flex: 1 1 0;
    background: {p['panel2']};
    border: 1px solid {p['border2']};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 0 6px;
  }}
  .brt-metric-label {{
    color: {p['muted']}; font-size: 11px;
    letter-spacing: 1.5px; text-transform: uppercase;
    font-weight: 700;
  }}
  .brt-metric-value {{
    color: {p['fg']}; font-size: 28px; font-weight: 700;
    margin-top: 6px;
  }}
  .brt-metric-sub {{
    color: {p['muted']}; font-size: 11px; margin-top: 4px;
  }}

  .widget-container, .widget-hbox, .widget-vbox {{
    background: transparent !important;
  }}
  .jupyter-widgets-output-area {{
    background: {p['entry_bg']} !important;
    color: {p['fg']} !important;
    border-radius: 4px;
  }}

  .brt-header-chip-row {{
    display: flex; flex-wrap: wrap; gap: 6px;
    padding: 0 14px 8px 18px;
    background: {p['bg']};
    border-bottom: 1px solid {p['border2']};
  }}
  .brt-header-chip-row .brt-chip {{ margin: 0; }}

  .brt-header-actions {{
    display: flex; align-items: center; gap: 8px;
  }}

  .brt-overlay {{
    background: {p['panel']};
    border: 1px solid {p['border2']};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 14px;
    min-height: 480px;
  }}
  .brt-overlay-title {{
    color: {p['accent2']}; font-weight: 700; font-size: 13px;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 10px;
  }}

  .brt-pred-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    color: {p['fg']};
    background: {p['panel2']};
    border-radius: 4px;
    overflow: hidden;
    margin-top: 6px;
  }}
  .brt-pred-table thead th {{
    background: {p['entry_bg']};
    color: {p['muted']};
    font-weight: 700; text-transform: uppercase;
    font-size: 10px; letter-spacing: 1px;
    padding: 8px 10px;
    border-bottom: 1px solid {p['border2']};
    text-align: left;
  }}
  .brt-pred-table tbody td {{
    padding: 7px 10px;
    border-bottom: 1px solid {p['border2']};
    vertical-align: middle;
  }}
  .brt-pred-table tbody tr:hover {{
    background: {p['border2']};
  }}
  .brt-pred-text {{
    max-width: 520px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    color: {p['fg']};
  }}

  .brt-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid {p['border']};
    background: {p['entry_bg']};
    color: {p['fg']};
  }}
  .brt-badge-ok   {{ color: {p['success']}; border-color: {p['accent3']}; background: {p['success']}14; }}
  .brt-badge-warn {{ color: {p['warning']}; border-color: {p['warn_border']}; background: {p['warning']}14; }}
  .brt-badge-err  {{ color: {p['error']};   border-color: {p['err_border']}; background: {p['error']}14; }}
  .brt-badge-info {{ color: {p['accent2']}; border-color: {p['accent3']}; background: {p['accent2']}0f; }}
  .brt-badge-accent {{ color: {p['warning']}; border-color: {p['warning']}; background: {p['warning']}1f; }}

  .brt-filter-tabs {{
    display: flex; gap: 6px; margin: 6px 0 4px 0;
    border-bottom: 1px solid {p['border2']}; padding-bottom: 6px;
  }}

  .jupyter-widgets.widget-accordion .p-Collapse-header,
  .jupyter-widgets.widget-accordion .lm-Collapse-header {{
    background: {p['panel2']} !important;
    color: {p['accent2']} !important;
    border: 1px solid {p['border2']} !important;
    font-weight: 700; letter-spacing: 1.0px; text-transform: uppercase;
    font-size: 11px;
  }}
  .jupyter-widgets.widget-accordion .p-Collapse-contents,
  .jupyter-widgets.widget-accordion .lm-Collapse-contents {{
    background: {p['panel']} !important;
    color: {p['fg']} !important;
    border: 1px solid {p['border2']} !important;
    border-top: none !important;
  }}
</style>
"""


# ── Runtime API ──────────────────────────────────────────────────────


def apply_theme(name: str) -> str:
    """Switch the active palette and re-render CSS in place.

    Returns the resolved theme name (after fallback to default for
    unknown keys, so that a corrupt ``last_session.json`` cannot break
    startup). Idempotent.

    Mutates the module-level ``COLORS`` dict in place rather than
    rebinding it, so existing ``from ui_widgets.theme import COLORS``
    consumers see the updated palette without re-importing.
    """
    global _CURRENT_THEME
    if name not in PALETTES:
        name = DEFAULT_THEME
    COLORS.clear()
    COLORS.update(PALETTES[name])
    _CURRENT_THEME = name
    if _css_widget_ref is not None:
        _css_widget_ref.value = _build_css(COLORS)
    return name


def current_theme() -> str:
    """Active palette name (``"dark-teal"`` / ``"paper"`` / ``"amber-crt"``)."""
    return _CURRENT_THEME


def inject_css() -> Any:
    """Return the shared HTML widget that carries the global stylesheet.

    Include it once at the top of ``build_app()``. Subsequent
    :func:`apply_theme` calls update this widget's ``.value`` so the
    UI re-themes without a kernel restart.
    """
    global _css_widget_ref
    import ipywidgets as w
    if _css_widget_ref is None:
        _css_widget_ref = w.HTML(value=_build_css(COLORS))
    else:
        _css_widget_ref.value = _build_css(COLORS)
    return _css_widget_ref


# ── HTML / Widget helpers ────────────────────────────────────────────


def section_header(title: str, subtitle: str = "") -> Any:
    """Uppercase teal section title + optional muted subtitle."""
    import ipywidgets as w
    sub_html = f"<div class='brt-card-sub'>{subtitle}</div>" if subtitle else ""
    return w.HTML(
        f"<div class='brt-card-title'>{title}</div>{sub_html}"
    )


def chip(text: str, kind: str = "default") -> str:
    """Return an HTML snippet for a single chip badge.

    ``kind`` ∈ ``{'default', 'ok', 'warn', 'err', 'info', 'accent'}``.
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
        border=f"1px solid {COLORS['border2']}",
        padding="10px 14px",
        margin="8px 14px",
    )


def section_card(
    title: str,
    children: Iterable[Any],
    subtitle: str = "",
    right: Optional[Any] = None,
) -> Any:
    """VBox with a teal section header + body children, styled as a card.

    ``right`` (optional) — an ipywidgets widget placed in the header
    flex row at the right edge. Use it for action buttons (e.g. an
    "Inspect" or "Export" button that belongs to the card's section).
    """
    import ipywidgets as w
    title_block = section_header(title, subtitle)
    if right is not None:
        header: Any = w.HBox(
            [title_block, right],
            layout=w.Layout(
                justify_content="space-between",
                align_items="flex-start",
                width="100%",
            ),
        )
    else:
        header = title_block
    return w.VBox([header, *children], layout=card_layout())


def field_label(text: str) -> Any:
    """Small uppercase muted label, drawn above an editable field.

    Mirrors ``_field_label`` from ``ctk_migration_pack/app_train_view_ctk.py``.
    """
    import ipywidgets as w
    return w.HTML(f"<div class='brt-field-label'>{text}</div>")


def separator() -> Any:
    """Thin horizontal divider, palette-aware."""
    import ipywidgets as w
    return w.HTML("<div class='brt-separator'></div>")


def status_badge(text: str = "idle") -> str:
    return f"<span class='brt-status'>{text}</span>"


def badge(text: str, kind: str = "ok") -> str:
    """HTML snippet for a small pill badge.

    ``kind`` ∈ ``{'ok', 'warn', 'err', 'info', 'accent', 'default'}``.
    """
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
