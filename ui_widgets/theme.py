# -*- coding: utf-8 -*-
"""Themed CSS for the Voilà dashboard with hot-swappable palettes.

Three palettes mirror ``ctk_migration_pack/ui_theme_ctk.py`` so the web
UI looks like the desktop CTK reference:

* ``dark-teal`` — default, deep teal/charcoal with mint accents.
* ``paper``     — light beige / brown for printed-doc feel.
* ``amber-crt`` — retro warm amber on near-black.

Theme switching is live: ``apply_theme(name)`` rebuilds the CSS string
and pushes it into every previously ``inject_css()``-ed HTML widget,
so the page restyles without reload.
"""
from __future__ import annotations

from typing import Any, Iterable


# ── Palettes (mirrors ctk_migration_pack/ui_theme_ctk.py) ───────────────
PALETTES: dict[str, dict[str, str]] = {
    "dark-teal": {
        "bg":          "#050f0c",
        "panel":       "#0b1c16",
        "panel2":      "#102418",
        "entry":       "#040d0a",
        "fg":          "#e4f4ee",
        "muted":       "#6db39a",
        "muted2":      "#3e7a62",
        "border":      "#1c4a35",
        "border2":     "#112d20",
        "accent":      "#00c896",
        "accent2":     "#1de9b6",
        "accent3":     "#009b75",
        "accent_text": "#04140e",
        "success":     "#1de9b6",
        "warning":     "#f0b429",
        "error":       "#ff5252",
        "warn_border": "#7a5a15",
        "err_border":  "#7a1f1f",
        "ok_bg":       "rgba(29,233,182,0.08)",
        "warn_bg":     "rgba(240,180,41,0.08)",
        "err_bg":      "rgba(255,82,82,0.08)",
        "info_bg":     "rgba(29,233,182,0.06)",
        "accent_bg":   "rgba(29,233,182,0.10)",
        "focus_ring":  "rgba(29,233,182,0.20)",
    },
    "paper": {
        "bg":          "#f5f3ee",
        "panel":       "#ffffff",
        "panel2":      "#faf7f1",
        "entry":       "#ffffff",
        "fg":          "#1a1a1a",
        "muted":       "#6b6356",
        "muted2":      "#504a3f",
        "border":      "#d8d2c4",
        "border2":     "#e8e3d6",
        "accent":      "#b85c2c",
        "accent2":     "#d27040",
        "accent3":     "#8c4520",
        "accent_text": "#ffffff",
        "success":     "#4a7d4a",
        "warning":     "#b8862c",
        "error":       "#b8392c",
        "warn_border": "#c8a060",
        "err_border":  "#d6928a",
        "ok_bg":       "rgba(74,125,74,0.10)",
        "warn_bg":     "rgba(184,134,44,0.10)",
        "err_bg":      "rgba(184,57,44,0.10)",
        "info_bg":     "rgba(184,92,44,0.08)",
        "accent_bg":   "rgba(210,112,64,0.12)",
        "focus_ring":  "rgba(184,92,44,0.24)",
    },
    "amber-crt": {
        "bg":          "#160a02",
        "panel":       "#1f1004",
        "panel2":      "#2a1707",
        "entry":       "#100600",
        "fg":          "#ffd9a3",
        "muted":       "#c89060",
        "muted2":      "#8a6030",
        "border":      "#4a2a10",
        "border2":     "#2e1908",
        "accent":      "#ff8a1f",
        "accent2":     "#ffb060",
        "accent3":     "#cc6a10",
        "accent_text": "#160a02",
        "success":     "#ffb060",
        "warning":     "#ffd060",
        "error":       "#ff5a3a",
        "warn_border": "#a06820",
        "err_border":  "#8a2a18",
        "ok_bg":       "rgba(255,176,96,0.10)",
        "warn_bg":     "rgba(255,208,96,0.10)",
        "err_bg":      "rgba(255,90,58,0.10)",
        "info_bg":     "rgba(255,176,96,0.06)",
        "accent_bg":   "rgba(255,176,96,0.12)",
        "focus_ring":  "rgba(255,176,96,0.24)",
    },
}

DEFAULT_THEME = "dark-teal"


# ── Backward-compat module-level color constants (frozen to default) ────
# Existing callers (``notebook_app.py``) imported these as plain strings.
# Live theme switching works through CSS swap — see ``apply_theme`` — so
# inline styles built from these constants stay on the default palette.
# ``notebook_app.py`` was migrated to CSS classes (``.brt-brand-strong``
# / ``.brt-header-sub``) to participate in hot-swap.
_DEFAULT = PALETTES[DEFAULT_THEME]
BG       = _DEFAULT["bg"]
PANEL    = _DEFAULT["panel"]
PANEL2   = _DEFAULT["panel2"]
FG       = _DEFAULT["fg"]
MUTED    = _DEFAULT["muted"]
MUTED2   = _DEFAULT["muted2"]
ENTRY_BG = _DEFAULT["entry"]
BORDER   = _DEFAULT["border"]
BORDER2  = _DEFAULT["border2"]
ACCENT   = _DEFAULT["accent"]
ACCENT2  = _DEFAULT["accent2"]
ACCENT3  = _DEFAULT["accent3"]
SUCCESS  = _DEFAULT["success"]
WARNING  = _DEFAULT["warning"]
ERROR    = _DEFAULT["error"]


# ── Live-swap state ────────────────────────────────────────────────────
_ACTIVE: dict[str, str] = {"name": DEFAULT_THEME}
_REGISTERED_WIDGETS: list[Any] = []


# ── CSS template (palette keys via ``str.format``) ──────────────────────
# Literal CSS braces are doubled (``{{`` / ``}}``); palette substitutions
# use single ``{key}`` matching ``PALETTES[name]`` keys.
_CSS_TEMPLATE = """
<style>
  /* ── page-level background ───────────────────────────────────────── */
  body, .jp-Notebook, .jp-Cell, .jp-OutputArea, #rendered_cells,
  .vl-OutputArea, .voila-container {{
    background: {bg} !important;
    color: {fg} !important;
  }}
  .jp-Cell {{ padding: 0 !important; }}
  .jp-OutputArea-output pre {{
    background: {entry} !important;
    color: {fg} !important;
  }}

  /* ── widget labels / text ────────────────────────────────────────── */
  .widget-label, .widget-inline-hbox > .widget-label,
  .widget-html-content, .jupyter-widgets label {{
    color: {fg} !important;
  }}
  .widget-html-content code {{
    background: {panel2};
    color: {accent2};
    padding: 1px 5px;
    border-radius: 3px;
  }}

  /* ── text inputs / textarea ──────────────────────────────────────── */
  .widget-text input, .widget-textarea textarea {{
    background: {entry} !important;
    color: {fg} !important;
    border: 1px solid {border} !important;
    border-radius: 4px !important;
  }}
  .widget-text input:focus, .widget-textarea textarea:focus {{
    border-color: {accent} !important;
    box-shadow: 0 0 0 1px {focus_ring} !important;
  }}

  /* ── dropdown ────────────────────────────────────────────────────── */
  .widget-dropdown > select {{
    background: {entry} !important;
    color: {fg} !important;
    border: 1px solid {border} !important;
    border-radius: 4px !important;
  }}

  /* ── sliders ─────────────────────────────────────────────────────── */
  .ui-slider {{
    background: {panel2} !important;
    border: 1px solid {border} !important;
  }}
  .ui-slider .ui-slider-range {{
    background: {accent} !important;
  }}
  .ui-slider .ui-slider-handle {{
    background: {accent2} !important;
    border: 1px solid {accent3} !important;
  }}

  /* ── buttons ─────────────────────────────────────────────────────── */
  .jupyter-widgets button.jupyter-button {{
    background: {panel2} !important;
    color: {fg} !important;
    border: 1px solid {border} !important;
    border-radius: 4px !important;
    font-weight: 600;
  }}
  .jupyter-widgets button.jupyter-button:hover {{
    background: {border} !important;
    border-color: {accent3} !important;
  }}
  .jupyter-widgets button.mod-primary {{
    background: {accent3} !important;
    color: {accent_text} !important;
    border-color: {accent} !important;
  }}
  .jupyter-widgets button.mod-primary:hover {{
    background: {accent} !important;
  }}

  /* ── progress bar ────────────────────────────────────────────────── */
  .progress {{ background: {panel2} !important; }}
  .progress .progress-bar {{ background: {accent} !important; }}
  .progress-bar.bg-success {{ background: {success} !important; }}
  .progress-bar.bg-danger  {{ background: {error}   !important; }}

  /* ── checkbox ────────────────────────────────────────────────────── */
  .widget-checkbox input[type="checkbox"] {{
    accent-color: {accent};
  }}

  /* ── FileUpload button ───────────────────────────────────────────── */
  .widget-upload > .jupyter-button {{
    background: {panel2} !important;
    color: {fg} !important;
    border: 1px dashed {border} !important;
  }}

  /* ── custom helpers used by Python code ──────────────────────────── */
  .brt-app {{
    font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: {fg};
  }}
  .brt-sidebar {{
    background: {panel};
    border-right: 1px solid {border2};
    padding: 14px 10px;
    min-width: 220px;
  }}
  .brt-brand {{
    display: flex; align-items: center; gap: 10px;
    padding: 4px 6px 14px 6px;
    border-bottom: 1px solid {border2};
    margin-bottom: 10px;
  }}
  .brt-brand-badge {{
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient(135deg, {accent3}, {accent2});
    color: {accent_text}; font-weight: 800; font-size: 13px;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: 0.5px;
  }}
  .brt-brand-name {{ font-size: 14px; font-weight: 700; color: {fg}; line-height: 1.15; }}
  .brt-brand-sub  {{ font-size: 10px; color: {muted}; letter-spacing: 1.5px; }}

  .brt-nav-section {{
    color: {muted2}; font-size: 10px; letter-spacing: 1.5px;
    font-weight: 700; padding: 14px 6px 6px 6px;
  }}

  .brt-hw-card {{
    margin-top: 16px;
    background: {panel2};
    border: 1px solid {border2};
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 11px; color: {muted};
    line-height: 1.7;
  }}
  .brt-hw-card b {{ color: {fg}; }}

  .brt-footer {{
    margin-top: auto; padding: 12px 6px 4px 6px;
    color: {muted2}; font-size: 10px;
    border-top: 1px solid {border2};
  }}

  .brt-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px 12px 18px;
    background: {bg};
    border-bottom: 1px solid {border2};
  }}
  .brt-header-title {{
    font-size: 13px; color: {fg}; font-weight: 600;
  }}
  .brt-header-title .muted {{ color: {muted}; font-weight: 400; }}
  .brt-brand-strong {{ color: {accent2}; font-weight: 800; }}
  .brt-header-sub   {{ color: {muted}; }}
  .brt-status {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: {muted}; text-transform: lowercase;
  }}
  .brt-status::before {{
    content: ""; width: 8px; height: 8px; border-radius: 50%;
    background: {accent};
  }}

  .brt-card {{
    background: {panel};
    border: 1px solid {border2};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 14px;
  }}
  .brt-card-title {{
    color: {accent2}; font-size: 11px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
  }}
  .brt-card-sub {{
    color: {muted}; font-size: 12px; margin-top: 2px;
  }}

  .brt-chip {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    border: 1px solid {border};
    background: {panel2};
    color: {fg};
    margin-right: 6px; margin-top: 4px;
  }}
  .brt-chip.ok      {{ color: {success}; border-color: {accent3}; }}
  .brt-chip.warn    {{ color: {warning}; border-color: {warn_border}; }}
  .brt-chip.err     {{ color: {error};   border-color: {err_border}; }}
  .brt-chip.info    {{ color: {accent2}; border-color: {accent3}; }}
  .brt-chip.accent  {{ color: {accent};  border-color: {accent};
                       background: {accent_bg}; font-weight: 700; }}

  .brt-metric {{
    flex: 1 1 0;
    background: {panel2};
    border: 1px solid {border2};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 0 6px;
  }}
  .brt-metric-label {{
    color: {muted}; font-size: 11px;
    letter-spacing: 1.5px; text-transform: uppercase;
    font-weight: 700;
  }}
  .brt-metric-value {{
    color: {fg}; font-size: 28px; font-weight: 700;
    margin-top: 6px;
  }}
  .brt-metric-sub {{
    color: {muted}; font-size: 11px; margin-top: 4px;
  }}

  /* Fix widgets default light borders/backgrounds */
  .widget-container, .widget-hbox, .widget-vbox {{
    background: transparent !important;
  }}
  .jupyter-widgets-output-area {{
    background: {entry} !important;
    color: {fg} !important;
    border-radius: 4px;
  }}

  /* ── header-chip row, predictions table, badges, accordion ──────── */
  .brt-header-chip-row {{
    display: flex; flex-wrap: wrap; gap: 6px;
    padding: 0 14px 8px 18px;
    background: {bg};
    border-bottom: 1px solid {border2};
  }}
  .brt-header-chip-row .brt-chip {{ margin: 0; }}

  .brt-header-actions {{
    display: flex; align-items: center; gap: 8px;
  }}

  .brt-overlay {{
    background: {panel};
    border: 1px solid {border2};
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 14px;
    min-height: 480px;
  }}
  .brt-overlay-title {{
    color: {accent2}; font-weight: 700; font-size: 13px;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 10px;
  }}

  .brt-pred-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    color: {fg};
    background: {panel2};
    border-radius: 4px;
    overflow: hidden;
    margin-top: 6px;
  }}
  .brt-pred-table thead th {{
    background: {entry};
    color: {muted};
    font-weight: 700; text-transform: uppercase;
    font-size: 10px; letter-spacing: 1px;
    padding: 8px 10px;
    border-bottom: 1px solid {border2};
    text-align: left;
  }}
  .brt-pred-table tbody td {{
    padding: 7px 10px;
    border-bottom: 1px solid {border2};
    vertical-align: middle;
  }}
  .brt-pred-table tbody tr:hover {{
    background: {border2};
  }}
  .brt-pred-text {{
    max-width: 520px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    color: {fg};
  }}

  .brt-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid {border};
    background: {entry};
    color: {fg};
  }}
  .brt-badge-ok     {{ color: {success}; border-color: {accent3};  background: {ok_bg}; }}
  .brt-badge-warn   {{ color: {warning}; border-color: {warn_border}; background: {warn_bg}; }}
  .brt-badge-err    {{ color: {error};   border-color: {err_border};  background: {err_bg}; }}
  .brt-badge-info   {{ color: {accent2}; border-color: {accent3};  background: {info_bg}; }}
  .brt-badge-accent {{ color: {accent};  border-color: {accent};   background: {accent_bg}; }}

  .brt-filter-tabs {{
    display: flex; gap: 6px; margin: 6px 0 4px 0;
    border-bottom: 1px solid {border2}; padding-bottom: 6px;
  }}

  /* override jupyter accordion to match active theme */
  .jupyter-widgets.widget-accordion .p-Collapse-header,
  .jupyter-widgets.widget-accordion .lm-Collapse-header {{
    background: {panel2} !important;
    color: {accent2} !important;
    border: 1px solid {border2} !important;
    font-weight: 700; letter-spacing: 1.0px; text-transform: uppercase;
    font-size: 11px;
  }}
  .jupyter-widgets.widget-accordion .p-Collapse-contents,
  .jupyter-widgets.widget-accordion .lm-Collapse-contents {{
    background: {panel} !important;
    color: {fg} !important;
    border: 1px solid {border2} !important;
    border-top: none !important;
  }}

  /* ── new helpers (Sprint 2) ──────────────────────────────────────── */
  .brt-field-label {{
    color: {muted}; font-size: 10px; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin: 6px 0 2px 0;
  }}
  .brt-separator {{
    height: 1px; background: {border2};
    margin: 10px 0;
  }}
  .brt-card-head-row {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; width: 100%;
  }}
  .brt-theme-switch-title {{
    color: {muted2}; font-size: 10px; letter-spacing: 1.5px;
    font-weight: 700; padding: 14px 6px 4px 6px;
  }}
</style>
"""


def _build_css(palette_name: str) -> str:
    """Render the CSS template with the named palette substituted in."""
    palette = PALETTES[palette_name]
    return _CSS_TEMPLATE.format(**palette)


# ── Public API ─────────────────────────────────────────────────────────
def available_themes() -> list[str]:
    """Return the list of theme names suitable for ``apply_theme``."""
    return list(PALETTES.keys())


def current_theme() -> str:
    """Return the currently active theme name."""
    return _ACTIVE["name"]


def apply_theme(name: str) -> None:
    """Switch to ``name`` and hot-swap CSS in every registered widget.

    Raises ``ValueError`` for unknown names. Safe to call before or
    after ``inject_css()`` — late inject calls pick up the new theme
    automatically.
    """
    if name not in PALETTES:
        raise ValueError(
            f"Unknown theme: {name!r}. Choose from {list(PALETTES)}."
        )
    _ACTIVE["name"] = name
    css = _build_css(name)
    for widget in _REGISTERED_WIDGETS:
        try:
            widget.value = css
        except Exception:  # noqa: BLE001 — widget may have been disposed
            continue


def inject_css() -> Any:
    """Return an ``ipywidgets.HTML`` carrying the active theme's CSS.

    The widget is also registered for live theme swap — call
    ``apply_theme(name)`` later and every previously injected widget
    updates without page reload. Include this once at the top of
    ``build_app()``.
    """
    import ipywidgets as w
    widget = w.HTML(value=_build_css(_ACTIVE["name"]))
    _REGISTERED_WIDGETS.append(widget)
    return widget


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
        border=f"1px solid {BORDER2}",
        padding="10px 14px",
        margin="8px 14px",
    )


def section_card(
    title: str,
    children: Iterable[Any],
    subtitle: str = "",
    right: Any | None = None,
) -> Any:
    """VBox with a teal section header + body children, styled as a card.

    ``right`` (optional widget) — placed on the right side of the
    header row, useful for inline action buttons or status chips.
    """
    import ipywidgets as w
    head_inner = section_header(title, subtitle)
    if right is None:
        head: Any = head_inner
    else:
        head = w.HBox(
            [
                head_inner,
                w.Box(layout=w.Layout(flex="1 1 auto")),
                right,
            ],
            layout=w.Layout(
                width="100%",
                align_items="center",
                justify_content="space-between",
            ),
        )
        head.add_class("brt-card-head-row")
    return w.VBox([head, *children], layout=card_layout())


def field_label(text: str) -> Any:
    """Small uppercase muted label for input fields and form rows."""
    import ipywidgets as w
    return w.HTML(f"<div class='brt-field-label'>{text}</div>")


def separator() -> Any:
    """Thin horizontal divider in the active theme's border color."""
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
