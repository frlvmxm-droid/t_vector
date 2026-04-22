"""Top-level entry point for the Voilà dashboard: sidebar app + themes."""
from __future__ import annotations

from typing import Any


# Sidebar theme-switcher labels → palette names in ui_widgets.theme.PALETTES.
_THEME_CHOICES: tuple[tuple[str, str], ...] = (
    ("Teal",  "dark-teal"),
    ("Paper", "paper"),
    ("CRT",   "amber-crt"),
)


def build_app() -> Any:
    """Construct the full BankReasonTrainer web UI.

    Returns an ``ipywidgets.VBox`` that can be ``display()``-ed from a
    notebook cell. Layout: left sidebar with workflow/context nav +
    hardware card + theme-switcher; main area shows one of three panels
    (**Обучение / Классификация / Кластеризация**) backed by the headless
    service layer. Three palettes — dark-teal (default), paper,
    amber-crt — are switchable at runtime; the choice is persisted in
    ``~/.classification_tool/last_session.json`` under key ``ui.theme``.
    """
    import ipywidgets as w

    from ui_widgets import theme as _theme
    from ui_widgets.apply_panel import build_apply_panel
    from ui_widgets.cluster_panel import build_cluster_panel
    from ui_widgets.dialogs import (
        build_artifacts_dialog,
        build_history_dialog,
        build_settings_dialog,
    )
    from ui_widgets.session import DebouncedSaver, load_last_session
    from ui_widgets.theme import (
        apply_theme,
        get_active_theme,
        inject_css,
        rebuild_css,
        status_badge,
    )
    from ui_widgets.train_panel import build_train_panel

    # ── Restore persisted theme BEFORE building the CSS widget ──────────
    try:
        _saved = load_last_session() or {}
    except Exception:  # noqa: BLE001 — session corruption is non-fatal
        _saved = {}
    _saved_theme = _saved.get("ui.theme") if isinstance(_saved, dict) else None
    if isinstance(_saved_theme, str) and _saved_theme in _theme.PALETTES:
        apply_theme(_saved_theme)

    # ── Main stack (one panel visible at a time) ────────────────────────
    train_p, train_widgets, train_snap = build_train_panel()
    apply_p, apply_widgets, apply_snap = build_apply_panel()
    cluster_p, cluster_widgets, cluster_snap = build_cluster_panel()

    # ── Session save/restore ────────────────────────────────────────────
    # Extra snap fn injects the active theme so `ui.theme` is written on
    # every debounced save.
    saver = _wire_session(
        widgets_by_key={**train_widgets, **apply_widgets, **cluster_widgets},
        snap_fns=(
            train_snap,
            apply_snap,
            cluster_snap,
            lambda: {"ui.theme": get_active_theme()},
        ),
        load_last_session=load_last_session,
        debounced_saver_cls=DebouncedSaver,
    )

    # Last-active workflow panel (0..2) so dialog close returns to it.
    last_panel_index = {"value": 1}

    def _show_panel(index: int) -> None:
        last_panel_index["value"] = index
        stack.selected_index = index
        for i, b in enumerate(nav_buttons):
            b.button_style = "primary" if i == index else ""
        for b in context_buttons:
            b.button_style = ""
        _render_header_title(index)

    def _close_dialog() -> None:
        _show_panel(last_panel_index["value"])

    history_d = build_history_dialog(_close_dialog)
    artifacts_d = build_artifacts_dialog(_close_dialog)
    settings_d = build_settings_dialog(_close_dialog)

    stack = w.Stack(
        children=[train_p, apply_p, cluster_p, history_d, artifacts_d, settings_d],
        selected_index=1,  # default to «Классификация» — matches screenshot
    )

    # ── Sidebar nav buttons ────────────────────────────────────────────
    nav_items: list[tuple[str, str, str]] = [
        ("📚", "Обучение",      ""),
        ("🎯", "Классификация", ""),
        ("🧩", "Кластеризация", ""),
    ]
    nav_buttons: list[Any] = []
    for icon, label, badge_text in nav_items:
        desc = f"{icon}  {label}" + (f"   ({badge_text})" if badge_text else "")
        btn = w.Button(
            description=desc,
            layout=w.Layout(width="96%", margin="2px 0",
                            display="flex", justify_content="flex-start"),
        )
        nav_buttons.append(btn)

    # Header title + subtitle reflect the active panel
    panel_titles = (
        ("Обучение",      "ml model training · TF-IDF + LinearSVC"),
        ("Классификация", "batch prediction · per-class thresholds"),
        ("Кластеризация", "unsupervised · TF-IDF / SBERT / Combo / Ensemble"),
    )
    header_title_html = w.HTML("")

    def _render_header_title(index: int) -> None:
        title, sub = panel_titles[index]
        header_title_html.value = (
            "<div class='brt-header-title'>"
            "<span class='brand'>BankReasonTrainer</span>"
            f"<span class='muted'> — {title}</span>"
            f"<span class='sub'>  ·  {sub}</span>"
            "</div>"
        )

    def _select(index: int) -> None:
        _show_panel(index)

    for i, btn in enumerate(nav_buttons):
        btn.on_click(lambda _b, _i=i: _select(_i))

    # Sidebar КОНТЕКСТ buttons — open dialogs in slots 3/4/5 of the stack.
    context_items = [
        ("🕘", "История экспериментов", 3),
        ("📦", "Артефакты моделей",     4),
        ("⚙️", "Настройки · LLM keys",  5),
    ]
    context_buttons: list[Any] = []
    for icon, label, _slot in context_items:
        btn = w.Button(
            description=f"{icon}  {label}",
            layout=w.Layout(width="96%", margin="2px 0",
                            display="flex", justify_content="flex-start"),
        )
        context_buttons.append(btn)

    def _open_dialog(slot: int, btn: Any) -> None:
        stack.selected_index = slot
        for b in nav_buttons:
            b.button_style = ""
        for b in context_buttons:
            b.button_style = ""
        btn.button_style = "primary"

    for (_icon, _label, slot), btn in zip(context_items, context_buttons):
        btn.on_click(lambda _b, _slot=slot, _btn=btn: _open_dialog(_slot, _btn))

    _select(1)  # highlight «Классификация» initially

    brand_html = w.HTML(
        "<div class='brt-brand'>"
        "<div class='brt-brand-badge'>BR</div>"
        "<div>"
        "<div class='brt-brand-name'>BankReason</div>"
        "<div class='brt-brand-sub'>TRAINER · RU</div>"
        "</div>"
        "</div>"
    )
    workflow_title = w.HTML("<div class='brt-nav-section'>WORKFLOW</div>")
    context_title = w.HTML("<div class='brt-nav-section'>КОНТЕКСТ</div>")
    hw_card = w.HTML(_hardware_card_html())
    footer_html = w.HTML("<div class='brt-footer'>v3.4.1 · build 1248</div>")

    # ── CSS widget (honours restored _ACTIVE_THEME) ─────────────────────
    css_widget = inject_css()

    # ── Theme-switcher in sidebar ───────────────────────────────────────
    theme_buttons: list[Any] = []
    for label, _palette in _THEME_CHOICES:
        btn = w.Button(
            description=label,
            layout=w.Layout(width="auto", margin="0"),
        )
        theme_buttons.append(btn)

    def _refresh_theme_buttons() -> None:
        active = get_active_theme()
        for btn, (_label, palette) in zip(theme_buttons, _THEME_CHOICES):
            btn.button_style = "primary" if palette == active else ""

    def _set_theme(palette: str) -> None:
        try:
            apply_theme(palette)
        except ValueError:
            return
        css_widget.value = rebuild_css()
        _refresh_theme_buttons()
        saver.schedule()

    for btn, (_label, palette) in zip(theme_buttons, _THEME_CHOICES):
        btn.on_click(lambda _b, _p=palette: _set_theme(_p))

    _refresh_theme_buttons()

    theme_switcher = w.HBox(theme_buttons)
    theme_switcher.add_class("brt-theme-switcher")

    sidebar = w.VBox(
        [
            brand_html,
            workflow_title, *nav_buttons,
            context_title, *context_buttons,
            hw_card,
            theme_switcher,
            footer_html,
        ],
        layout=w.Layout(
            width="230px",
            min_height="720px",
            padding="12px 10px",
        ),
    )
    sidebar.add_class("brt-sidebar")

    # ── Main area: header strip + stack ─────────────────────────────────
    status_html = w.HTML(
        f"<div class='brt-header-actions'>{status_badge('idle')}</div>"
    )
    header = w.HBox(
        [header_title_html, status_html],
        layout=w.Layout(
            justify_content="space-between",
            align_items="center",
            padding="10px 14px 12px 18px",
            border_bottom="1px solid #112d20",
            width="100%",
        ),
    )
    header.add_class("brt-header")
    main = w.VBox(
        [header, stack],
        layout=w.Layout(flex="1 1 auto", min_height="720px"),
    )

    # ── Root ────────────────────────────────────────────────────────────
    root = w.HBox(
        [sidebar, main],
        layout=w.Layout(align_items="stretch", width="100%"),
    )
    root.add_class("brt-app")

    return w.VBox([css_widget, root])


def _wire_session(
    *,
    widgets_by_key: dict,
    snap_fns: tuple,
    load_last_session: Any,
    debounced_saver_cls: Any,
) -> Any:
    """Restore the last saved snap into ``widgets_by_key`` and wire a
    ``DebouncedSaver`` that coalesces rapid widget changes into one write.

    Returns the saver so the caller (e.g. theme-switcher) can trigger
    ``saver.schedule()`` on non-widget events.

    Failures are swallowed silently — session state is nice-to-have, not
    load-bearing.  One bad key (e.g. a legacy snap-key whose widget lost
    its option) must not derail restore of the remaining values.
    """
    try:
        saved = load_last_session()
    except Exception:
        saved = None
    if isinstance(saved, dict):
        for key, value in saved.items():
            widget = widgets_by_key.get(key)
            if widget is None:
                continue
            try:
                widget.value = value
            except Exception:  # noqa: BLE001 — TraitError / TypeError / ValueError
                pass

    def _collect_snap() -> dict:
        merged: dict = {}
        for fn in snap_fns:
            try:
                merged.update(fn())
            except Exception:  # noqa: BLE001
                continue
        return merged

    saver = debounced_saver_cls(_collect_snap, delay_sec=2.0)

    def _on_change(_change: Any) -> None:
        saver.schedule()

    for widget in widgets_by_key.values():
        try:
            widget.observe(_on_change, names="value")
        except Exception:  # noqa: BLE001 — widget without `value` trait
            continue

    return saver


def _hardware_card_html() -> str:
    """Best-effort local hardware stats for the sidebar card.

    Kept cheap on purpose: no ``import torch`` and no CUDA probe at
    ``build_app()`` time — both can stall for 5–30 s on JupyterHub
    kernels with heavy ML stacks or misconfigured GPU drivers. Presence
    of ``torch`` is probed via ``importlib.util.find_spec`` (metadata
    only, no module load).
    """
    import importlib.util
    import os
    cpu = os.cpu_count() or "?"
    ram = "—"
    try:
        import psutil  # optional
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        used_gb = psutil.virtual_memory().used / (1024 ** 3)
        ram = f"{used_gb:.1f} / {total_gb:.1f} ГБ"
    except Exception:
        pass
    torch_status = (
        "установлен" if importlib.util.find_spec("torch") is not None else "—"
    )
    return (
        "<div class='brt-hw-card'>"
        f"<div>CPU&nbsp;&nbsp;&nbsp;<b>{cpu} cores</b></div>"
        f"<div>RAM&nbsp;&nbsp;&nbsp;<b>{ram}</b></div>"
        f"<div>torch&nbsp;<b>{torch_status}</b></div>"
        "</div>"
    )
