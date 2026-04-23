"""Top-level entry point for the Voilà dashboard: dark-teal sidebar app."""
from __future__ import annotations

from typing import Any


def build_app() -> Any:
    """Construct the full BankReasonTrainer web UI.

    Returns an ``ipywidgets.VBox`` that can be ``display()``-ed from a
    notebook cell. Layout: left sidebar with workflow/context nav +
    hardware card; main area shows one of three panels
    (**Обучение / Классификация / Кластеризация**) backed by the headless
    service layer. Dark-teal palette mirrors ``ui_theme.py``.
    """
    import ipywidgets as w

    from ui_widgets.apply_panel import build_apply_panel
    from ui_widgets.cluster_panel import build_cluster_panel
    from ui_widgets.dialogs import (
        build_artifacts_dialog,
        build_history_dialog,
        build_settings_dialog,
    )
    from ui_widgets.session import DebouncedSaver, load_last_session
    from ui_widgets.theme import (
        DEFAULT_THEME,
        apply_theme,
        available_themes,
        inject_css,
        status_badge,
    )
    from ui_widgets.train_panel import build_train_panel

    # ── Main stack (one panel visible at a time) ────────────────────────
    train_p, train_widgets, train_snap = build_train_panel()
    apply_p, apply_widgets, apply_snap = build_apply_panel()
    cluster_p, cluster_widgets, cluster_snap = build_cluster_panel()

    # ── Theme switcher widget (Sprint 2) ────────────────────────────────
    # ToggleButtons holds the active theme name; wiring through
    # widgets_by_key/snap_fn propagates it to ``last_session.json`` under
    # the ``ui.theme`` key, on the same debounced write path as panel state.
    theme_options = available_themes()
    theme_picker = w.ToggleButtons(
        options=[
            ("Teal",  "dark-teal"),
            ("Paper", "paper"),
            ("CRT",   "amber-crt"),
        ],
        value=DEFAULT_THEME,
        description="",
        button_style="",
        layout=w.Layout(width="96%", margin="2px 0 6px 0"),
        style={"button_width": "32%"},
    )

    def _on_theme_change(change: Any) -> None:
        new_name = change.get("new") if isinstance(change, dict) else None
        if not isinstance(new_name, str) or new_name not in theme_options:
            return
        try:
            apply_theme(new_name)
        except ValueError:
            return

    theme_picker.observe(_on_theme_change, names="value")

    theme_widgets = {"ui.theme": theme_picker}
    theme_snap = lambda: {"ui.theme": theme_picker.value}

    # ── Session save/restore ────────────────────────────────────────────
    _wire_session(
        widgets_by_key={
            **train_widgets, **apply_widgets, **cluster_widgets, **theme_widgets,
        },
        snap_fns=(train_snap, apply_snap, cluster_snap, theme_snap),
        load_last_session=load_last_session,
        debounced_saver_cls=DebouncedSaver,
    )

    # Restore propagated to widget.value above; mirror it into CSS now.
    apply_theme(theme_picker.value)

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
    for icon, label, badge in nav_items:
        desc = f"{icon}  {label}" + (f"   ({badge})" if badge else "")
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
            "<span class='brt-brand-strong'>BankReasonTrainer</span>"
            f"<span class='muted'> — {title}</span>"
            f"<span class='brt-header-sub'>  ·  {sub}</span>"
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
    theme_title = w.HTML("<div class='brt-theme-switch-title'>ТЕМА</div>")
    hw_card = w.HTML(_hardware_card_html())
    footer_html = w.HTML("<div class='brt-footer'>v3.4.1 · build 1248</div>")

    sidebar = w.VBox(
        [
            brand_html,
            workflow_title, *nav_buttons,
            context_title, *context_buttons,
            theme_title, theme_picker,
            hw_card,
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

    return w.VBox([inject_css(), root])


def _wire_session(
    *,
    widgets_by_key: dict,
    snap_fns: tuple,
    load_last_session: Any,
    debounced_saver_cls: Any,
) -> None:
    """Restore the last saved snap into ``widgets_by_key`` and wire a
    ``DebouncedSaver`` that coalesces rapid widget changes into one write.

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
