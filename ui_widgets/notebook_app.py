# -*- coding: utf-8 -*-
"""Top-level entry point for the Voilà dashboard: dark-teal sidebar app."""
from __future__ import annotations

from typing import Any, List, Tuple


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
    from ui_widgets.theme import (
        ACCENT2, MUTED, inject_css, status_badge,
    )
    from ui_widgets.train_panel import build_train_panel

    # ── Main stack (one panel visible at a time) ────────────────────────
    train_p = build_train_panel()
    apply_p = build_apply_panel()
    cluster_p = build_cluster_panel()
    stack = w.Stack(
        children=[train_p, apply_p, cluster_p],
        selected_index=1,  # default to «Классификация» — matches screenshot
    )

    # ── Sidebar nav buttons ────────────────────────────────────────────
    nav_items: List[Tuple[str, str, str]] = [
        ("📚", "Обучение",      ""),
        ("🎯", "Классификация", ""),
        ("🧩", "Кластеризация", ""),
    ]
    nav_buttons: List[Any] = []
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
            f"<span style='color:{ACCENT2};font-weight:800'>BankReasonTrainer</span>"
            f"<span class='muted'> — {title}</span>"
            f"<span style='color:{MUTED}'>  ·  {sub}</span>"
            "</div>"
        )

    def _select(index: int) -> None:
        stack.selected_index = index
        for i, b in enumerate(nav_buttons):
            b.button_style = "primary" if i == index else ""
        _render_header_title(index)

    for i, btn in enumerate(nav_buttons):
        btn.on_click(lambda _b, _i=i: _select(_i))
    _select(1)  # highlight «Классификация» initially

    context_items = [
        ("🕘", "История экспериментов"),
        ("📦", "Артефакты моделей"),
        ("⚙️", "Настройки · LLM keys"),
    ]
    context_buttons: List[Any] = []
    for icon, label in context_items:
        btn = w.Button(
            description=f"{icon}  {label}",
            disabled=True,
            tooltip="Пока не реализовано в веб-UI (см. desktop)",
            layout=w.Layout(width="96%", margin="2px 0",
                            display="flex", justify_content="flex-start"),
        )
        context_buttons.append(btn)

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

    sidebar = w.VBox(
        [
            brand_html,
            workflow_title, *nav_buttons,
            context_title, *context_buttons,
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
            border_bottom=f"1px solid #112d20",
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


def _hardware_card_html() -> str:
    """Best-effort local hardware stats for the sidebar card."""
    try:
        import os
        cpu = os.cpu_count() or "?"
    except Exception:
        cpu = "?"
    ram = "—"
    try:
        import psutil  # optional
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        used_gb = psutil.virtual_memory().used / (1024 ** 3)
        ram = f"{used_gb:.1f} / {total_gb:.1f} ГБ"
    except Exception:
        pass
    gpu = "—"
    torch_ver = "—"
    try:
        import torch  # optional
        torch_ver = torch.__version__.split("+")[0]
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return (
        "<div class='brt-hw-card'>"
        f"<div>CPU&nbsp;&nbsp;&nbsp;<b>{cpu} cores</b></div>"
        f"<div>RAM&nbsp;&nbsp;&nbsp;<b>{ram}</b></div>"
        f"<div>GPU&nbsp;&nbsp;&nbsp;<b>{gpu}</b></div>"
        f"<div>torch&nbsp;<b>{torch_ver}</b></div>"
        "</div>"
    )
