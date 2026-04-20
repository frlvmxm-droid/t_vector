# -*- coding: utf-8 -*-
"""Top-level entry point for the Voilà dashboard: a 3-tab widget app."""
from __future__ import annotations

from typing import Any


def build_app() -> Any:
    """Construct the full BankReasonTrainer web UI.

    Returns an ``ipywidgets.VBox`` that can be ``display()``-ed from a
    notebook cell. Three tabs — **Обучение**, **Применение**,
    **Кластеризация** — each backed by the headless service layer.
    """
    import ipywidgets as w

    from ui_widgets.apply_panel import build_apply_panel
    from ui_widgets.cluster_panel import build_cluster_panel
    from ui_widgets.train_panel import build_train_panel

    tabs = w.Tab(children=[
        build_train_panel(),
        build_apply_panel(),
        build_cluster_panel(),
    ])
    tabs.set_title(0, "📚 Обучение")
    tabs.set_title(1, "🎯 Применение")
    tabs.set_title(2, "🧩 Кластеризация")

    header = w.HTML(
        "<div style='padding:8px 0;border-bottom:2px solid #4a90e2;margin-bottom:8px'>"
        "<h2 style='margin:0;color:#222'>BankReasonTrainer — Web UI</h2>"
        "<span style='color:#666;font-size:0.9em'>"
        "Классификация и кластеризация обращений. Работает через сервисный слой — "
        "Tkinter-GUI не требуется."
        "</span></div>"
    )

    footer = w.HTML(
        "<div style='margin-top:12px;padding-top:8px;border-top:1px solid #eee;"
        "color:#888;font-size:0.85em'>"
        "Документация: <code>docs/JUPYTERHUB_UI.md</code> • "
        "CLI: <code>python -m bank_reason_trainer {train,apply,cluster}</code>"
        "</div>"
    )

    return w.VBox([header, tabs, footer])
