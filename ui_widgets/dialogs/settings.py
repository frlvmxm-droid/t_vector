# -*- coding: utf-8 -*-
"""Settings dialog — read-only deps + LLM-keys indicator."""
from __future__ import annotations

import html
import os
from typing import Any, Callable

OPTIONAL_DEPS = (
    ("ipywidgets", "ipywidgets"),
    ("voila", "voila"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sentence_transformers", "sentence-transformers"),
    ("hdbscan", "hdbscan"),
    ("umap", "umap-learn"),
    ("bertopic", "BERTopic"),
    ("setfit", "SetFit"),
    ("kneed", "kneed"),
    ("pymorphy3", "pymorphy3"),
    ("psutil", "psutil"),
    ("openpyxl", "openpyxl"),
)

LLM_ENV_VARS = (
    ("BRT_LLM_PROVIDER",  "Provider override"),
    ("BRT_LLM_API_KEY",   "API key"),
    ("BRT_LLM_MODEL",     "Model override"),
    ("LLM_SNAPSHOT_KEY",  "Fernet snapshot key"),
    ("ANTHROPIC_API_KEY", "Anthropic"),
    ("OPENAI_API_KEY",    "OpenAI"),
)


def build_settings_dialog(on_close: Callable[[], None]) -> Any:
    import ipywidgets as w

    close_btn = w.Button(
        description="✕ Закрыть",
        button_style="primary",
        layout=w.Layout(width="auto"),
    )
    close_btn.on_click(lambda _b: on_close())

    deps_html = w.HTML(_render_deps())
    keys_html = w.HTML(_render_keys())

    header = w.HBox(
        [
            w.HTML("<div class='brt-overlay-title'>⚙️ Настройки · LLM keys</div>"),
            close_btn,
        ],
        layout=w.Layout(
            justify_content="space-between",
            align_items="center",
            padding="0 0 8px 0",
        ),
    )
    info = w.HTML(
        "<div class='muted' style='padding:6px 0'>"
        "Read-only. Установка ключей — через переменные окружения "
        "перед запуском Voilà (см. docs/JUPYTERHUB_UI.md)."
        "</div>"
    )
    box = w.VBox([header, deps_html, keys_html, info])
    box.add_class("brt-overlay")
    return box


def _render_deps() -> str:
    rows = []
    for mod, pkg in OPTIONAL_DEPS:
        try:
            __import__(mod)
            kind, label = "ok", "✅ установлен"
        except ImportError:
            kind, label = "warn", "— не установлен"
        rows.append(
            "<tr>"
            f"<td style='font-family:monospace'>{html.escape(pkg)}</td>"
            f"<td><span class='brt-badge brt-badge-{kind}'>{label}</span></td>"
            "</tr>"
        )
    return (
        "<h4 style='color:#9ec9b8;margin:6px 0 4px 0'>Зависимости</h4>"
        "<table class='brt-pred-table'>"
        "<thead><tr><th>Пакет</th><th>Статус</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_keys() -> str:
    rows = []
    for var, descr in LLM_ENV_VARS:
        present = bool((os.environ.get(var) or "").strip())
        kind, label = ("ok", "задан") if present else ("warn", "не задан")
        rows.append(
            "<tr>"
            f"<td style='font-family:monospace'>{html.escape(var)}</td>"
            f"<td>{html.escape(descr)}</td>"
            f"<td><span class='brt-badge brt-badge-{kind}'>{label}</span></td>"
            "</tr>"
        )
    return (
        "<h4 style='color:#9ec9b8;margin:14px 0 4px 0'>Переменные окружения</h4>"
        "<table class='brt-pred-table'>"
        "<thead><tr><th>Переменная</th><th>Назначение</th><th>Статус</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
