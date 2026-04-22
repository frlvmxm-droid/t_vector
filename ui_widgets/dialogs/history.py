"""History dialog — last N experiments from `experiment_log`."""
from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

LIMIT = 20


def build_history_dialog(on_close: Callable[[], None]) -> Any:
    """VBox card with experiment history table and a close button."""
    import ipywidgets as w

    refresh_btn = w.Button(
        description="🔄 Обновить",
        layout=w.Layout(width="auto"),
    )
    close_btn = w.Button(
        description="✕ Закрыть",
        button_style="primary",
        layout=w.Layout(width="auto"),
    )
    close_btn.on_click(lambda _b: on_close())

    table_html = w.HTML(_render_table())

    def _refresh(_b: Any = None) -> None:
        table_html.value = _render_table()

    refresh_btn.on_click(_refresh)

    header = w.HBox(
        [
            w.HTML("<div class='brt-overlay-title'>🕘 История экспериментов</div>"),
            w.HBox([refresh_btn, close_btn], layout=w.Layout(margin="0")),
        ],
        layout=w.Layout(
            justify_content="space-between",
            align_items="center",
            padding="0 0 8px 0",
        ),
    )
    box = w.VBox([header, table_html])
    box.add_class("brt-overlay")
    return box


def _render_table() -> str:
    try:
        from experiment_log import read_experiments
    except Exception as exc:  # noqa: BLE001 — defensive
        return f"<div class='muted'>experiment_log unavailable: {html.escape(str(exc))}</div>"

    records = read_experiments(LIMIT)
    if not records:
        return (
            "<div class='muted' style='padding:18px;text-align:center'>"
            f"— нет записей в {_log_path()} —"
            "</div>"
        )

    rows = []
    for rec in reversed(records):  # newest first
        ts = html.escape(str(rec.get("timestamp") or "—"))
        model = html.escape(str(rec.get("model_file") or "—"))
        macro_f1 = rec.get("macro_f1")
        f1_str = f"{macro_f1:.3f}" if isinstance(macro_f1, (int, float)) else "—"
        n_train = rec.get("n_train")
        n_test = rec.get("n_test")
        size_b = rec.get("model_size_bytes")
        size_str = _fmt_bytes(size_b) if isinstance(size_b, (int, float)) else "—"
        dur = rec.get("training_duration_sec")
        dur_str = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "—"
        rows.append(
            "<tr>"
            f"<td style='font-family:monospace;color:#9ec9b8'>{ts}</td>"
            f"<td style='font-family:monospace;font-size:11px'>{model}</td>"
            f"<td><span class='brt-badge brt-badge-info'>F1 {f1_str}</span></td>"
            f"<td>{n_train or '—'} / {n_test or '—'}</td>"
            f"<td>{dur_str}</td>"
            f"<td>{size_str}</td>"
            "</tr>"
        )
    return (
        "<table class='brt-pred-table'>"
        "<thead><tr>"
        "<th>Дата</th><th>Модель</th><th>F1</th>"
        "<th>train/test</th><th>time</th><th>size</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _log_path() -> str:
    try:
        from experiment_log import EXPERIMENT_LOG

        return str(EXPERIMENT_LOG)
    except Exception:  # noqa: BLE001
        return "~/.classification_tool/experiments.jsonl"


def _fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
