# -*- coding: utf-8 -*-
"""HTML table widget for predictions preview with chip-badges and filter tabs."""
from __future__ import annotations

import html
from typing import Any, Iterable, List, Sequence


_BUCKETS = (
    ("all",   "Все",                    None),
    ("high",  "Высокая ≥ 0.85",         "high"),
    ("low",   "< 0.50",                  "low"),
    ("rev",   "needs_review",            "review"),
)


def render_predictions_html(
    texts: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[float],
    needs_review: Sequence[int],
    *,
    bucket: str = "all",
    limit: int = 12,
) -> str:
    """Return an HTML table string with chip-badged predictions.

    ``bucket`` ∈ ``{'all', 'high', 'low', 'review'}`` filters rows before
    truncating to ``limit``.
    """
    rows = list(zip(texts, labels, confidences, needs_review))
    rows = _filter_bucket(rows, bucket)
    rows = rows[:limit]

    body: List[str] = []
    for text, label, conf, rev in rows:
        body.append(
            "<tr>"
            f"<td class='brt-pred-text' title='{html.escape(str(text))}'>"
            f"{html.escape(_truncate(str(text), 140))}</td>"
            f"<td>{_label_badge(str(label))}</td>"
            f"<td>{_conf_badge(float(conf))}</td>"
            f"<td>{_review_badge(int(rev))}</td>"
            "</tr>"
        )
    if not body:
        body.append(
            "<tr><td colspan='4' style='padding:14px;text-align:center;"
            "color:#6db39a'>— нет строк под фильтр —</td></tr>"
        )

    return (
        "<table class='brt-pred-table'>"
        "<thead><tr>"
        "<th>Текст</th><th>Метка</th><th>Confidence</th><th>Review</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


def build_predictions_preview(
    texts: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[float],
    needs_review: Sequence[int],
    *,
    limit: int = 12,
) -> Any:
    """Return a ``VBox`` with filter tabs + table that re-renders on tab click."""
    import ipywidgets as wgt

    state = {"bucket": "all"}
    table_html = wgt.HTML(
        value=render_predictions_html(
            texts, labels, confidences, needs_review,
            bucket="all", limit=limit,
        ),
    )
    counts = _bucket_counts(texts, labels, confidences, needs_review)

    tab_buttons: List[Any] = []
    for key, label, _ in _BUCKETS:
        n = counts[key]
        btn = wgt.Button(
            description=f"{label}  ({n})",
            layout=wgt.Layout(width="auto", margin="0"),
        )
        tab_buttons.append((key, btn))

    def _on_click(key: str) -> None:
        state["bucket"] = key
        for k, b in tab_buttons:
            b.button_style = "primary" if k == key else ""
        table_html.value = render_predictions_html(
            texts, labels, confidences, needs_review,
            bucket=key, limit=limit,
        )

    for key, btn in tab_buttons:
        btn.on_click(lambda _b, _k=key: _on_click(_k))
    if tab_buttons:
        tab_buttons[0][1].button_style = "primary"

    tabs_box = wgt.HBox(
        [b for _, b in tab_buttons],
        layout=wgt.Layout(margin="2px 0 6px 0"),
    )
    tabs_box.add_class("brt-filter-tabs")
    return wgt.VBox([tabs_box, table_html])


# ─── internals ─────────────────────────────────────────────────────────
def _filter_bucket(
    rows: Iterable[tuple], bucket: str,
) -> List[tuple]:
    if bucket == "all":
        return list(rows)
    if bucket == "high":
        return [r for r in rows if float(r[2]) >= 0.85]
    if bucket == "low":
        return [r for r in rows if float(r[2]) < 0.50]
    if bucket == "review":
        return [r for r in rows if int(r[3]) == 1]
    return list(rows)


def _bucket_counts(
    texts: Sequence[str], labels: Sequence[str],
    confidences: Sequence[float], needs_review: Sequence[int],
) -> dict:
    rows = list(zip(texts, labels, confidences, needs_review))
    return {
        "all":  len(rows),
        "high": sum(1 for r in rows if float(r[2]) >= 0.85),
        "low":  sum(1 for r in rows if float(r[2]) < 0.50),
        "rev":  sum(1 for r in rows if int(r[3]) == 1),
    }


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def _label_badge(label: str) -> str:
    safe = html.escape(label or "—")
    return f"<span class='brt-badge brt-badge-info'>{safe}</span>"


def _conf_badge(conf: float) -> str:
    pct = f"{conf:.2f}"
    if conf >= 0.85:
        kind = "ok"
    elif conf >= 0.50:
        kind = "warn"
    else:
        kind = "err"
    return f"<span class='brt-badge brt-badge-{kind}'>{pct}</span>"


def _review_badge(flag: int) -> str:
    if flag:
        return "<span class='brt-badge brt-badge-warn'>review</span>"
    return "<span class='brt-badge brt-badge-ok'>ok</span>"
