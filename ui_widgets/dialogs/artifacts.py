"""Artifacts dialog — list `.joblib` bundles in `~/.classification_tool/`."""
from __future__ import annotations

import hashlib
import html
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT_CANDIDATES = (
    "~/.classification_tool",
    "~/classification_tool",
)

_INITIAL_PLACEHOLDER = (
    "<div class='muted' style='padding:18px;text-align:center'>"
    "Нажмите 🔄 Сканировать — список артефактов загрузится."
    "</div>"
)


def build_artifacts_dialog(on_close: Callable[[], None]) -> Any:
    """VBox card listing locally cached `.joblib` artifacts."""
    import ipywidgets as w

    refresh_btn = w.Button(
        description="🔄 Сканировать",
        layout=w.Layout(width="auto"),
    )
    close_btn = w.Button(
        description="✕ Закрыть",
        button_style="primary",
        layout=w.Layout(width="auto"),
    )
    close_btn.on_click(lambda _b: on_close())

    # Deliberately NOT calling ``_render_table()`` here — on a JupyterHub
    # kernel backed by NFS the rglob + SHA-256 hash of every .joblib can
    # take minutes on ``build_app()`` cold-start, blocking the Voilà
    # "Executing 2 of 2" cell. Table is populated on first refresh click.
    table_html = w.HTML(_INITIAL_PLACEHOLDER)

    def _refresh(_b: Any = None) -> None:
        table_html.value = _render_table()

    refresh_btn.on_click(_refresh)

    header = w.HBox(
        [
            w.HTML("<div class='brt-overlay-title'>📦 Артефакты моделей</div>"),
            w.HBox([refresh_btn, close_btn]),
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


def _scan_artifacts() -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for root_str in ROOT_CANDIDATES:
        root = Path(root_str).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*.joblib"):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return out


def _render_table() -> str:
    paths = _scan_artifacts()
    if not paths:
        roots = " · ".join(html.escape(r) for r in ROOT_CANDIDATES)
        return (
            "<div class='muted' style='padding:18px;text-align:center'>"
            f"— нет .joblib файлов в: {roots} —"
            "</div>"
        )

    rows = []
    for path in paths[:50]:  # cap to keep DOM cheap
        try:
            stat = path.stat()
        except OSError:
            continue
        size_str = _fmt_bytes(stat.st_size)
        sha256 = _sha256_short(path)
        name = html.escape(path.name)
        parent = html.escape(str(path.parent))
        rows.append(
            "<tr>"
            f"<td style='font-family:monospace'>{name}</td>"
            f"<td style='font-family:monospace;font-size:11px;color:#6db39a'>{parent}</td>"
            f"<td>{size_str}</td>"
            f"<td><code style='font-size:11px'>{sha256}</code></td>"
            "</tr>"
        )
    return (
        "<table class='brt-pred-table'>"
        "<thead><tr>"
        "<th>Файл</th><th>Папка</th><th>Размер</th><th>SHA-256 (16)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _sha256_short(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                buf = fh.read(chunk)
                if not buf:
                    break
                h.update(buf)
    except OSError:
        return "—"
    return h.hexdigest()[:16]


def _fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
