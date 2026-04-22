# -*- coding: utf-8 -*-
"""File-IO helpers for the Voilà UI: upload → tmp, result → download link."""
from __future__ import annotations

import base64
import mimetypes
import pathlib
import tempfile
from typing import Any, List, Optional


def save_upload_to_tmp(
    upload_widget: Any,
    dest_dir: Optional[pathlib.Path] = None,
) -> List[pathlib.Path]:
    """Persist files from ``ipywidgets.FileUpload`` to a tmp directory.

    ipywidgets 8.x: ``upload.value`` is a tuple of dicts
        ``{'name': str, 'type': str, 'size': int, 'content': memoryview}``.
    Returns the list of written paths (preserves upload order).
    """
    dest = dest_dir or pathlib.Path(tempfile.mkdtemp(prefix="brt_upload_"))
    dest.mkdir(parents=True, exist_ok=True)
    paths: List[pathlib.Path] = []
    for item in upload_widget.value:
        name = item.get("name") or "upload.bin"
        content = item.get("content")
        if content is None:
            continue
        p = dest / name
        p.write_bytes(bytes(content))
        paths.append(p)
    return paths


def download_link(
    path: pathlib.Path,
    label: Optional[str] = None,
) -> Any:
    """Return an ``ipywidgets.HTML`` with a base64 data-URL download anchor.

    Works without a server-side endpoint — the file contents are inlined
    in the DOM. Fine for <100 MB artifacts (model bundles, CSVs).
    """
    import ipywidgets as w

    p = pathlib.Path(path)
    data = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(data).decode("ascii")
    href = f"data:{mime};base64,{b64}"
    text = label or f"Скачать {p.name}"
    style = (
        "display:inline-block;padding:6px 14px;border:1px solid #4a90e2;"
        "border-radius:4px;color:#4a90e2;text-decoration:none;font-weight:600;"
    )
    html = (
        f'<a download="{p.name}" href="{href}" style="{style}">{text}</a>'
        f' <span style="color:#888;font-size:0.9em;">({_human_size(len(data))})</span>'
    )
    return w.HTML(value=html)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def detect_tabular_format(path: pathlib.Path) -> str:
    """Return ``'xlsx'`` or ``'csv'`` based on extension. Raises for unknown."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return "xlsx"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Неизвестный формат файла: {suffix!r}. Нужен .xlsx/.csv")
