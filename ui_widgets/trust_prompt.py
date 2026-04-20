"""Voilà-compatible trust-store hook for `.joblib` bundles.

Wraps ``model_loader.ensure_trusted_model_path`` so that the web-UI
Apply panel prompts the user before loading an unfamiliar bundle, in
parity with the desktop Tk dialog and the CLI ``require_trusted=True``
flow.

Public API
----------
``ensure_trusted_model_path_interactive(path, *, log_cb, confirm_cb)``
    Returns the trusted ``Path`` or raises :class:`TrustDenied`.

``build_confirm_prompt()``
    Creates an ipywidgets host + a blocking ``confirm_cb`` pair. The
    host is a ``VBox`` you place in the panel; ``confirm_cb(label) ->
    bool`` is invoked from the worker thread and blocks on a
    ``threading.Event`` until the user clicks Yes/No or the timeout
    fires.

``get_trust_store()``
    Returns the kernel-wide :class:`TrustStore` singleton. Use its
    ``trusted_canonical_paths()`` / ``get_hash()`` to pass
    ``trusted_paths=`` / ``precomputed_sha256=`` into
    ``load_model_artifact`` once trust is established.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from model_loader import (
    TrustStore,
    ensure_trusted_model_path,
    file_sha256,
)

DEFAULT_TIMEOUT_SEC = 60.0

LogCB = Callable[[str], None]
ConfirmCB = Callable[[str], bool]


class TrustDenied(Exception):
    """User declined (or timed out on) the trust prompt."""


_STORE = TrustStore()


def get_trust_store() -> TrustStore:
    """Return the kernel-wide TrustStore singleton."""
    return _STORE


def _format_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


def build_confirm_prompt(
    *,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> tuple[Any, ConfirmCB]:
    """Return ``(prompt_host_widget, confirm_cb)``.

    The host widget must be placed somewhere visible in the panel.
    ``confirm_cb(label)`` is thread-safe: it fills the host with a
    Yes/No button pair, blocks the caller on a ``threading.Event``, and
    returns the user's decision. Call from a worker thread only.
    """
    import ipywidgets as w

    host = w.VBox([], layout=w.Layout(margin="6px 0 0 0"))

    def _confirm(label: str) -> bool:
        done = threading.Event()
        decision: list[bool] = [False]

        yes_btn = w.Button(
            description="✅ Да, загрузить",
            button_style="warning",
            layout=w.Layout(width="200px"),
        )
        no_btn = w.Button(
            description="❌ Отмена",
            layout=w.Layout(width="140px"),
        )

        def _yes(_b: Any) -> None:
            decision[0] = True
            done.set()

        def _no(_b: Any) -> None:
            decision[0] = False
            done.set()

        yes_btn.on_click(_yes)
        no_btn.on_click(_no)

        banner = w.HTML(
            "<div style='padding:10px;border:1px solid #f97316;"
            "border-radius:6px;background:#1a1f26;color:#f5a96b'>"
            f"⚠ Файл <code>{label}</code> (.joblib) содержит "
            "сериализованный Python-код.<br>"
            "Загружай <b>только</b> из доверенных источников — "
            "вредоносный файл может выполнить произвольный код.<br>"
            "SHA-256 зафиксируется для этой сессии; при изменении файла "
            "на диске подтверждение запросится повторно.</div>"
        )
        host.children = (banner, w.HBox([yes_btn, no_btn]))

        try:
            if not done.wait(timeout=timeout_sec):
                return False
            return bool(decision[0])
        finally:
            host.children = ()

    return host, _confirm


def ensure_trusted_model_path_interactive(
    path: str | Path,
    *,
    log_cb: LogCB,
    confirm_cb: ConfirmCB,
) -> Path:
    """Ensure ``path`` is in the session trust store, prompting if new.

    Returns the resolved ``Path`` when trust is established. Raises
    :class:`TrustDenied` if the user declines or the prompt times out.
    Side effect: adds the path (with its current SHA-256) to the
    kernel-wide store, so subsequent calls are silent.
    """
    p = Path(path)
    already_trusted = _STORE.is_trusted(str(p))

    if not already_trusted:
        try:
            sha = file_sha256(str(p))
            size = p.stat().st_size if p.exists() else 0
            log_cb(
                f"  trust-prompt: {p.name} · {_format_size(size)} · "
                f"sha256={sha[:16]}…"
            )
        except OSError as exc:
            log_cb(f"  trust-prompt: не удалось посчитать SHA-256: {exc}")

    approved = ensure_trusted_model_path(
        _STORE,
        str(p),
        label=p.name,
        confirm_fn=confirm_cb,
    )
    if not approved:
        raise TrustDenied(
            f"Пользователь не подтвердил загрузку модели: {p.name}"
        )
    return p
