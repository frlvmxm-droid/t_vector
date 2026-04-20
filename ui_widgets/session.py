"""Persistent session state for the Voilà dashboard.

The desktop app writes its last snapshot to
``~/.classification_tool/last_session.json`` via
``app.App._save_session`` / ``_restore_session``; the web-UI never did.
This module offers a tkinter-free parallel used by ``notebook_app``
for the same round-trip.

Transient keys (uploaded file bytes, active-worker flags, tmp-paths)
are filtered before writing — we only persist UI-editable widget
values.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_DEFAULT_DIR = Path.home() / ".classification_tool"
_DEFAULT_FILE = _DEFAULT_DIR / "last_session.json"

# Prefixes/keys that should never be persisted. Uploaded bytes are large
# and user-specific; tmp-paths break across restarts; active-worker flags
# reflect momentary runtime state that must start fresh after reload.
_TRANSIENT_PREFIXES: tuple[str, ...] = (
    "_upload_bytes_",
    "_tmp_",
    "_worker_",
    "_active_",
)
_TRANSIENT_KEYS: frozenset[str] = frozenset(
    {
        "uploaded_bytes",
        "upload_bytes",
        "tmp_path",
        "active_worker",
        "is_running",
    }
)


def _is_transient(key: str) -> bool:
    if key in _TRANSIENT_KEYS:
        return True
    return any(key.startswith(p) for p in _TRANSIENT_PREFIXES)


def _strip_transient(snap: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in snap.items() if not _is_transient(k)}


def _is_json_safe(value: Any) -> bool:
    """True if ``value`` survives a json round-trip without coercion loss."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False


def _json_safe_only(snap: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in snap.items() if _is_json_safe(v)}


def session_path() -> Path:
    """Return the default session-file path (``~/.classification_tool/last_session.json``)."""
    return _DEFAULT_FILE


def save_session(snap: Mapping[str, Any], *, path: Path | None = None) -> Path:
    """Persist ``snap`` atomically (tmp-file + rename).

    Transient keys and non-JSON-safe values are filtered out. The
    on-disk layout is ``{"schema_version": 1, "snap": {...}}`` so future
    migrations can bump the version without breaking old readers.
    """
    target = path or _DEFAULT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "snap": _json_safe_only(_strip_transient(snap)),
    }

    # Write to a tmp file in the same directory so `os.replace` is atomic.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".session-", suffix=".json.tmp", dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def load_last_session(*, path: Path | None = None) -> dict[str, Any] | None:
    """Return the saved snap, or ``None`` if absent / unreadable / corrupt."""
    target = path or _DEFAULT_FILE
    if not target.is_file():
        return None
    try:
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    snap = payload.get("snap")
    if not isinstance(snap, dict):
        return None
    return dict(snap)


class DebouncedSaver:
    """Debounced ``save_session`` wrapper.

    Coalesces rapid ``schedule`` calls into a single write. Intended to
    be wired as an ``observe`` callback on every widget the panels
    expose; frequent slider changes therefore cost one disk write, not
    dozens. Cancel any pending flush with ``cancel``.
    """

    def __init__(
        self,
        get_snap: Any,
        *,
        delay_sec: float = 2.0,
        path: Path | None = None,
    ) -> None:
        self._get_snap = get_snap
        self._delay = float(delay_sec)
        self._path = path
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        try:
            snap = self._get_snap()
        except Exception:
            return
        if not isinstance(snap, Mapping):
            return
        try:
            save_session(snap, path=self._path)
        except Exception:
            pass

    def flush_now(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._flush()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
