# -*- coding: utf-8 -*-
"""Unit tests for ``ui_widgets.session``: ``ui.theme`` persistence.

Sprint 2.3 added the ``ui.theme`` key to the session snap written by
``DebouncedSaver`` every 2s. These tests lock in that the key survives
the json round-trip, is not filtered as transient, and lives alongside
existing widget-keyed state.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ui_widgets import session as session_mod
from ui_widgets.session import (
    _TRANSIENT_KEYS,
    _TRANSIENT_PREFIXES,
    _is_transient,
    load_last_session,
    save_session,
)


@pytest.fixture
def tmp_session(tmp_path: Path) -> Path:
    return tmp_path / "session.json"


# ── Classification of the ui.theme key ────────────────────────────────


def test_ui_theme_is_not_transient():
    assert not _is_transient("ui.theme")


def test_transient_filter_does_not_capture_ui_theme():
    # Defence against accidental regressions if somebody later adds a
    # "ui." prefix to _TRANSIENT_PREFIXES.
    assert "ui.theme" not in _TRANSIENT_KEYS
    assert not any("ui.theme".startswith(p) for p in _TRANSIENT_PREFIXES)


# ── Round-trip ────────────────────────────────────────────────────────


def test_save_load_ui_theme_round_trip(tmp_session: Path):
    save_session({"ui.theme": "paper"}, path=tmp_session)
    loaded = load_last_session(path=tmp_session)
    assert loaded == {"ui.theme": "paper"}


@pytest.mark.parametrize("palette", ["dark-teal", "paper", "amber-crt"])
def test_all_three_palettes_round_trip(tmp_session: Path, palette: str):
    save_session({"ui.theme": palette}, path=tmp_session)
    assert load_last_session(path=tmp_session) == {"ui.theme": palette}


def test_ui_theme_coexists_with_widget_keys(tmp_session: Path):
    snap = {
        "ui.theme": "amber-crt",
        "k_clusters": 8,
        "vec_mode": "sbert",
        "threshold": 0.5,
    }
    save_session(snap, path=tmp_session)
    assert load_last_session(path=tmp_session) == snap


def test_transient_keys_stripped_ui_theme_kept(tmp_session: Path):
    snap = {
        "ui.theme": "paper",
        "_upload_bytes_train": b"\x00" * 10,   # filtered — bytes and prefix
        "_tmp_path": "/tmp/xxx",               # filtered — prefix
        "active_worker": True,                 # filtered — explicit key
        "k_clusters": 8,                       # kept
    }
    save_session(snap, path=tmp_session)
    loaded = load_last_session(path=tmp_session)
    assert loaded is not None
    assert loaded.get("ui.theme") == "paper"
    assert loaded.get("k_clusters") == 8
    assert "_upload_bytes_train" not in loaded
    assert "_tmp_path" not in loaded
    assert "active_worker" not in loaded


# ── Persistence shape ─────────────────────────────────────────────────


def test_on_disk_payload_has_schema_version(tmp_session: Path):
    save_session({"ui.theme": "dark-teal"}, path=tmp_session)
    import json
    payload = json.loads(tmp_session.read_text("utf-8"))
    assert payload["schema_version"] == session_mod.SCHEMA_VERSION
    assert payload["snap"] == {"ui.theme": "dark-teal"}


def test_save_is_atomic_no_tmp_leak(tmp_session: Path):
    save_session({"ui.theme": "paper"}, path=tmp_session)
    # ``.session-*.json.tmp`` sidecar must not survive a successful write.
    leftovers = list(tmp_session.parent.glob(".session-*.json.tmp"))
    assert leftovers == []


def test_load_missing_file_returns_none(tmp_path: Path):
    assert load_last_session(path=tmp_path / "absent.json") is None


def test_load_corrupt_file_returns_none(tmp_session: Path):
    tmp_session.write_text("{not valid json", encoding="utf-8")
    assert load_last_session(path=tmp_session) is None
