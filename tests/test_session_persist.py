"""Unit tests for ui_widgets.session — save/restore round-trip."""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from ui_widgets.session import (
    DebouncedSaver,
    load_last_session,
    save_session,
)


def test_save_then_load_round_trip(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "session.json"
    snap: dict[str, Any] = {
        "cluster_algo": "kmeans",
        "k_clusters": 7,
        "merge_threshold": 0.85,
        "use_umap": True,
        "sbert_model": "cointegrated/rubert-tiny2",
    }
    save_session(snap, path=target)
    loaded = load_last_session(path=target)
    assert loaded == snap


def test_schema_version_present_on_disk(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "session.json"
    save_session({"foo": "bar"}, path=target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["snap"] == {"foo": "bar"}


def test_load_missing_returns_none(tmp_path: pathlib.Path) -> None:
    assert load_last_session(path=tmp_path / "nope.json") is None


def test_load_corrupt_returns_none(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "corrupt.json"
    target.write_text("this is not json", encoding="utf-8")
    assert load_last_session(path=target) is None


def test_load_empty_file_returns_none(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "empty.json"
    target.write_bytes(b"")
    assert load_last_session(path=target) is None


def test_load_non_dict_payload_returns_none(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "array.json"
    target.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_last_session(path=target) is None


def test_transient_keys_excluded(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "session.json"
    snap = {
        "k_clusters": 5,
        "_upload_bytes_train": b"binary-junk",
        "_tmp_path": "/tmp/abc",
        "_worker_active": True,
        "_active_flag": True,
        "uploaded_bytes": b"junk",
        "is_running": True,
    }
    save_session(snap, path=target)
    loaded = load_last_session(path=target)
    assert loaded == {"k_clusters": 5}


def test_non_json_safe_values_excluded(tmp_path: pathlib.Path) -> None:
    """bytes / sets / custom objects must be silently dropped."""
    target = tmp_path / "session.json"

    class _Ignored:
        pass

    snap = {
        "k_clusters": 5,
        "ignore_obj": _Ignored(),
        "raw_bytes": b"raw",
        "a_set": {1, 2},
        "nested_ok": {"a": 1, "b": [1, 2, "x"]},
    }
    save_session(snap, path=target)
    loaded = load_last_session(path=target)
    assert loaded == {"k_clusters": 5, "nested_ok": {"a": 1, "b": [1, 2, "x"]}}


def test_atomic_write_leaves_no_tmp_file(tmp_path: pathlib.Path) -> None:
    save_session({"foo": 1}, path=tmp_path / "session.json")
    leftovers = list(tmp_path.glob(".session-*"))
    assert leftovers == []


def test_overwrite_preserves_latest(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "session.json"
    save_session({"k_clusters": 3}, path=target)
    save_session({"k_clusters": 9, "algo": "kmeans"}, path=target)
    loaded = load_last_session(path=target)
    assert loaded == {"k_clusters": 9, "algo": "kmeans"}


def test_debounced_saver_flushes_once_after_delay(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "session.json"
    state = {"k_clusters": 1}
    saver = DebouncedSaver(lambda: state, delay_sec=0.05, path=target)
    # Schedule repeatedly within the debounce window.
    saver.schedule()
    state["k_clusters"] = 2
    saver.schedule()
    state["k_clusters"] = 7
    saver.schedule()
    time.sleep(0.15)
    # Only the final value should be persisted.
    loaded = load_last_session(path=target)
    assert loaded == {"k_clusters": 7}


def test_debounced_saver_cancel_prevents_write(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "session.json"
    saver = DebouncedSaver(lambda: {"x": 1}, delay_sec=0.1, path=target)
    saver.schedule()
    saver.cancel()
    time.sleep(0.2)
    assert load_last_session(path=target) is None


def test_debounced_saver_flush_now_is_synchronous(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "session.json"
    saver = DebouncedSaver(lambda: {"x": 42}, delay_sec=10.0, path=target)
    saver.schedule()
    saver.flush_now()
    loaded = load_last_session(path=target)
    assert loaded == {"x": 42}
