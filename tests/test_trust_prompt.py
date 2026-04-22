"""Unit tests for ui_widgets.trust_prompt (Voilà trust-store hook)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ui_widgets.trust_prompt import (
    TrustDenied,
    ensure_trusted_model_path_interactive,
    get_trust_store,
)


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    store = get_trust_store()
    store._trusted_paths.clear()
    store._trusted_hashes.clear()
    store._revoked_paths.clear()


def _write_dummy(tmp_path: Path) -> Path:
    p = tmp_path / "bundle.joblib"
    p.write_bytes(b"\x00\x01\x02")
    return p


def test_new_path_invokes_confirm(tmp_path: Path) -> None:
    p = _write_dummy(tmp_path)
    calls: list[str] = []

    def confirm_cb(label: str) -> bool:
        calls.append(label)
        return True

    logs: list[str] = []
    result = ensure_trusted_model_path_interactive(
        p, log_cb=logs.append, confirm_cb=confirm_cb,
    )
    assert result == p
    assert calls == ["bundle.joblib"]
    assert any("sha256=" in line for line in logs)
    assert get_trust_store().is_trusted(str(p))


def test_trusted_path_is_silent(tmp_path: Path) -> None:
    p = _write_dummy(tmp_path)
    calls: list[str] = []

    def confirm_cb(label: str) -> bool:
        calls.append(label)
        return True

    ensure_trusted_model_path_interactive(p, log_cb=lambda _: None, confirm_cb=confirm_cb)
    # Second call: already trusted → confirm_cb must NOT be invoked.
    ensure_trusted_model_path_interactive(p, log_cb=lambda _: None, confirm_cb=confirm_cb)
    assert calls == ["bundle.joblib"]


def test_decline_raises_trust_denied(tmp_path: Path) -> None:
    p = _write_dummy(tmp_path)

    with pytest.raises(TrustDenied):
        ensure_trusted_model_path_interactive(
            p, log_cb=lambda _: None, confirm_cb=lambda _lbl: False,
        )
    assert not get_trust_store().is_trusted(str(p))


def test_sha256_recorded_in_store(tmp_path: Path) -> None:
    p = _write_dummy(tmp_path)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()

    ensure_trusted_model_path_interactive(
        p, log_cb=lambda _: None, confirm_cb=lambda _lbl: True,
    )
    assert get_trust_store().get_hash(str(p)) == expected


def test_hash_change_reprompts(tmp_path: Path) -> None:
    p = _write_dummy(tmp_path)
    calls: list[str] = []

    def confirm_cb(label: str) -> bool:
        calls.append(label)
        return True

    ensure_trusted_model_path_interactive(p, log_cb=lambda _: None, confirm_cb=confirm_cb)
    assert calls == ["bundle.joblib"]

    # Mutate file → TrustStore should clear trust on next call.
    p.write_bytes(b"\xFF" * 16)
    ensure_trusted_model_path_interactive(p, log_cb=lambda _: None, confirm_cb=confirm_cb)
    assert calls == ["bundle.joblib", "bundle.joblib"]


def test_revoked_path_is_rejected(tmp_path: Path) -> None:
    p = _write_dummy(tmp_path)
    store = get_trust_store()
    store.add_trusted(str(p), "abc")
    store.revoke(str(p))

    with pytest.raises(TrustDenied):
        ensure_trusted_model_path_interactive(
            p, log_cb=lambda _: None, confirm_cb=lambda _lbl: True,
        )
