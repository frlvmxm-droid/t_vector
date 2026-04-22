# -*- coding: utf-8 -*-
"""Tests for the MinHash LSH path in detect_near_duplicate_conflicts
and for the opt-in SBERT weight-embedding pickling round-trip.
"""
from __future__ import annotations

import io
import os
import pickle
import tarfile
import tempfile

import pytest

from ml_diagnostics import (
    _datasketch_available,
    _minhash_candidate_pairs,
    detect_near_duplicate_conflicts,
)
from ml_vectorizers import (
    PerFieldSBERTVectorizer,
    SBERTVectorizer,
    _extract_embedded_st_model,
    _sbert_make_state,
    _sbert_restore_defaults,
    _serialize_st_model,
)


# ── LSH near-duplicate detection ─────────────────────────────────────────────


def test_lsh_disabled_falls_back_to_naive():
    X = ["клиент спрашивает про кредит"] * 10
    y = ["A"] * 5 + ["B"] * 5
    pairs = detect_near_duplicate_conflicts(X, y, threshold=0.9, use_lsh=False)
    assert len(pairs) > 0
    for t1, t2, l1, l2, sim in pairs:
        assert l1 != l2
        assert sim >= 0.9


@pytest.mark.skipif(not _datasketch_available(),
                    reason="datasketch optional dep not installed")
def test_lsh_produces_same_pair_set_as_naive():
    base = [f"банковский запрос клиента номер {i}" for i in range(80)]
    X = base + base[:6]
    y = ["A"] * 80 + ["B"] * 6
    lsh = detect_near_duplicate_conflicts(X, y, threshold=0.9, use_lsh=True)
    naive = detect_near_duplicate_conflicts(X, y, threshold=0.9, use_lsh=False)

    def normalize(pairs):
        return sorted(
            (tuple(sorted((t1, t2))), l1 if l1 < l2 else l2, round(s, 2))
            for t1, t2, l1, l2, s in pairs
        )
    assert normalize(lsh) == normalize(naive)


@pytest.mark.skipif(not _datasketch_available(),
                    reason="datasketch optional dep not installed")
def test_lsh_auto_enabled_for_large_n(monkeypatch):
    """When n exceeds _LSH_N_THRESHOLD and datasketch is available, LSH is used."""
    import ml_diagnostics as md
    monkeypatch.setattr(md, "_LSH_N_THRESHOLD", 10)
    X = [f"текст документ {i}" for i in range(30)]
    X += X[:3]
    y = ["A"] * 30 + ["B"] * 3
    logs: list[str] = []
    detect_near_duplicate_conflicts(X, y, threshold=0.9, log_fn=logs.append)
    assert any("LSH-режим" in m for m in logs), logs


@pytest.mark.skipif(not _datasketch_available(),
                    reason="datasketch optional dep not installed")
def test_lsh_force_without_datasketch_falls_back(monkeypatch):
    """use_lsh=True but datasketch absent → auto-fallback to O(n²)."""
    import ml_diagnostics as md
    monkeypatch.setattr(md, "_datasketch_available", lambda: False)
    X = ["повторяющийся текст"] * 10
    y = ["A"] * 5 + ["B"] * 5
    logs: list[str] = []
    pairs = detect_near_duplicate_conflicts(
        X, y, threshold=0.9, use_lsh=True, log_fn=logs.append,
    )
    assert any("fallback" in m.lower() for m in logs)
    assert len(pairs) > 0


@pytest.mark.skipif(not _datasketch_available(),
                    reason="datasketch optional dep not installed")
def test_minhash_candidate_pairs_respects_label_constraint():
    """Candidates must satisfy (i < j) and y[i] != y[j]."""
    texts = ["текст " + "одинаковый" * 5] * 4
    labels = ["A", "A", "B", "B"]
    cand = _minhash_candidate_pairs(texts, labels, jaccard_threshold=0.3)
    for i, j in cand:
        assert i < j
        assert labels[i] != labels[j]


def test_detect_duplicates_empty_input():
    assert detect_near_duplicate_conflicts([], [], threshold=0.9) == []
    assert detect_near_duplicate_conflicts(["only"], ["A"], threshold=0.9) == []


def test_detect_duplicates_zero_threshold_returns_empty():
    assert detect_near_duplicate_conflicts(
        ["a", "b", "c"], ["A", "B", "C"], threshold=0.0,
    ) == []


# ── Shared pickling helpers (DRY) ────────────────────────────────────────────


def test_sbert_make_state_blanks_callbacks_and_model():
    src = {"log_cb": lambda m: None, "progress_cb": lambda p, m: None,
           "_model": object(), "model_name": "x", "batch_size": 32}
    state = _sbert_make_state(src, "_model")
    assert state["log_cb"] is None
    assert state["progress_cb"] is None
    assert state["_model"] is None
    assert state["model_name"] == "x"
    assert state["batch_size"] == 32


def test_sbert_restore_defaults_backfills_missing_attrs():
    class Dummy:
        pass
    d = Dummy()
    d.model_name = "m"
    _sbert_restore_defaults(d, "_model", device="auto", embed_weights=False)
    assert d._model is None
    assert d.log_cb is None
    assert d.progress_cb is None
    assert d.device == "auto"
    assert d.embed_weights is False


def test_sbert_restore_defaults_preserves_existing():
    class Dummy:
        pass
    d = Dummy()
    d.device = "cuda:0"
    d.embed_weights = True
    _sbert_restore_defaults(d, "_model", device="auto", embed_weights=False)
    assert d.device == "cuda:0"
    assert d.embed_weights is True


# ── SBERT opt-in weight embedding ────────────────────────────────────────────


def test_sbert_default_does_not_embed_weights():
    v = SBERTVectorizer(model_name="test/model", batch_size=16)
    assert v.embed_weights is False
    assert v._model_archive is None
    data = pickle.dumps(v)
    v2 = pickle.loads(data)
    assert v2.model_name == "test/model"
    assert v2._model_archive is None
    assert v2.embed_weights is False


def test_sbert_embed_weights_flag_survives_pickle():
    v = SBERTVectorizer(embed_weights=True)
    data = pickle.dumps(v)
    v2 = pickle.loads(data)
    assert v2.embed_weights is True
    # Model was never loaded → archive stays None
    assert v2._model_archive is None


def test_sbert_callbacks_are_cleared_on_pickle():
    v = SBERTVectorizer(log_cb=lambda m: None, progress_cb=lambda p, m: None)
    data = pickle.dumps(v)
    v2 = pickle.loads(data)
    assert v2.log_cb is None
    assert v2.progress_cb is None


def test_sbert_backward_compat_missing_new_attrs():
    """State from older versions without embed_weights / _model_archive must load cleanly."""
    v = SBERTVectorizer()
    state = v.__dict__.copy()
    state.pop("embed_weights", None)
    state.pop("_model_archive", None)
    state.pop("device", None)
    revived = SBERTVectorizer.__new__(SBERTVectorizer)
    revived.__setstate__(state)
    assert revived.embed_weights is False
    assert revived._model_archive is None
    assert revived.device == "auto"


def test_perfield_sbert_embed_weights_forward():
    pf = PerFieldSBERTVectorizer(embed_weights=True, normalize=True)
    data = pickle.dumps(pf)
    pf2 = pickle.loads(data)
    assert pf2.embed_weights is True
    assert pf2._sbert is None
    assert pf2.log_cb is None


def test_perfield_sbert_backward_compat():
    """Older PerFieldSBERTVectorizer state (no embed_weights/normalize) loads safely."""
    pf = PerFieldSBERTVectorizer()
    state = pf.__dict__.copy()
    state.pop("embed_weights", None)
    state.pop("_model_archive", None)
    state.pop("normalize", None)
    state.pop("device", None)
    revived = PerFieldSBERTVectorizer.__new__(PerFieldSBERTVectorizer)
    revived.__setstate__(state)
    assert revived.embed_weights is False
    assert revived.normalize is True
    assert revived.device == "auto"


def test_serialize_st_model_returns_none_for_no_model():
    assert _serialize_st_model(None) is None


def test_extract_embedded_st_model_idempotent():
    """Same archive bytes must extract to the same path without re-extracting."""
    with tempfile.TemporaryDirectory() as src:
        # Build a minimal tar.gz with one file
        with open(os.path.join(src, "config.json"), "w") as f:
            f.write('{"hidden_size": 8}')
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(src, arcname=".")
        data = buf.getvalue()

    try:
        target1 = _extract_embedded_st_model(data)
        target2 = _extract_embedded_st_model(data)
        assert target1 == target2
        assert (target1 / "config.json").exists()
        assert (target1 / ".extracted_ok").exists()
        # Second call must not fail even though directory exists
    finally:
        import shutil
        if target1.exists():
            shutil.rmtree(target1, ignore_errors=True)


def test_extract_embedded_st_model_different_bytes_different_paths():
    def _pack(content: str) -> bytes:
        with tempfile.TemporaryDirectory() as src:
            with open(os.path.join(src, "config.json"), "w") as f:
                f.write(content)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(src, arcname=".")
            return buf.getvalue()

    data_a = _pack('{"hidden_size": 8}')
    data_b = _pack('{"hidden_size": 16}')
    try:
        target_a = _extract_embedded_st_model(data_a)
        target_b = _extract_embedded_st_model(data_b)
        assert target_a != target_b
    finally:
        import shutil
        for t in (target_a, target_b):
            if t.exists():
                shutil.rmtree(t, ignore_errors=True)
