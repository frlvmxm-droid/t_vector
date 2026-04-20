"""Unit tests for `cluster_summarization_service` (Phase 11 port).

Note: we never actually load a 1 GB T5 model in CI — instead we
monkey-patch ``T5RussianSummarizer`` to a controllable stub. This keeps
the test fast and offline-safe.
"""
from __future__ import annotations

from collections.abc import Sequence

import pytest

import cluster_summarization_service as svc
from cluster_summarization_service import summarize_clusters_with_t5


class _StubSummarizer:
    """Deterministic stand-in for T5RussianSummarizer used in tests."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def summarize(self, texts: Sequence[str]) -> list[str]:
        return [f"summary[{len(t)}]" for t in texts]


class _LoadFailingSummarizer(_StubSummarizer):
    def load(self) -> None:
        raise ImportError("transformers not installed")


class _BatchFailingSummarizer(_StubSummarizer):
    def summarize(self, texts: Sequence[str]) -> list[str]:
        # First call (batch) fails; per-cluster calls succeed.
        if len(texts) > 1:
            raise RuntimeError("batch oom")
        return [f"single[{len(texts[0])}]"]


def test_summarize_returns_per_cluster_dict(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "t5_summarizer.T5RussianSummarizer",
        _StubSummarizer,
        raising=False,
    )
    texts_by_cluster = {
        0: ["short", "tiny"],
        1: ["a longer paragraph for cluster one", "another one"],
    }
    out = summarize_clusters_with_t5(texts_by_cluster)
    assert set(out.keys()) == {0, 1}
    assert all(v.startswith("summary[") for v in out.values())


def test_summarize_skips_negative_cluster_ids(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "t5_summarizer.T5RussianSummarizer",
        _StubSummarizer,
        raising=False,
    )
    out = summarize_clusters_with_t5({-1: ["noise"], 0: ["x"], 1: ["y"]})
    assert -1 not in out
    assert set(out.keys()) == {0, 1}


def test_summarize_progress_callback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "t5_summarizer.T5RussianSummarizer",
        _StubSummarizer,
        raising=False,
    )
    calls = []
    out = summarize_clusters_with_t5(
        {0: ["a"], 1: ["b"], 2: ["c"]},
        progress_cb=lambda f, m: calls.append((f, m)),
    )
    assert len(out) == 3
    assert len(calls) == 3
    assert calls[-1][0] == pytest.approx(1.0)


def test_summarize_returns_empty_when_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """Importing T5RussianSummarizer raises ImportError → graceful skip."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "t5_summarizer":
            raise ImportError("no transformers")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    log_lines: list[str] = []
    out = summarize_clusters_with_t5(
        {0: ["a"]}, log_cb=log_lines.append,
    )
    assert out == {}
    assert any("T5 пропущен" in line for line in log_lines)


def test_summarize_skips_when_load_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "t5_summarizer.T5RussianSummarizer",
        _LoadFailingSummarizer,
        raising=False,
    )
    log_lines: list[str] = []
    out = summarize_clusters_with_t5({0: ["a"]}, log_cb=log_lines.append)
    assert out == {}
    assert any("T5 пропущен" in line for line in log_lines)


def test_summarize_falls_back_per_cluster_on_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "t5_summarizer.T5RussianSummarizer",
        _BatchFailingSummarizer,
        raising=False,
    )
    out = summarize_clusters_with_t5({0: ["a"], 1: ["b"]})
    # Batch failed → fallback path produces "single[N]" for each.
    assert set(out.keys()) == {0, 1}
    assert all(v.startswith("single[") for v in out.values())


def test_summarize_empty_input_returns_empty():
    assert summarize_clusters_with_t5({}) == {}


def test_join_texts_truncates():
    long = ["x" * 5000]
    s = svc._join_texts(long, max_chars=200)
    assert len(s) <= 200
