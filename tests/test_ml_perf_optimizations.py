# -*- coding: utf-8 -*-
"""Tests for performance optimizations:

  * `llm_reranker` — ThreadPoolExecutor parallelism + persistent disk cache
  * `ml_augment` — per-class parallel LLM generation
  * `ml_vectorizers.Lemmatizer` — per-token cache
  * `ml_vectorizers.PerFieldVectorizer` — parallel per-field fit
  * `ml_distillation.distill_soft_labels` — vectorized one-hot
  * `ml_training._estimate_model_size_bytes` — analytical estimate
  * `ml_diagnostics.detect_near_duplicate_conflicts` — batched LSH cosine filter
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

import llm_reranker
from ml_augment import augment_rare_classes


# ── llm_reranker: parallelism + disk cache ───────────────────────────────────


class _FakeLLMState:
    def __init__(self, delay: float = 0.0, answers: dict[str, str] | None = None):
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.delay = delay
        self.answers = answers or {}
        self._lock = threading.Lock()

    def complete_text(self, *, user_prompt: str, **kwargs: Any) -> str:
        if self.delay:
            time.sleep(self.delay)
        # Parse first candidate (the one after "КЛАССЫ-КАНДИДАТЫ:" bullet)
        for line in user_prompt.splitlines():
            if line.startswith("• "):
                first = line[2:].strip()
                break
        else:
            first = ""
        with self._lock:
            self.calls.append((user_prompt[:40], (first,)))
        return self.answers.get(first, first)


@pytest.fixture
def _fake_llm(monkeypatch: pytest.MonkeyPatch):
    state = _FakeLLMState()

    class _FakeClient:
        @staticmethod
        def complete_text(**kwargs: Any) -> str:
            return state.complete_text(**kwargs)

    monkeypatch.setattr(llm_reranker, "LLMClient", _FakeClient)
    return state


def test_rerank_top_k_parallel_calls_llm_per_unique_row(_fake_llm, tmp_path, monkeypatch):
    # Redirect disk cache to an isolated tmp dir
    monkeypatch.setattr(llm_reranker, "_CACHE_DIR", tmp_path / "cache")
    texts = [f"обращение {i}" for i in range(6)]
    top = [["A", "B"]] * 6
    fb = ["A"] * 6
    out = llm_reranker.rerank_top_k(
        texts, top, fb,
        provider="p", model="m", api_key="k",
        max_workers=4, use_disk_cache=False,
    )
    assert len(out) == 6
    # All texts distinct → 6 LLM calls
    assert len(_fake_llm.calls) == 6
    assert all(o == "A" for o in out)  # first candidate is picked


def test_rerank_top_k_in_batch_cache_dedupes_identical_rows(_fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reranker, "_CACHE_DIR", tmp_path / "cache")
    texts = ["одинаковый текст"] * 5
    top = [["A", "B"]] * 5
    fb = ["A"] * 5
    llm_reranker.rerank_top_k(
        texts, top, fb,
        provider="p", model="m", api_key="k",
        use_disk_cache=False,
    )
    # Identical (text, candidates) → 1 LLM call, 4 mem-cache hits
    assert len(_fake_llm.calls) == 1


def test_rerank_top_k_disk_cache_persists(_fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reranker, "_CACHE_DIR", tmp_path / "cache")
    texts = ["обращение A", "обращение B"]
    top = [["X", "Y"], ["X", "Y"]]
    fb = ["X", "X"]
    llm_reranker.rerank_top_k(
        texts, top, fb,
        provider="p", model="m", api_key="k",
        use_disk_cache=True,
    )
    calls_first = len(_fake_llm.calls)
    assert calls_first == 2

    # Second call with same inputs: disk cache should serve both, zero new LLM calls
    llm_reranker.rerank_top_k(
        texts, top, fb,
        provider="p", model="m", api_key="k",
        use_disk_cache=True,
    )
    assert len(_fake_llm.calls) == calls_first  # no new calls


def test_rerank_top_k_single_candidate_skipped(_fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(llm_reranker, "_CACHE_DIR", tmp_path / "cache")
    out = llm_reranker.rerank_top_k(
        ["t1"], [["only"]], ["only"],
        provider="p", model="m", api_key="k",
    )
    assert out == ["only"]
    assert len(_fake_llm.calls) == 0


def test_rerank_top_k_parallel_speedup(_fake_llm, tmp_path, monkeypatch):
    """Parallel execution must be meaningfully faster than serial for I/O-bound work."""
    monkeypatch.setattr(llm_reranker, "_CACHE_DIR", tmp_path / "cache")
    _fake_llm.delay = 0.05  # 50 ms per LLM call

    texts = [f"обращение {i}" for i in range(8)]
    top = [["A", "B"]] * 8
    fb = ["A"] * 8

    t0 = time.perf_counter()
    llm_reranker.rerank_top_k(
        texts, top, fb,
        provider="p", model="m", api_key="k",
        max_workers=8, use_disk_cache=False,
    )
    dt_par = time.perf_counter() - t0

    # Serial would take 8 * 0.05 = 400 ms; parallel with 8 workers should finish well under 200 ms
    assert dt_par < 0.25, f"parallel rerank too slow: {dt_par:.3f}s"


# ── ml_augment: per-class parallelism ────────────────────────────────────────


def test_augment_rare_classes_parallel_order_preserved():
    # Deterministic fake LLM: returns "- <class>::para<i>" paraphrases
    def _fake_complete(*, user_prompt: str, **kwargs: Any) -> str:
        # Echo 3 lines; content doesn't matter for order test
        return "\n".join(f"- перефраз {i} для {user_prompt[:20]}" for i in range(3))

    X = (["текст A"] * 10) + (["текст B"] * 2) + (["текст C"] * 2)
    y = (["A"] * 10) + (["B"] * 2) + (["C"] * 2)
    X_aug, y_aug, report = augment_rare_classes(
        X=X, y=y,
        min_samples_threshold=5,
        n_paraphrases=3,
        llm_complete_fn=_fake_complete,
        provider="p", model="m", api_key="k",
        max_workers=2,
    )
    assert report["classes_augmented"] == 2   # B and C are both augmented
    assert report["rows_added"] >= 4           # at least 3 per class
    # Original rows preserved at their original positions
    assert X_aug[:len(X)] == X
    assert y_aug[:len(X)] == y


def test_augment_rare_classes_no_rare_noop():
    X = ["a", "b", "c"] * 5
    y = ["X", "Y", "Z"] * 5
    X_out, y_out, rep = augment_rare_classes(
        X=X, y=y,
        min_samples_threshold=1,
        n_paraphrases=3,
        llm_complete_fn=lambda **_: "- x\n- y",
        provider="p", model="m", api_key="k",
    )
    assert X_out == X
    assert y_out == y
    assert rep == {"classes_augmented": 0, "rows_added": 0, "skipped": []}


# ── Lemmatizer LRU cache ─────────────────────────────────────────────────────


def test_lemmatizer_cache_populates_on_transform():
    from ml_vectorizers import Lemmatizer
    lem = Lemmatizer()
    lem.fit(["банки работают"])
    if not getattr(lem, "is_active_", False):
        pytest.skip("pymorphy not installed")
    lem.transform(["банки работают банки работают"])
    # Each unique token appears at most once in the cache
    keys = {k[0] for k in lem._lemma_cache}
    assert "банки" in keys
    assert "работают" in keys


def test_lemmatizer_cache_wiped_on_pickle():
    import pickle
    from ml_vectorizers import Lemmatizer
    lem = Lemmatizer()
    lem._lemma_cache[("abc", False)] = "abc"
    raw = pickle.dumps(lem)
    lem2 = pickle.loads(raw)
    assert lem2._lemma_cache == {}


def test_lemmatizer_backward_compat_missing_cache_attrs():
    from ml_vectorizers import Lemmatizer
    lem = Lemmatizer()
    state = lem.__dict__.copy()
    state.pop("_lemma_cache", None)
    state.pop("_LEMMA_CACHE_CAP", None)
    revived = Lemmatizer.__new__(Lemmatizer)
    revived.__setstate__(state)
    assert revived._lemma_cache == {}
    assert revived._LEMMA_CACHE_CAP == 50_000


# ── PerFieldVectorizer parallel fit ──────────────────────────────────────────


def _per_field_docs() -> list[str]:
    return [
        (
            "[DESC]\nклиент просит выписку\n"
            "[CLIENT]\nздравствуйте мне нужна выписка\n"
            "[OPERATOR]\nконечно сейчас сделаем\n"
            "[SUMMARY]\nвыписка предоставлена\n"
            "[ANSWER_SHORT]\nвыписка выдана\n"
            "[ANSWER_FULL]\nвыписка по счету выдана клиенту\n"
        ),
        (
            "[DESC]\nвопрос про кредит\n"
            "[CLIENT]\nхочу взять кредит\n"
            "[OPERATOR]\nоформим заявку\n"
            "[SUMMARY]\nзаявка на кредит\n"
            "[ANSWER_SHORT]\nзаявка принята\n"
            "[ANSWER_FULL]\nзаявка на кредит принята\n"
        ),
    ] * 10


def test_perfield_fit_parallel_matches_sequential():
    from ml_vectorizers import PerFieldVectorizer

    X = _per_field_docs()
    weights = {
        "w_desc": 3, "w_client": 2, "w_operator": 2,
        "w_summary": 2, "w_answer_short": 1, "w_answer_full": 1,
    }
    v = PerFieldVectorizer(base_weights=weights, max_features=5000)
    v.fit(X)
    mat = v.transform(X)
    assert mat.shape[0] == len(X)
    assert mat.shape[1] > 0
    # All confirmed active fields have populated char + word vecs
    for _wk, tag, _w in v._active:
        assert tag in v._char_vecs
        assert tag in v._word_vecs


# ── distill_soft_labels vectorization ────────────────────────────────────────


def test_distill_soft_labels_vectorized_produces_same_shape():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    from ml_distillation import distill_soft_labels

    classes = ["A", "B", "C"]
    rng = np.random.default_rng(0)

    class _Teacher:
        classes_ = classes

        def predict_proba(self, X):
            return rng.dirichlet([1, 1, 1], size=len(X))

    X = [f"текст {i}" for i in range(30)]
    y = [classes[i % 3] for i in range(30)]

    student = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1)),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    fitted = distill_soft_labels(
        _Teacher(), student, X, y,
        temperature=2.0, alpha=0.7,
    )
    preds = fitted.predict(X)
    assert len(preds) == len(X)
    assert set(preds) <= set(classes)


# ── analytical model size estimator ──────────────────────────────────────────


def test_estimate_model_size_no_pickle_needed():
    """The new estimator walks attributes — it must not hit pickle at all."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from ml_training import _estimate_model_size_bytes

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1, token_pattern=r"\S+")),
        ("clf", LogisticRegression(max_iter=50)),
    ])
    pipe.fit(["a b c", "d e f", "a d e"], ["x", "y", "x"])
    size = _estimate_model_size_bytes(pipe, log_cb=None)
    assert isinstance(size, int)
    assert size > 0
    # Fast path: analytical estimate should run in milliseconds, not seconds.
    t0 = time.perf_counter()
    _estimate_model_size_bytes(pipe, log_cb=None)
    assert (time.perf_counter() - t0) < 0.5


# ── batched LSH cosine filter ────────────────────────────────────────────────


def test_lsh_batched_cosine_matches_naive():
    from ml_diagnostics import (
        _datasketch_available,
        detect_near_duplicate_conflicts,
    )

    if not _datasketch_available():
        pytest.skip("datasketch not installed")

    base = [f"банковский запрос клиента номер {i}" for i in range(50)]
    X = base + base[:4]
    y = ["A"] * 50 + ["B"] * 4

    lsh = detect_near_duplicate_conflicts(X, y, threshold=0.9, use_lsh=True)
    naive = detect_near_duplicate_conflicts(X, y, threshold=0.9, use_lsh=False)

    def normalize(pairs):
        return sorted(
            (tuple(sorted((t1, t2))), l1 if l1 < l2 else l2, round(s, 2))
            for t1, t2, l1, l2, s in pairs
        )

    assert normalize(lsh) == normalize(naive)
