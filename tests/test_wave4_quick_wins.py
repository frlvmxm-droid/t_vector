# -*- coding: utf-8 -*-
"""Wave 4 (quick wins) regression tests.

Покрывает:
    • W4.1 — `fuzzy_string_dedup`: rapidfuzz путь и graceful fallback
            на exact-dedup (strip+casefold), если rapidfuzz недоступен.
    • W4.4 — `llm_reranker.rerank_top_k`: по умолчанию температура 0.2
            передаётся в `LLMClient.complete_text`; допускается явная
            передача другой температуры и None.
    • W4.5 — `_compute_brier_score` / `_compute_per_class_ece`: корректность
            на ручной one-hot калибровке.

Тесты не зависят от реальной LLM и от установленного rapidfuzz:
rapidfuzz-fallback воспроизводится через подмену модуля на sys.modules.
"""
from __future__ import annotations

import builtins
import pathlib
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# W4.1 — fuzzy_string_dedup
# ---------------------------------------------------------------------------

def test_fuzzy_dedup_empty_returns_empty():
    from ml_training import fuzzy_string_dedup

    X_out, y_out, removed = fuzzy_string_dedup([], [])
    assert X_out == []
    assert y_out == []
    assert removed == 0


def test_fuzzy_dedup_exact_path_when_rapidfuzz_missing():
    """Если rapidfuzz.ImportError — должен отработать exact strip+casefold dedup."""
    import ml_training

    real_import = builtins.__import__

    def _raise_for_rapidfuzz(name, *args, **kwargs):
        if name == "rapidfuzz" or name.startswith("rapidfuzz."):
            raise ImportError("simulated: rapidfuzz not installed")
        return real_import(name, *args, **kwargs)

    X = ["Привет, мир", "привет, МИР ", "другая строка", "другая строка"]
    y = ["A", "A", "B", "B"]
    with patch.object(builtins, "__import__", side_effect=_raise_for_rapidfuzz):
        X_out, y_out, removed = ml_training.fuzzy_string_dedup(X, y)

    # first "Привет, мир" + second "другая строка" — две уникальных пары
    assert len(X_out) == 2
    assert removed == 2
    assert y_out.count("A") == 1
    assert y_out.count("B") == 1


def test_fuzzy_dedup_exact_path_does_not_collapse_different_classes():
    """Одинаковый текст под разными классами не удаляется (fallback-ветка)."""
    import ml_training

    real_import = builtins.__import__

    def _raise_for_rapidfuzz(name, *args, **kwargs):
        if name == "rapidfuzz" or name.startswith("rapidfuzz."):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    X = ["одинаковый текст", "одинаковый текст"]
    y = ["класс_1", "класс_2"]
    with patch.object(builtins, "__import__", side_effect=_raise_for_rapidfuzz):
        X_out, y_out, removed = ml_training.fuzzy_string_dedup(X, y)

    assert len(X_out) == 2
    assert removed == 0
    assert set(y_out) == {"класс_1", "класс_2"}


def test_fuzzy_dedup_rapidfuzz_path_collapses_near_duplicates():
    """Если rapidfuzz есть — near-duplicates в одном классе схлопываются."""
    pytest.importorskip("rapidfuzz")
    from ml_training import fuzzy_string_dedup

    X = [
        "Не прошёл платёж, верните деньги",
        "не прошёл платёж верните деньги",   # near-dup: тот же смысл
        "Заблокирована карта, что делать",
    ]
    y = ["платёж", "платёж", "карта"]
    X_out, y_out, removed = fuzzy_string_dedup(X, y, threshold=85)
    assert removed == 1
    assert len(X_out) == 2
    # Остался один из платёж и единственная карта
    assert y_out.count("платёж") == 1
    assert y_out.count("карта") == 1


def test_fuzzy_dedup_log_callback_receives_message():
    """При удалении хотя бы одного дубликата должен быть один лог."""
    import ml_training

    real_import = builtins.__import__

    def _raise_for_rapidfuzz(name, *args, **kwargs):
        if name == "rapidfuzz" or name.startswith("rapidfuzz."):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    messages: list[str] = []
    X = ["дубль", "дубль"]
    y = ["a", "a"]
    with patch.object(builtins, "__import__", side_effect=_raise_for_rapidfuzz):
        ml_training.fuzzy_string_dedup(X, y, log_cb=messages.append)

    assert any("Dedup" in m for m in messages)


# ---------------------------------------------------------------------------
# W4.4 — rerank_top_k temperature default
# ---------------------------------------------------------------------------

def test_rerank_top_k_passes_default_temperature_0_2():
    """По умолчанию `rerank_top_k` должен передать temperature=0.2."""
    from llm_reranker import rerank_top_k

    with patch("llm_reranker.LLMClient.complete_text", return_value="cls_a") as mock_llm:
        rerank_top_k(
            texts=["обращение"],
            top_candidates=[["cls_a", "cls_b"]],
            argmax_labels=["cls_b"],
            provider="openai", model="gpt-4o-mini", api_key="test",
        )
    mock_llm.assert_called_once()
    kwargs = mock_llm.call_args.kwargs
    assert kwargs.get("temperature") == 0.2


def test_rerank_top_k_honours_explicit_temperature():
    from llm_reranker import rerank_top_k

    with patch("llm_reranker.LLMClient.complete_text", return_value="cls_a") as mock_llm:
        rerank_top_k(
            texts=["обращение"],
            top_candidates=[["cls_a", "cls_b"]],
            argmax_labels=["cls_b"],
            provider="openai", model="gpt-4o-mini", api_key="test",
            temperature=0.0,
        )
    assert mock_llm.call_args.kwargs.get("temperature") == 0.0


def test_rerank_top_k_passes_none_temperature_unchanged():
    """temperature=None означает «использовать provider default» и должен
    пробрасываться без подмены на 0.2."""
    from llm_reranker import rerank_top_k

    with patch("llm_reranker.LLMClient.complete_text", return_value="cls_a") as mock_llm:
        rerank_top_k(
            texts=["обращение"],
            top_candidates=[["cls_a", "cls_b"]],
            argmax_labels=["cls_b"],
            provider="openai", model="gpt-4o-mini", api_key="test",
            temperature=None,
        )
    assert mock_llm.call_args.kwargs.get("temperature") is None


# ---------------------------------------------------------------------------
# W4.5 — Brier Score + per-class ECE
# ---------------------------------------------------------------------------

def test_brier_score_perfect_predictions_is_zero():
    """Идеальные one-hot предсказания → Brier = 0."""
    np = pytest.importorskip("numpy")
    from ml_training import _compute_brier_score

    classes = ["a", "b", "c"]
    proba = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    yva = ["a", "b", "c"]
    brier = _compute_brier_score(proba, yva, classes)
    assert brier == pytest.approx(0.0, abs=1e-9)


def test_brier_score_worst_case_is_two():
    """Полностью неверные one-hot предсказания → Brier = 2.0 (верхняя граница)."""
    np = pytest.importorskip("numpy")
    from ml_training import _compute_brier_score

    classes = ["a", "b"]
    proba = np.array([[1.0, 0.0], [1.0, 0.0]])
    yva = ["b", "b"]
    brier = _compute_brier_score(proba, yva, classes)
    # каждая строка вносит 1² + 1² = 2 → среднее 2.0
    assert brier == pytest.approx(2.0, abs=1e-9)


def test_brier_score_uniform_uncalibrated():
    """Равномерное распределение по k классам для истинного класса → 1 − 2/k + 1/k."""
    np = pytest.importorskip("numpy")
    from ml_training import _compute_brier_score

    classes = ["a", "b", "c", "d"]
    k = 4
    proba = np.full((10, k), 1.0 / k)
    yva = ["a"] * 10
    brier = _compute_brier_score(proba, yva, classes)
    # (1 − 1/k)² + (k−1) · (1/k)² = 1 − 2/k + 1/k
    expected = 1.0 - 2.0 / k + 1.0 / k
    assert brier == pytest.approx(expected, abs=1e-9)


def test_per_class_ece_perfect_calibration_is_zero():
    np = pytest.importorskip("numpy")
    from ml_training import _compute_per_class_ece

    classes = ["a", "b"]
    # В каждой строке правильный класс предсказан с confidence 1.0.
    # Для перфектной калибровки ECE = 0 по каждому классу.
    proba = np.array([[1.0, 0.0]] * 60 + [[0.0, 1.0]] * 60)
    yva = ["a"] * 60 + ["b"] * 60
    out = _compute_per_class_ece(proba, yva, classes)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"] == pytest.approx(0.0, abs=1e-9)
    assert out["b"] == pytest.approx(0.0, abs=1e-9)


def test_per_class_ece_returns_dict_keyed_by_class():
    np = pytest.importorskip("numpy")
    from ml_training import _compute_per_class_ece

    classes = ["x", "y", "z"]
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=50)
    yva = list(rng.choice(classes, size=50))
    out = _compute_per_class_ece(proba, yva, classes)
    assert set(out.keys()) == set(classes)
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_per_class_ece_empty_returns_empty_dict_on_failure():
    """Совсем пустой вход → пустой словарь (через except-ветку)."""
    np = pytest.importorskip("numpy")
    from ml_training import _compute_per_class_ece

    proba = np.zeros((0, 0))
    out = _compute_per_class_ece(proba, [], [])
    assert out == {}
