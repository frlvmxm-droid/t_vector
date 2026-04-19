# -*- coding: utf-8 -*-
"""E2E + unit tests for the train_model() decomposition helpers.

Covers stage helpers extracted from train_model() and the augmentation
flag combinations (use_fuzzy_dedup / use_smote / use_hard_negatives /
use_field_dropout / use_label_smoothing) which previously had no direct
test coverage.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_training import (
    _apply_label_smoothing,
    _augment_training_data,
    _estimate_model_size_bytes,
    _log_svd_explained_variance,
    _log_training_start,
    train_model,
)


def _simple_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=100)


def _fitted_pipeline() -> Pipeline:
    X = ["один два три"] * 15 + ["четыре пять шесть"] * 15
    y = ["A"] * 15 + ["B"] * 15
    pipe = Pipeline([("features", _simple_vectorizer()),
                     ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(X, y)
    return pipe


# ── _log_training_start ──────────────────────────────────────────────────────

def test_log_training_start_no_cb_is_noop():
    # Must not raise when log_cb is None
    _log_training_start(["a"] * 4, ["A", "A", "B", "B"],
                        clf_type="LogReg", C=1.0, max_iter=100,
                        balanced=False, log_cb=None)


def test_log_training_start_emits_basic_lines():
    logs: list[str] = []
    _log_training_start(
        ["x"] * 10, ["A"] * 5 + ["B"] * 5,
        clf_type="LogReg", C=0.5, max_iter=200, balanced=True,
        log_cb=logs.append,
    )
    assert any("Выборка: 10" in m for m in logs)
    assert any("классов: 2" in m for m in logs)
    assert any("class_weight=balanced" in m for m in logs)
    # No imbalance warning for perfectly balanced case
    assert not any("Дисбаланс" in m for m in logs)


def test_log_training_start_warns_on_imbalance():
    logs: list[str] = []
    # 30:1 imbalance — ratio > 10 triggers warning
    y = ["big"] * 30 + ["small"]
    _log_training_start(
        ["x"] * 31, y,
        clf_type="LogReg", C=1.0, max_iter=100, balanced=False,
        log_cb=logs.append,
    )
    assert any("Дисбаланс" in m and "big" in m and "small" in m for m in logs)


def test_log_training_start_no_warning_for_balanced():
    logs: list[str] = []
    _log_training_start(
        ["x"] * 20, ["A"] * 10 + ["B"] * 10,
        clf_type="LogReg", C=1.0, max_iter=100, balanced=False,
        log_cb=logs.append,
    )
    assert not any("Дисбаланс" in m for m in logs)


def test_log_training_start_single_class_no_warning():
    logs: list[str] = []
    _log_training_start(
        ["x"] * 5, ["only"] * 5,
        clf_type="LogReg", C=1.0, max_iter=100, balanced=False,
        log_cb=logs.append,
    )
    assert not any("Дисбаланс" in m for m in logs)


# ── _apply_label_smoothing ───────────────────────────────────────────────────

def test_apply_label_smoothing_returns_copy_when_eps_zero():
    y = ["A", "B", "A", "B"] * 5
    out = _apply_label_smoothing(y, eps=0.0, random_state=42, log_cb=None)
    assert out == y
    assert out is not y  # must be a new list


def test_apply_label_smoothing_flips_expected_count():
    y = ["A"] * 50 + ["B"] * 50
    logs: list[str] = []
    out = _apply_label_smoothing(y, eps=0.10, random_state=42, log_cb=logs.append)
    # eps=0.10 × 100 = 10 flips expected
    flips = sum(1 for a, b in zip(y, out) if a != b)
    assert flips == 10
    assert any("перемаркировано 10" in m for m in logs)


def test_apply_label_smoothing_single_class_returns_unchanged():
    y = ["only"] * 20
    out = _apply_label_smoothing(y, eps=0.5, random_state=0, log_cb=None)
    assert out == y


def test_apply_label_smoothing_deterministic_with_seed():
    y = ["A", "B", "C"] * 30
    a = _apply_label_smoothing(y, eps=0.15, random_state=7, log_cb=None)
    b = _apply_label_smoothing(y, eps=0.15, random_state=7, log_cb=None)
    assert a == b


def test_apply_label_smoothing_zero_flip_count():
    # eps so small that int(n * eps) == 0 → early return
    y = ["A", "B"] * 3  # len=6; 6 * 0.01 = 0.06 → int → 0
    out = _apply_label_smoothing(y, eps=0.01, random_state=0, log_cb=None)
    assert out == y


# ── _augment_training_data ───────────────────────────────────────────────────

def test_augment_all_flags_off_is_identity():
    X = ["a", "b", "c", "d"]
    y = ["A", "B", "A", "B"]
    Xo, yo = _augment_training_data(
        list(X), list(y), random_state=0,
        use_fuzzy_dedup=False, fuzzy_dedup_threshold=92,
        use_smote=False, oversample_strategy="cap", max_dup_per_sample=5,
        use_hard_negatives=False,
        use_field_dropout=False, field_dropout_prob=0.15, field_dropout_copies=2,
        use_label_smoothing=False, label_smoothing_eps=0.0,
        log_cb=None, progress_cb=None,
    )
    assert Xo == X
    assert yo == y


def test_augment_smote_grows_rare_class():
    # Class B has 2 samples vs A's 20 → SMOTE should oversample B
    X = ["текст пример " + str(i) for i in range(20)] + ["редкий", "редкий2"]
    y = ["A"] * 20 + ["B", "B"]
    Xo, yo = _augment_training_data(
        list(X), list(y), random_state=0,
        use_fuzzy_dedup=False, fuzzy_dedup_threshold=92,
        use_smote=True, oversample_strategy="cap", max_dup_per_sample=5,
        use_hard_negatives=False,
        use_field_dropout=False, field_dropout_prob=0.15, field_dropout_copies=2,
        use_label_smoothing=False, label_smoothing_eps=0.0,
        log_cb=None, progress_cb=None,
    )
    # SMOTE must increase total or at least B's count
    from collections import Counter
    assert Counter(yo)["B"] >= Counter(y)["B"]


def test_augment_label_smoothing_changes_labels():
    rng = np.random.default_rng(0)
    X = [f"текст {i}" for i in range(100)]
    y = list(rng.choice(["A", "B", "C"], size=100))
    Xo, yo = _augment_training_data(
        list(X), list(y), random_state=7,
        use_fuzzy_dedup=False, fuzzy_dedup_threshold=92,
        use_smote=False, oversample_strategy="cap", max_dup_per_sample=5,
        use_hard_negatives=False,
        use_field_dropout=False, field_dropout_prob=0.15, field_dropout_copies=2,
        use_label_smoothing=True, label_smoothing_eps=0.2,
        log_cb=None, progress_cb=None,
    )
    # X unchanged; y should differ in about 20% of positions
    assert Xo == X
    flips = sum(1 for a, b in zip(y, yo) if a != b)
    assert flips == 20


def test_augment_field_dropout_expands_tagged_texts():
    # Each example has multiple [TAG] sections — dropout should add copies
    X = [
        "[DESC]\nописание\n[CLIENT]\nклиент\n[OPERATOR]\nоператор",
    ] * 20
    y = ["A"] * 20
    Xo, yo = _augment_training_data(
        list(X), list(y), random_state=0,
        use_fuzzy_dedup=False, fuzzy_dedup_threshold=92,
        use_smote=False, oversample_strategy="cap", max_dup_per_sample=5,
        use_hard_negatives=False,
        use_field_dropout=True, field_dropout_prob=0.5, field_dropout_copies=3,
        use_label_smoothing=False, label_smoothing_eps=0.0,
        log_cb=None, progress_cb=None,
    )
    # field_dropout adds n_copies × len(X) copies (minus ones that drop everything)
    assert len(Xo) > len(X)
    assert len(Xo) == len(yo)


def test_augment_preserves_length_alignment():
    """After every combination, len(X) == len(y) invariant must hold."""
    X = ["короткий текст"] * 10 + ["длинный текст с большим числом слов"] * 10
    y = ["A"] * 10 + ["B"] * 10
    Xo, yo = _augment_training_data(
        list(X), list(y), random_state=0,
        use_fuzzy_dedup=True, fuzzy_dedup_threshold=90,
        use_smote=True, oversample_strategy="cap", max_dup_per_sample=3,
        use_hard_negatives=True,
        use_field_dropout=False, field_dropout_prob=0.15, field_dropout_copies=2,
        use_label_smoothing=True, label_smoothing_eps=0.05,
        log_cb=None, progress_cb=None,
    )
    assert len(Xo) == len(yo)


# ── _log_svd_explained_variance ──────────────────────────────────────────────

def test_svd_log_no_cb_is_noop():
    pipe = _fitted_pipeline()
    _log_svd_explained_variance(pipe, log_cb=None)


def test_svd_log_quiet_when_no_svd():
    """Pipeline without TruncatedSVD must not log anything."""
    pipe = _fitted_pipeline()
    logs: list[str] = []
    _log_svd_explained_variance(pipe, log_cb=logs.append)
    assert logs == []


def test_svd_log_finds_svd_in_nested_pipeline():
    """If features step is itself a Pipeline containing 'svd', log explained variance."""
    from sklearn.decomposition import TruncatedSVD
    X = ["один два три " * 10] * 30 + ["четыре пять шесть " * 10] * 30
    y = ["A"] * 30 + ["B"] * 30
    features = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=200)),
        ("svd", TruncatedSVD(n_components=5, random_state=0)),
    ])
    pipe = Pipeline([("features", features), ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(X, y)
    logs: list[str] = []
    _log_svd_explained_variance(pipe, log_cb=logs.append)
    assert any("[SVD] объяснённая дисперсия" in m and "5 компонент" in m for m in logs)


# ── _estimate_model_size_bytes ───────────────────────────────────────────────

def test_estimate_model_size_returns_positive_int():
    pipe = _fitted_pipeline()
    size = _estimate_model_size_bytes(pipe, log_cb=None)
    assert isinstance(size, int)
    assert size > 0


def test_estimate_model_size_logs_kb():
    pipe = _fitted_pipeline()
    logs: list[str] = []
    _estimate_model_size_bytes(pipe, log_cb=logs.append)
    assert any("Размер модели" in m and "КБ" in m for m in logs)


def test_estimate_model_size_analytical_on_empty_object():
    """The analytical estimator walks ML-specific attributes (vocabulary_,
    coef_, components_, …). An object with none of them yields just the
    fixed framing overhead — not None — because no pickling is attempted."""
    class _Empty:
        pass
    size = _estimate_model_size_bytes(_Empty(), log_cb=None)
    assert isinstance(size, int)
    assert size < 50_000  # only the fixed ~20 KB overhead


# ── train_model E2E: each augmentation flag independently ─────────────────────


@pytest.fixture
def _balanced_dataset():
    X = ["банк карта платеж " + str(i) for i in range(30)] + \
        ["кредит долг займ " + str(i) for i in range(30)] + \
        ["вклад депозит процент " + str(i) for i in range(30)]
    y = ["карты"] * 30 + ["кредиты"] * 30 + ["вклады"] * 30
    return X, y


def test_train_model_use_fuzzy_dedup(_balanced_dataset):
    X, y = _balanced_dataset
    # Duplicate a few rows — fuzzy_dedup should drop them
    X2 = X + X[:5]
    y2 = y + y[:5]
    pipe, _, _, _, _, extras = train_model(
        X=X2, y=y2, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        use_smote=False, use_fuzzy_dedup=True, fuzzy_dedup_threshold=90,
    )
    assert pipe is not None
    assert isinstance(extras, dict)


def test_train_model_use_smote(_balanced_dataset):
    X, y = _balanced_dataset
    # Introduce a rare class with only 3 samples
    X_rare = X + ["редкий " + str(i) for i in range(3)]
    y_rare = y + ["редкий_класс"] * 3
    pipe, _, _, _, _, extras = train_model(
        X=X_rare, y=y_rare, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        use_smote=True, oversample_strategy="cap", max_dup_per_sample=5,
    )
    assert pipe is not None


def test_train_model_use_hard_negatives(_balanced_dataset):
    X, y = _balanced_dataset
    logs: list[str] = []
    pipe, _, _, _, _, _ = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        use_smote=False, use_hard_negatives=True,
        log_cb=logs.append,
    )
    assert pipe is not None


def test_train_model_use_field_dropout():
    # Build tagged-format inputs — field dropout operates on [TAG] sections
    sample = "[DESC]\nописание транзакции\n[CLIENT]\nклиент говорит\n[OPERATOR]\nоператор отвечает"
    X = [sample] * 30 + [sample.replace("транзакции", "кредита")] * 30
    y = ["A"] * 30 + ["B"] * 30
    pipe, _, _, _, _, _ = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        use_smote=False,
        use_field_dropout=True, field_dropout_prob=0.3, field_dropout_copies=2,
    )
    assert pipe is not None


def test_train_model_use_label_smoothing(_balanced_dataset):
    X, y = _balanced_dataset
    pipe, _, _, _, _, _ = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        use_smote=False,
        use_label_smoothing=True, label_smoothing_eps=0.1,
    )
    assert pipe is not None


def test_train_model_all_augmentations_combined(_balanced_dataset):
    X, y = _balanced_dataset
    pipe, _, _, _, _, extras = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        use_smote=True,
        use_hard_negatives=True,
        use_fuzzy_dedup=True, fuzzy_dedup_threshold=95,
        use_label_smoothing=True, label_smoothing_eps=0.05,
    )
    assert pipe is not None
    assert "n_train" in extras
    assert "n_test" in extras
    assert "training_duration_sec" in extras


def test_train_model_extras_contains_model_size(_balanced_dataset):
    """Ensure the decomposed size-estimator writes model_size_bytes into extras."""
    X, y = _balanced_dataset
    _, _, _, _, _, extras = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        use_smote=False,
    )
    assert "model_size_bytes" in extras
    assert extras["model_size_bytes"] > 0
