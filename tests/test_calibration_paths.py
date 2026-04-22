"""Покрытие auto-выбора калибровки и ECE/MCE/Brier-фолбэков.

Wave 6.5 Block 1. Дополняет существующие train_model-тесты ветками
``calib_method='auto'`` (sigmoid vs isotonic в зависимости от
avg-per-class), маленькими датасетами (< CV-фолды) и corrupted-input
для вычисления ECE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_training import (  # noqa: E402
    _compute_brier_score,
    _compute_ece_mce,
    _compute_per_class_ece,
    make_classifier,
)


# ---------------------------------------------------------------------------
# make_classifier — auto-selection и fallback
# ---------------------------------------------------------------------------


def _make_y(n_per_class: int, n_classes: int = 3) -> list[str]:
    return [f"c{i % n_classes}" for i in range(n_per_class * n_classes)]


def test_make_classifier_auto_picks_sigmoid_for_small_data() -> None:
    """avg_per_class < 200 → 'sigmoid' (надёжнее на малых данных)."""
    y = _make_y(n_per_class=50, n_classes=3)  # 150 samples / 3 = 50 < 200
    _est, name = make_classifier(y, C=1.0, max_iter=2000, balanced=False,
                                 calib_method="auto")
    assert "sigmoid" in name, f"expected sigmoid, got {name}"


def test_make_classifier_auto_picks_isotonic_for_large_data() -> None:
    """avg_per_class >= 200 → 'isotonic' (точнее на больших данных)."""
    y = _make_y(n_per_class=250, n_classes=2)  # 500 samples / 2 = 250 >= 200
    _est, name = make_classifier(y, C=1.0, max_iter=2000, balanced=False,
                                 calib_method="auto")
    assert "isotonic" in name, f"expected isotonic, got {name}"


def test_make_classifier_explicit_sigmoid_passes_through() -> None:
    y = _make_y(n_per_class=20, n_classes=2)
    _est, name = make_classifier(y, C=1.0, max_iter=2000, balanced=False,
                                 calib_method="sigmoid")
    assert "sigmoid" in name


def test_make_classifier_invalid_method_falls_back_to_sigmoid() -> None:
    """Unknown calib_method → 'sigmoid' (defensive, не поднимает)."""
    y = _make_y(n_per_class=20, n_classes=2)
    _est, name = make_classifier(y, C=1.0, max_iter=2000, balanced=False,
                                 calib_method="unknown_method")
    assert "sigmoid" in name


def test_make_classifier_balanced_class_weight() -> None:
    y = _make_y(n_per_class=20, n_classes=2)
    est, _name = make_classifier(y, C=1.0, max_iter=2000, balanced=True,
                                 calib_method="sigmoid")
    # Выставлен class_weight на base estimator-е CalibratedClassifierCV.
    base = getattr(est, "estimator", None) or getattr(est, "base_estimator", None)
    assert getattr(base, "class_weight", None) == "balanced"


def test_make_classifier_too_small_falls_to_logreg() -> None:
    """min_class < CV → LogisticRegression fallback."""
    y = ["a", "a", "b"]  # min_class = 1, ниже любого CV
    _est, name = make_classifier(y, C=1.0, max_iter=2000, balanced=False,
                                 calib_method="auto")
    assert name == "LogReg"


def test_make_classifier_single_class_falls_to_logreg() -> None:
    """len(set(y)) < 2 → LogisticRegression fallback (нет калибровки)."""
    y = ["a"] * 50
    _est, name = make_classifier(y, C=1.0, max_iter=2000, balanced=False,
                                 calib_method="sigmoid")
    assert name == "LogReg"


# ---------------------------------------------------------------------------
# _compute_ece_mce / _compute_brier_score / _compute_per_class_ece — fallbacks
# ---------------------------------------------------------------------------


def test_compute_ece_mce_basic_perfectly_calibrated() -> None:
    """Идеально калиброванный бинарный класс — ECE и MCE близки к 0."""
    rng = np.random.default_rng(42)
    proba = rng.dirichlet([1.0, 1.0], size=100)
    y_idx = proba.argmax(axis=1)
    classes = ["a", "b"]
    yva = [classes[int(i)] for i in y_idx]
    ece, mce = _compute_ece_mce(proba, yva, classes, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0


def test_compute_ece_mce_handles_unknown_label_via_fallback_index() -> None:
    """Метка не из classes → mapped to 0 (не raise, не NaN)."""
    proba = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
    yva = ["a", "b", "unknown"]  # "unknown" → fallback to index 0
    classes = ["a", "b"]
    ece, mce = _compute_ece_mce(proba, yva, classes, n_bins=3)
    assert isinstance(ece, float)
    assert isinstance(mce, float)


def test_compute_ece_mce_corrupted_proba_returns_zero_zero() -> None:
    """Сломанный proba (1D) → graceful (0.0, 0.0), не исключение."""
    bad_proba = np.array([0.5, 0.5])  # 1D — должно упасть в except
    ece, mce = _compute_ece_mce(bad_proba, ["a", "b"], ["a", "b"])
    assert ece == 0.0
    assert mce == 0.0


def test_compute_brier_score_perfect_one_hot_is_zero() -> None:
    """Брайер идеального one-hot предикта = 0.0."""
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    yva = ["a", "b", "a"]
    classes = ["a", "b"]
    score = _compute_brier_score(proba, yva, classes)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_compute_brier_score_uniform_is_nontrivial() -> None:
    """Равномерный uniform predict → положительный Brier."""
    proba = np.full((10, 3), 1.0 / 3.0)
    yva = ["a"] * 10
    score = _compute_brier_score(proba, yva, ["a", "b", "c"])
    assert score > 0.0


def test_compute_brier_score_corrupted_input_returns_zero() -> None:
    """Сломанный proba (1D) → 0.0 (graceful)."""
    score = _compute_brier_score(np.array([0.5]), ["a"], ["a", "b"])
    assert score == 0.0


def test_compute_per_class_ece_returns_dict_of_floats() -> None:
    rng = np.random.default_rng(7)
    proba = rng.dirichlet([1.0, 1.0, 1.0], size=60)
    y_idx = proba.argmax(axis=1)
    classes = ["a", "b", "c"]
    yva = [classes[int(i)] for i in y_idx]
    out = _compute_per_class_ece(proba, yva, classes)
    assert isinstance(out, dict)
    for cls in classes:
        assert cls in out
        assert isinstance(out[cls], float)
        assert 0.0 <= out[cls] <= 1.0
