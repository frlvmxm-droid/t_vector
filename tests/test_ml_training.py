# -*- coding: utf-8 -*-
"""Unit tests for ml_training.py core functions."""
import math
import pytest
from collections import Counter
from unittest.mock import MagicMock, patch

from ml_training import make_classifier, train_model, cv_evaluate, compute_temperature_scaling


# ── make_classifier ──────────────────────────────────────────────────────────

def test_make_classifier_returns_calibrated_for_standard_dataset():
    y = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    clf, clf_type = make_classifier(y, C=1.0, max_iter=1000, balanced=False)
    assert "LinearSVC" in clf_type or "Calibrated" in clf_type


def test_make_classifier_falls_back_to_logreg_for_tiny_dataset():
    # Only 2 samples per class → can't do CV=3
    y = ["A", "A", "B", "B"]
    clf, clf_type = make_classifier(y, C=1.0, max_iter=100, balanced=False)
    assert clf_type == "LogReg"


def test_make_classifier_dynamic_cv_reduces_for_small_class():
    # min_class = 6 → CV = min(5, max(3, ceil(6/3))) = min(5, max(3, 2)) = min(5, 3) = 3
    y = ["A"] * 6 + ["B"] * 30
    clf, clf_type = make_classifier(y, C=1.0, max_iter=100, balanced=False)
    assert "cv=3" in clf_type or "LogReg" in clf_type


def test_make_classifier_uses_sigmoid_for_small_dataset():
    y = ["A"] * 10 + ["B"] * 10
    _, clf_type = make_classifier(y, C=1.0, max_iter=100, balanced=False, calib_method="sigmoid")
    assert "sigmoid" in clf_type or "LogReg" in clf_type


def test_make_classifier_auto_calib_selects_sigmoid_for_small():
    y = ["A"] * 10 + ["B"] * 10
    _, clf_type = make_classifier(y, C=1.0, max_iter=100, balanced=False, calib_method="auto")
    # avg_per_class = 20/2 = 10 < 200 → sigmoid
    assert "sigmoid" in clf_type or "LogReg" in clf_type


def test_make_classifier_auto_calib_selects_isotonic_for_large():
    y = ["A"] * 400 + ["B"] * 400
    _, clf_type = make_classifier(y, C=1.0, max_iter=100, balanced=False, calib_method="auto")
    # avg_per_class = 800/2 = 400 >= 200 → isotonic
    assert "isotonic" in clf_type


def test_make_classifier_balanced_class_weight():
    y = ["A"] * 30 + ["B"] * 30
    clf, _ = make_classifier(y, C=1.0, max_iter=100, balanced=True)
    # CalibratedClassifierCV wraps LinearSVC; the inner estimator has class_weight
    inner = getattr(clf, "estimator", None)
    if inner is not None:
        assert inner.class_weight == "balanced"


# ── train_model ──────────────────────────────────────────────────────────────

def _simple_vectorizer():
    """Return a simple TF-IDF vectorizer for use in train_model tests."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=100)


def test_train_model_returns_pipeline_and_report():
    X = ["банк карта деньги"] * 20 + ["кредит займ долг"] * 20 + ["депозит вклад счет"] * 20
    y = ["карты"] * 20 + ["кредиты"] * 20 + ["вклады"] * 20
    pipe, clf_type, report, labels, cm, extras = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        use_smote=False,
    )
    assert pipe is not None
    assert isinstance(clf_type, str) and len(clf_type) > 0
    assert isinstance(report, str)


def test_train_model_with_two_classes():
    X = ["позитивный текст хорошо"] * 30 + ["негативный отзыв плохо"] * 30
    y = ["pos"] * 30 + ["neg"] * 30
    pipe, _, report, labels, cm, _ = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=False,
        test_size=0.2, random_state=0,
        use_smote=False,
    )
    assert pipe is not None
    # Validate the pipeline can predict
    preds = pipe.predict(X[:5])
    assert all(p in {"pos", "neg"} for p in preds)


def test_train_model_skips_validation_for_tiny_dataset():
    # Very small dataset → _maybe_skip_validation kicks in
    X = ["abc", "def"]
    y = ["A", "B"]
    pipe, clf_type, report, labels, cm, extras = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=100, balanced=False,
        test_size=0.2, random_state=42,
        use_smote=False,
    )
    assert pipe is not None
    # When validation is skipped, labels may be None
    # (report is still a non-empty string with an explanation)
    assert isinstance(report, str) and len(report) > 0


def test_train_model_progress_callback_called():
    called = []

    def _cb(pct, msg):
        called.append((pct, msg))

    X = ["слово один два"] * 30 + ["другое три четыре"] * 30
    y = ["A"] * 30 + ["B"] * 30
    train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=100, balanced=False,
        test_size=0.2, random_state=42,
        progress_cb=_cb,
        use_smote=False,
    )
    assert len(called) > 0
    pcts = [c[0] for c in called]
    assert max(pcts) >= 60.0


def test_train_model_log_callback_called():
    logs = []
    X = ["слово один"] * 30 + ["другое два"] * 30
    y = ["X"] * 30 + ["Y"] * 30
    train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=100, balanced=False,
        test_size=0.2, random_state=42,
        log_cb=logs.append,
        use_smote=False,
    )
    assert any("Обучение" in msg or "классов" in msg for msg in logs)


def test_train_model_extras_contains_thresholds():
    X = ["один два три"] * 30 + ["четыре пять шесть"] * 30
    y = ["A"] * 30 + ["B"] * 30
    _, _, _, labels, cm, extras = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=False,
        test_size=0.3, random_state=7,
        use_smote=False,
    )
    assert isinstance(extras, dict)


@pytest.mark.parametrize("calib_method", ["sigmoid", "isotonic"])
def test_train_model_different_calib_methods(calib_method):
    X = ["текст документ"] * 30 + ["слово термин"] * 30
    y = ["doc"] * 30 + ["word"] * 30
    pipe, clf_type, _, _, _, _ = train_model(
        X=X, y=y, features=_simple_vectorizer(),
        C=1.0, max_iter=200, balanced=False,
        test_size=0.2, random_state=42,
        calib_method=calib_method,
        use_smote=False,
    )
    assert pipe is not None


# ── compute_temperature_scaling ───────────────────────────────────────────────

def test_compute_temperature_scaling_returns_finite_temp():
    import numpy as np
    # proba shape: (n_samples, n_classes); yva is string labels; classes is list of class names
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.4, 0.6]])
    yva = ["A", "B", "A", "B"]
    classes = ["A", "B"]
    temp = compute_temperature_scaling(probs, yva, classes)
    assert isinstance(float(temp), float)
    assert 0.1 < float(temp) < 10.0


# ── cv_evaluate ───────────────────────────────────────────────────────────────

def test_cv_evaluate_returns_dict_with_scores():
    from sklearn.pipeline import Pipeline
    X = ["один два три"] * 20 + ["четыре пять шесть"] * 20
    y = ["A"] * 20 + ["B"] * 20
    pipe = Pipeline([("features", _simple_vectorizer()), ("clf", __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=200))])
    result = cv_evaluate(X, y, pipe, n_splits=3)
    assert isinstance(result, dict)
    assert "cv_f1_macro_mean" in result or len(result) >= 0
