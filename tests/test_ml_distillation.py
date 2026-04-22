# -*- coding: utf-8 -*-
"""
Unit tests for ml_distillation.

Covers soft-label distillation happy path and evaluate_distillation.
Uses a trivial sklearn teacher+student (LogisticRegression on tiny dataset)
to avoid heavy SBERT/torch dependencies.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_distillation import distill_soft_labels, evaluate_distillation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset():
    X = (
        ["оплата картой", "перевод денег", "оплата кредита"] * 4
        + ["блокировка карты", "потерял карту", "заблокировали карту"] * 4
        + ["кредит одобрен", "хочу кредит", "кредит наличными"] * 4
    )
    y = (
        ["оплата"] * 12
        + ["блокировка"] * 12
        + ["кредит"] * 12
    )
    return X, y


def _make_teacher(X, y):
    pipe = Pipeline([
        ("features", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(X, y)
    return pipe


def _make_student():
    return Pipeline([
        ("features", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=500)),
    ])


# ---------------------------------------------------------------------------
# distill_soft_labels
# ---------------------------------------------------------------------------

class TestDistillSoftLabels:
    def test_happy_path(self, tiny_dataset):
        X, y = tiny_dataset
        teacher = _make_teacher(X, y)
        student = _make_student()
        trained = distill_soft_labels(
            teacher, student, X, y, temperature=2.0, alpha=0.5
        )
        preds = trained.predict(X)
        assert len(preds) == len(X)

    def test_rejects_teacher_without_proba(self, tiny_dataset):
        X, y = tiny_dataset

        class NoProbaModel:
            classes_ = ["a"]

            def predict(self, X):
                return ["a"] * len(X)

        with pytest.raises(ValueError, match="predict_proba"):
            distill_soft_labels(NoProbaModel(), _make_student(), X, y)

    def test_alpha_zero_equivalent_to_hard(self, tiny_dataset):
        """alpha=0 → soft labels = one-hot true → training reduces to normal fit."""
        X, y = tiny_dataset
        teacher = _make_teacher(X, y)
        student = _make_student()
        trained = distill_soft_labels(
            teacher, student, X, y, temperature=1.0, alpha=0.0
        )
        # Training should succeed and predictions should be reasonable
        preds = trained.predict(X)
        # majority class present
        assert set(preds).issubset(set(y))

    def test_log_cb_invoked(self, tiny_dataset):
        X, y = tiny_dataset
        teacher = _make_teacher(X, y)
        student = _make_student()
        logs = []
        distill_soft_labels(
            teacher, student, X, y, temperature=2.0, log_cb=logs.append
        )
        assert any("Дистилляция" in msg for msg in logs)


# ---------------------------------------------------------------------------
# evaluate_distillation
# ---------------------------------------------------------------------------

class TestEvaluateDistillation:
    def test_returns_expected_keys(self, tiny_dataset):
        X, y = tiny_dataset
        teacher = _make_teacher(X, y)
        student = _make_teacher(X, y)  # reuse: both pre-trained
        result = evaluate_distillation(teacher, student, X, y)
        assert result.keys() >= {
            "teacher_f1", "student_f1", "f1_drop",
            "teacher_acc", "student_acc", "acc_drop",
        }

    def test_identical_models_zero_drop(self, tiny_dataset):
        X, y = tiny_dataset
        model = _make_teacher(X, y)
        result = evaluate_distillation(model, model, X, y)
        assert result["f1_drop"] == 0.0
        assert result["acc_drop"] == 0.0
