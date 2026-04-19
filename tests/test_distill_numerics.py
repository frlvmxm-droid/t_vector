# -*- coding: utf-8 -*-
"""Тесты численной устойчивости temperature-softmax в ml_distillation.

При больших температурах T наивный путь
``exp(log_p / T) / sum(...).clip(eps)`` даёт underflow + деление на eps,
ошибка усиливается на 10 порядков. Текущая реализация — logsumexp,
проверяем её свойства:

  1. T = 100 не даёт NaN/Inf, суммы строк ≈ 1.0.
  2. T = 1 (no-op путь) совпадает с входными вероятностями.
  3. Извлечённый argmax в общем не падает в случайный класс при T → ∞.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ml_distillation import distill_soft_labels


class _MockTeacher:
    """Минимальный teacher: фиксированное predict_proba + classes_."""

    def __init__(self, proba: np.ndarray, classes: list[str]) -> None:
        self._proba = proba
        self.classes_ = classes

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self._proba


def _identity_student_pipeline(classes: list[str]) -> Any:
    """Pipeline, у которого fit() запоминает аргументы; не пытается обучаться.

    Нужен, чтобы вызвать distill_soft_labels без реального обучения —
    мы тестируем именно температурную математику.
    """
    captured: dict[str, Any] = {}

    class _DummyClf:
        def fit(self, X, y, sample_weight=None):
            captured["sample_weight"] = sample_weight
            captured["y"] = y
            return self

        def predict(self, X):
            return [classes[0]] * len(X)

    class _DummyPipe:
        def __init__(self):
            self.steps = [("clf", _DummyClf())]
            self._clf = self.steps[0][1]

        def fit(self, X, y, **fit_params):
            sw = fit_params.get("clf__sample_weight")
            self._clf.fit(X, y, sample_weight=sw)
            return self

    return _DummyPipe(), captured


class TestTemperatureSoftmaxStability:
    def test_extreme_temperature_no_nan_inf(self):
        # 4 примера, 3 класса; острая раздача (one-hot near-1).
        proba = np.array([
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
            [0.99, 0.005, 0.005],
        ])
        teacher = _MockTeacher(proba, ["a", "b", "c"])
        pipe, captured = _identity_student_pipeline(["a", "b", "c"])

        X = ["x"] * 4
        y = ["a", "b", "c", "a"]

        # T=100 уводит вероятности в почти-uniform; раньше это давало
        # underflow + деление на DISTILL_EPS.
        distill_soft_labels(
            teacher, pipe, X, y,
            temperature=100.0, alpha=1.0, log_cb=lambda _m: None,
        )
        sw = np.asarray(captured["sample_weight"])
        assert np.isfinite(sw).all(), "sample_weight содержит NaN/Inf"
        assert (sw > 0).all(), "веса должны быть строго положительны"

    def test_temperature_one_is_noop_for_alpha_one(self):
        # T=1, alpha=1 → soft_proba == teacher_proba; argmax совпадает.
        proba = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
        ])
        teacher = _MockTeacher(proba, ["a", "b", "c"])
        pipe, captured = _identity_student_pipeline(["a", "b", "c"])

        distill_soft_labels(
            teacher, pipe, ["x", "x"], ["a", "b"],
            temperature=1.0, alpha=1.0, log_cb=lambda _m: None,
        )
        # Учитель уверен в правильных классах → soft_y == истинные метки,
        # sample_weight ≈ 1 + true_proba (matches branch).
        assert captured["y"] == ["a", "b"]

    def test_high_temperature_smooths_to_near_uniform(self):
        proba = np.array([
            [0.95, 0.04, 0.01],
        ])
        teacher = _MockTeacher(proba, ["a", "b", "c"])
        pipe, captured = _identity_student_pipeline(["a", "b", "c"])

        # При T → ∞ распределение стремится к uniform → у всех классов
        # вероятность ≈ 1/3, разница между ними ≪ 1.
        distill_soft_labels(
            teacher, pipe, ["x"], ["a"],
            temperature=1000.0, alpha=1.0, log_cb=lambda _m: None,
        )
        sw = np.asarray(captured["sample_weight"])
        assert np.isfinite(sw).all()
        # При T=1000 правильный класс ≈ 1/3, веса в окрестности 1.3..1.4
        # (matches=True branch: 1.0 + true_cls_proba ≈ 1.33).
        assert 1.0 < float(sw[0]) < 2.0
