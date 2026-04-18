# -*- coding: utf-8 -*-
"""
Tests for three ML-support modules:
  - cluster_elbow.ClusterElbowSelector
  - ml_mlm_pretrain.is_available / estimate_mlm_time_minutes / pretrain_mlm
  - ml_distillation.distill_soft_labels / evaluate_distillation
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from cluster_elbow import ClusterElbowSelector
from ml_distillation import distill_soft_labels, evaluate_distillation
from ml_mlm_pretrain import estimate_mlm_time_minutes, is_available, pretrain_mlm


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_X = [
    "клиент хочет закрыть счёт",
    "вопрос по кредиту",
    "проблема с переводом",
    "блокировка карты",
    "снятие наличных",
    "пополнение вклада",
]
_y = ["закрытие", "кредит", "перевод", "карта", "наличные", "вклад"]


def _make_teacher():
    """Fitted DummyClassifier that exposes predict_proba."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(_X, _y)
    return clf


def _make_student_pipe():
    """Unfitted student pipeline: TF-IDF + DummyClassifier."""
    return Pipeline([("feat", TfidfVectorizer()), ("clf", DummyClassifier())])


# ===========================================================================
# MODULE 1: ClusterElbowSelector
# ===========================================================================


class TestClusterElbowSelectorEdgeCases:
    """Boundary conditions that do not touch kneed."""

    def test_single_k_returns_it(self):
        assert ClusterElbowSelector.pick_elbow_k([100.0], [5]) == 5

    def test_two_ks_returns_first(self):
        result = ClusterElbowSelector.pick_elbow_k([100.0, 60.0], [3, 4])
        assert result == 3

    def test_three_ks_minimum_needed_for_heuristic(self):
        result = ClusterElbowSelector.pick_elbow_k([100.0, 60.0, 50.0], [2, 3, 4])
        assert result in {2, 3, 4}

    def test_result_always_from_input_ks(self):
        ks = [2, 3, 4, 5, 6]
        inertias = [200.0, 150.0, 100.0, 98.0, 97.0]
        result = ClusterElbowSelector.pick_elbow_k(inertias, ks)
        assert result in ks

    def test_return_type_is_int(self):
        result = ClusterElbowSelector.pick_elbow_k([10.0, 8.0, 7.5, 7.4], [2, 3, 4, 5])
        assert isinstance(result, int)

    def test_flat_curve_returns_valid_k(self):
        ks = [2, 3, 4, 5]
        inertias = [5.0, 5.0, 5.0, 5.0]
        result = ClusterElbowSelector.pick_elbow_k(inertias, ks)
        assert result in ks

    def test_steep_drop_picks_early_k(self):
        # Big drop at index 1 (k=3), then flat — second derivative highest there
        ks = [2, 3, 4, 5, 6]
        inertias = [1000.0, 100.0, 95.0, 94.0, 93.0]
        result = ClusterElbowSelector.pick_elbow_k(inertias, ks)
        # The elbow should be around k=3 (index 1, interior point index 0 of sec list)
        assert result in ks


class TestClusterElbowSelectorWithKneed:
    """Patch kneed to control what KneeLocator returns."""

    def test_kneed_knee_is_used_when_available(self):
        """When KneeLocator returns a valid knee, pick_elbow_k must return it."""
        fake_kn = MagicMock()
        fake_kn.knee = 4

        fake_kneed_mod = types.ModuleType("kneed")
        fake_kneed_mod.KneeLocator = MagicMock(return_value=fake_kn)

        with patch.dict(sys.modules, {"kneed": fake_kneed_mod}):
            result = ClusterElbowSelector.pick_elbow_k(
                [100.0, 60.0, 30.0, 28.0, 27.0], [2, 3, 4, 5, 6]
            )
        assert result == 4

    def test_kneed_returns_none_falls_back_to_heuristic(self):
        """When KneeLocator.knee is None the fallback heuristic kicks in."""
        fake_kn = MagicMock()
        fake_kn.knee = None

        fake_kneed_mod = types.ModuleType("kneed")
        fake_kneed_mod.KneeLocator = MagicMock(return_value=fake_kn)

        ks = [2, 3, 4, 5]
        inertias = [200.0, 100.0, 95.0, 94.0]
        with patch.dict(sys.modules, {"kneed": fake_kneed_mod}):
            result = ClusterElbowSelector.pick_elbow_k(inertias, ks)
        assert result in ks

    def test_kneed_import_error_falls_back_to_heuristic(self):
        """ImportError from kneed must be silently caught; heuristic is used instead."""
        with patch.dict(sys.modules, {"kneed": None}):
            ks = [2, 3, 4, 5]
            inertias = [500.0, 200.0, 180.0, 179.0]
            result = ClusterElbowSelector.pick_elbow_k(inertias, ks)
        assert result in ks

    def test_kneed_generic_exception_falls_back_to_heuristic(self):
        """Any non-ImportError from KneeLocator must also be caught gracefully."""
        bad_kl = MagicMock(side_effect=RuntimeError("unexpected"))

        fake_kneed_mod = types.ModuleType("kneed")
        fake_kneed_mod.KneeLocator = bad_kl

        ks = [2, 3, 4, 5, 6]
        inertias = [300.0, 150.0, 100.0, 98.0, 97.0]
        with patch.dict(sys.modules, {"kneed": fake_kneed_mod}):
            result = ClusterElbowSelector.pick_elbow_k(inertias, ks)
        assert result in ks


# ===========================================================================
# MODULE 2: ml_mlm_pretrain
# ===========================================================================


class TestMlmIsAvailable:

    def test_returns_bool(self):
        assert isinstance(is_available(), bool)

    def test_false_when_transformers_missing(self):
        with patch("importlib.util.find_spec", return_value=None):
            assert is_available() is False

    def test_true_when_both_specs_found(self):
        fake_spec = MagicMock()
        with patch("importlib.util.find_spec", return_value=fake_spec):
            assert is_available() is True

    def test_false_when_transformers_absent_but_datasets_present(self):
        def _selective_spec(name):
            return None if name == "transformers" else MagicMock()

        with patch("importlib.util.find_spec", side_effect=_selective_spec):
            assert is_available() is False

    def test_false_when_datasets_absent_but_transformers_present(self):
        def _selective_spec(name):
            return None if name == "datasets" else MagicMock()

        with patch("importlib.util.find_spec", side_effect=_selective_spec):
            assert is_available() is False


class TestEstimateMlmTimeMinutes:

    def test_returns_tuple_of_two(self):
        result = estimate_mlm_time_minutes(1000)
        assert isinstance(result, tuple) and len(result) == 2

    def test_gpu_faster_than_cpu(self):
        low_gpu, high_gpu = estimate_mlm_time_minutes(5000, epochs=3, has_gpu=True)
        low_cpu, high_cpu = estimate_mlm_time_minutes(5000, epochs=3, has_gpu=False)
        assert high_gpu < low_cpu

    def test_low_less_than_high(self):
        low, high = estimate_mlm_time_minutes(2000, epochs=3, has_gpu=True)
        assert low < high

    def test_gpu_formula_correct(self):
        # secs_per_text_epoch=0.1, total=1000*3*0.1=300s, low=300*0.7/60=3.5, high=300*1.5/60=7.5
        low, high = estimate_mlm_time_minutes(1000, epochs=3, has_gpu=True)
        assert low == pytest.approx(3.5, abs=0.05)
        assert high == pytest.approx(7.5, abs=0.05)

    def test_cpu_formula_correct(self):
        # secs_per_text_epoch=1.5, total=100*1*1.5=150s, low=150*0.7/60=1.75→1.8, high=150*1.5/60=3.75→3.8
        low, high = estimate_mlm_time_minutes(100, epochs=1, has_gpu=False)
        assert low == pytest.approx(round(150 * 0.7 / 60, 1), abs=0.05)
        assert high == pytest.approx(round(150 * 1.5 / 60, 1), abs=0.05)

    def test_more_epochs_more_time(self):
        low1, _ = estimate_mlm_time_minutes(1000, epochs=1, has_gpu=True)
        low3, _ = estimate_mlm_time_minutes(1000, epochs=3, has_gpu=True)
        assert low3 > low1

    def test_more_texts_more_time(self):
        low_small, _ = estimate_mlm_time_minutes(100, epochs=3, has_gpu=True)
        low_large, _ = estimate_mlm_time_minutes(10000, epochs=3, has_gpu=True)
        assert low_large > low_small

    def test_result_rounded_to_one_decimal(self):
        low, high = estimate_mlm_time_minutes(333, epochs=2, has_gpu=True)
        assert low == round(low, 1)
        assert high == round(high, 1)


class TestPrtrainMlmRaisesWhenUnavailable:

    def test_raises_import_error_when_not_available(self):
        with patch("ml_mlm_pretrain.is_available", return_value=False):
            with pytest.raises(ImportError, match="transformers"):
                pretrain_mlm(["text1", "text2"], model_name="fake-model")


# ===========================================================================
# MODULE 3: ml_distillation
# ===========================================================================


class TestDistillSoftLabelsValidation:

    def test_raises_value_error_when_no_predict_proba(self):
        """Teacher without predict_proba must raise ValueError."""

        class NoProbaTeacher:
            def predict(self, X):
                return ["a"] * len(X)

        with pytest.raises(ValueError, match="predict_proba"):
            distill_soft_labels(
                NoProbaTeacher(), _make_student_pipe(), _X, _y
            )

    def test_accepts_teacher_with_predict_proba(self):
        teacher = _make_teacher()
        student = _make_student_pipe()
        result = distill_soft_labels(teacher, student, _X, _y)
        assert result is student

    def test_returns_fitted_pipeline(self):
        teacher = _make_teacher()
        student = _make_student_pipe()
        fitted = distill_soft_labels(teacher, student, _X, _y)
        # fitted pipeline must be able to predict
        preds = fitted.predict(_X)
        assert len(preds) == len(_X)


class TestDistillSoftLabelsAlpha:

    def test_alpha_zero_pure_hard_labels(self):
        """alpha=0 → labels == hard; student still trains."""
        teacher = _make_teacher()
        student = _make_student_pipe()
        fitted = distill_soft_labels(teacher, student, _X, _y, alpha=0.0)
        assert fitted.predict(_X) is not None

    def test_alpha_one_pure_teacher(self):
        """alpha=1 → pure teacher soft labels; must not error."""
        teacher = _make_teacher()
        student = _make_student_pipe()
        fitted = distill_soft_labels(teacher, student, _X, _y, alpha=1.0)
        assert fitted.predict(_X) is not None

    def test_default_alpha_midpoint(self):
        teacher = _make_teacher()
        student = _make_student_pipe()
        fitted = distill_soft_labels(teacher, student, _X, _y)  # alpha=0.5
        preds = fitted.predict(_X)
        assert len(preds) == len(_X)


class TestDistillSoftLabelsTemperature:

    def test_temperature_1_no_smoothing(self):
        teacher = _make_teacher()
        student = _make_student_pipe()
        fitted = distill_soft_labels(teacher, student, _X, _y, temperature=1.0)
        assert fitted.predict(_X) is not None

    def test_high_temperature_works(self):
        teacher = _make_teacher()
        student = _make_student_pipe()
        fitted = distill_soft_labels(teacher, student, _X, _y, temperature=10.0)
        assert fitted.predict(_X) is not None


class TestDistillSoftLabelsLogCallback:

    def test_log_cb_is_called(self):
        messages = []
        teacher = _make_teacher()
        student = _make_student_pipe()
        distill_soft_labels(teacher, student, _X, _y, log_cb=messages.append)
        assert len(messages) >= 1
        assert all(isinstance(m, str) for m in messages)


class TestEvaluateDistillation:

    def _fitted_pair(self):
        teacher = _make_teacher()
        student = _make_student_pipe()
        student.fit(_X, _y)
        return teacher, student

    def test_returns_dict_with_required_keys(self):
        teacher, student = self._fitted_pair()
        result = evaluate_distillation(teacher, student, _X, _y)
        required = {"teacher_f1", "student_f1", "f1_drop", "teacher_acc", "student_acc", "acc_drop"}
        assert required.issubset(result.keys())

    def test_f1_drop_equals_teacher_minus_student(self):
        teacher, student = self._fitted_pair()
        result = evaluate_distillation(teacher, student, _X, _y)
        expected = round(result["teacher_f1"] - result["student_f1"], 4)
        assert result["f1_drop"] == pytest.approx(expected, abs=1e-6)

    def test_acc_drop_equals_teacher_minus_student(self):
        teacher, student = self._fitted_pair()
        result = evaluate_distillation(teacher, student, _X, _y)
        expected = round(result["teacher_acc"] - result["student_acc"], 4)
        assert result["acc_drop"] == pytest.approx(expected, abs=1e-6)

    def test_metrics_are_floats(self):
        teacher, student = self._fitted_pair()
        result = evaluate_distillation(teacher, student, _X, _y)
        for key in ("teacher_f1", "student_f1", "teacher_acc", "student_acc"):
            assert isinstance(result[key], float), f"{key} should be float"

    def test_f1_values_in_range(self):
        teacher, student = self._fitted_pair()
        result = evaluate_distillation(teacher, student, _X, _y)
        for key in ("teacher_f1", "student_f1"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of [0,1]"

    def test_acc_values_in_range(self):
        teacher, student = self._fitted_pair()
        result = evaluate_distillation(teacher, student, _X, _y)
        for key in ("teacher_acc", "student_acc"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of [0,1]"

    def test_log_cb_is_called(self):
        messages = []
        teacher, student = self._fitted_pair()
        evaluate_distillation(teacher, student, _X, _y, log_cb=messages.append)
        assert len(messages) >= 1

    def test_same_model_zero_drop(self):
        """Comparing a model against itself should yield zero drops."""
        teacher = _make_teacher()
        result = evaluate_distillation(teacher, teacher, _X, _y)
        assert result["f1_drop"] == pytest.approx(0.0, abs=1e-6)
        assert result["acc_drop"] == pytest.approx(0.0, abs=1e-6)
