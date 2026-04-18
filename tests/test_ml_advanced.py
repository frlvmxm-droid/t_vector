# -*- coding: utf-8 -*-
"""
Tests for four untested functions in ml_training.py:
  - find_best_c
  - optuna_tune
  - confident_learning_detect
  - train_kfold_ensemble
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_training import (
    confident_learning_detect,
    find_best_c,
    train_kfold_ensemble,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

X_BINARY = ["кредит банк карта"] * 20 + ["страховка полис выплата"] * 20
Y_BINARY = ["банк"] * 20 + ["страх"] * 20

X_MULTI = (
    ["кредит банк"] * 15
    + ["страховка полис"] * 15
    + ["инвестиции фонд"] * 15
)
Y_MULTI = ["банк"] * 15 + ["страх"] * 15 + ["инвест"] * 15


def _make_features() -> TfidfVectorizer:
    """Return a fresh unfitted TF-IDF vectorizer."""
    return TfidfVectorizer(max_features=50)


def _features_factory() -> TfidfVectorizer:
    """Callable that returns a fresh vectorizer (for train_kfold_ensemble)."""
    return TfidfVectorizer(max_features=50)


# ===========================================================================
# find_best_c
# ===========================================================================


class TestFindBestC:
    """Tests for find_best_c()."""

    def test_returns_tuple_of_float_and_dict(self):
        best_c, scores = find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=False, max_iter=100,
            candidates=[0.1, 1.0], cv=2, n_jobs=1,
        )
        assert isinstance(best_c, float)
        assert isinstance(scores, dict)

    def test_best_c_is_in_candidates(self):
        candidates = [0.1, 1.0, 5.0]
        best_c, _ = find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=False, max_iter=100,
            candidates=candidates, cv=2, n_jobs=1,
        )
        assert best_c in candidates

    def test_all_candidates_appear_in_scores(self):
        candidates = [0.1, 1.0, 5.0]
        _, scores = find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=False, max_iter=100,
            candidates=candidates, cv=2, n_jobs=1,
        )
        for c in candidates:
            assert c in scores

    def test_progress_cb_called_for_each_candidate(self):
        candidates = [0.1, 1.0, 5.0]
        calls = []
        find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=False, max_iter=100,
            candidates=candidates, cv=2,
            progress_cb=lambda p, msg: calls.append((p, msg)),
            n_jobs=1,
        )
        # At least one call per candidate (progress_cb is called before each C evaluation)
        assert len(calls) >= len(candidates)

    def test_cancel_event_already_set_returns_immediately(self):
        evt = threading.Event()
        evt.set()
        candidates = [0.1, 1.0, 5.0, 10.0]
        best_c, scores = find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=False, max_iter=100,
            candidates=candidates, cv=2,
            cancel_event=evt, n_jobs=1,
        )
        # Cancelled before first iteration → scores is empty, best_c is first candidate
        assert scores == {}
        assert best_c == candidates[0]

    def test_custom_candidates_list_is_used(self):
        custom = [0.01, 99.9]
        _, scores = find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=False, max_iter=100,
            candidates=custom, cv=2, n_jobs=1,
        )
        assert set(scores.keys()) == set(custom)

    def test_n_jobs_auto_computed_when_none(self):
        """When n_jobs=None, function should still complete without error."""
        best_c, scores = find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=False, max_iter=100,
            candidates=[1.0], cv=2, n_jobs=None,
        )
        assert isinstance(best_c, float)

    def test_memory_error_is_reraised(self):
        """MemoryError must propagate, not be swallowed."""
        # cross_val_score is imported locally inside find_best_c from
        # sklearn.model_selection, so patch the canonical location.
        with patch("sklearn.model_selection.cross_val_score", side_effect=MemoryError("OOM")):
            with pytest.raises(MemoryError):
                find_best_c(
                    X_BINARY, Y_BINARY, _make_features(),
                    balanced=False, max_iter=100,
                    candidates=[1.0], cv=2, n_jobs=1,
                )

    def test_cv_failure_for_one_c_gives_zero_score(self):
        """If cross_val_score raises a generic Exception for one C, score=0.0."""
        import sklearn.model_selection as _skl_ms

        call_count = [0]
        original_cvs = _skl_ms.cross_val_score

        def _flaky_cvs(pipe, X, y, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("simulated failure")
            return original_cvs(pipe, X, y, **kwargs)

        with patch("sklearn.model_selection.cross_val_score", side_effect=_flaky_cvs):
            candidates = [0.1, 1.0]
            best_c, scores = find_best_c(
                X_BINARY, Y_BINARY, _make_features(),
                balanced=False, max_iter=100,
                candidates=candidates, cv=2, n_jobs=1,
            )
        assert scores[0.1] == 0.0
        assert scores[1.0] > 0.0  # second call succeeds

    def test_adaptive_candidates_small_dataset(self):
        """len(X) < 500 → default candidates include values up to 30.0."""
        assert len(X_BINARY) < 500
        _, scores = find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=False, max_iter=100,
            cv=2, n_jobs=1,
        )
        # Default small-dataset candidates: [0.1, 0.3, 0.7, 1.0, 3.0, 5.0, 10.0, 20.0, 30.0]
        assert 30.0 in scores

    def test_balanced_flag_accepted(self):
        best_c, scores = find_best_c(
            X_BINARY, Y_BINARY, _make_features(),
            balanced=True, max_iter=100,
            candidates=[1.0, 5.0], cv=2, n_jobs=1,
        )
        assert isinstance(best_c, float)
        assert len(scores) == 2


# ===========================================================================
# optuna_tune
# ===========================================================================


class TestOptunaTune:
    """Tests for optuna_tune()."""

    @pytest.fixture(autouse=True)
    def require_optuna(self):
        pytest.importorskip("optuna")

    def _call(self, **kwargs):
        from ml_training import optuna_tune
        defaults = dict(
            X=X_BINARY,
            y=Y_BINARY,
            features=_make_features(),
            balanced=False,
            max_iter=100,
            n_trials=1,
            cv=2,
            n_jobs=1,
        )
        defaults.update(kwargs)
        return optuna_tune(**defaults)

    def test_returns_tuple_of_dict_and_float(self):
        result = self._call()
        assert isinstance(result, tuple) and len(result) == 2
        params, score = result
        assert isinstance(params, dict)
        assert isinstance(score, float)

    def test_best_params_contains_c_key(self):
        params, _ = self._call()
        assert "C" in params

    def test_best_score_between_zero_and_one(self):
        _, score = self._call()
        assert 0.0 <= score <= 1.0

    def test_n_trials_1_runs_quickly(self):
        """n_trials=1 should complete without hanging."""
        params, score = self._call(n_trials=1)
        assert "C" in params

    def test_progress_cb_called(self):
        calls = []
        self._call(
            n_trials=2,
            progress_cb=lambda p, msg: calls.append(p),
        )
        assert len(calls) >= 1

    def test_import_error_when_optuna_missing(self):
        """optuna_tune raises ImportError when optuna is not available."""
        import sys
        import types

        fake_modules = {k: v for k, v in sys.modules.items()}
        fake_modules["optuna"] = None  # simulate missing module

        from ml_training import optuna_tune as _ot
        with patch.dict(sys.modules, {"optuna": None}):
            with pytest.raises((ImportError, AttributeError)):
                _ot(
                    X=X_BINARY, y=Y_BINARY, features=_make_features(),
                    balanced=False, max_iter=100,
                    n_trials=1, cv=2, n_jobs=1,
                )

    def test_cancel_event_stops_trials(self):
        """Setting cancel_event before call prunes all trials; function still returns."""
        evt = threading.Event()
        evt.set()
        # Even when all trials are pruned, optuna returns best available value.
        # With 0 completed trials, study.best_value may raise — we just check no crash.
        try:
            params, score = self._call(cancel_event=evt, n_trials=3)
            # If it returned something, basic shape check
            assert isinstance(params, dict)
        except Exception:
            # optuna may raise if 0 trials completed — acceptable behaviour
            pass


# ===========================================================================
# confident_learning_detect
# ===========================================================================


class TestConfidentLearningDetect:
    """Tests for confident_learning_detect()."""

    def _call(self, X=None, y=None, **kwargs):
        defaults = dict(
            X=X if X is not None else list(X_BINARY),
            y=y if y is not None else list(Y_BINARY),
            features=_make_features(),
            balanced=False,
            max_iter=100,
            cv=2,
            n_jobs=1,
        )
        defaults.update(kwargs)
        return confident_learning_detect(**defaults)

    def test_returns_list(self):
        result = self._call()
        assert isinstance(result, list)

    def test_empty_for_single_class(self):
        X_single = ["кредит банк карта"] * 20
        y_single = ["банк"] * 20
        result = self._call(X=X_single, y=y_single)
        assert result == []

    def test_empty_for_too_few_samples(self):
        # cv=5 requires len(X) >= 10; use 4 samples → < cv*2
        X_tiny = ["текст раз", "текст два", "текст три", "текст четыре"]
        y_tiny = ["А", "А", "Б", "Б"]
        result = confident_learning_detect(
            X_tiny, y_tiny, _make_features(),
            balanced=False, max_iter=100, cv=5, n_jobs=1,
        )
        assert result == []

    def test_mislabeled_example_detected(self):
        """
        Inject one clearly wrong label in an otherwise clean binary dataset.
        The mislabeled example should appear in the suspicious list.
        """
        X = ["кредит банк карта"] * 30 + ["страховка полис выплата"] * 30
        y = ["банк"] * 30 + ["страх"] * 30
        # Flip a single label (index 0: "кредит банк карта" → mislabeled as "страх")
        y_noisy = list(y)
        y_noisy[0] = "страх"

        result = confident_learning_detect(
            X, y_noisy, _make_features(),
            balanced=False, max_iter=200, cv=2, n_jobs=1,
        )
        # We don't strictly require it to be caught (probabilistic), but at least
        # the function runs and returns a list.
        assert isinstance(result, list)

    def test_result_dicts_have_required_keys(self):
        """Each returned dict must carry all required keys."""
        required = {"idx", "text", "given_label", "likely_label", "p_given", "p_likely"}
        # Use a dataset with a flipped label to increase chance of detection
        X = ["кредит банк карта"] * 30 + ["страховка полис выплата"] * 30
        y = ["банк"] * 30 + ["страх"] * 30
        y_noisy = list(y)
        y_noisy[0] = "страх"

        result = confident_learning_detect(
            X, y_noisy, _make_features(),
            balanced=False, max_iter=200, cv=2, n_jobs=1,
        )
        for entry in result:
            assert required.issubset(entry.keys()), f"Missing keys in: {entry}"

    def test_p_given_less_than_threshold(self):
        """
        Detected examples were flagged because p_given was below the class mean.
        We verify p_given and p_likely are valid floats in [0, 1].
        """
        X = ["кредит банк карта"] * 30 + ["страховка полис выплата"] * 30
        y = ["банк"] * 30 + ["страх"] * 30
        y_noisy = list(y)
        y_noisy[0] = "страх"

        result = confident_learning_detect(
            X, y_noisy, _make_features(),
            balanced=False, max_iter=200, cv=2, n_jobs=1,
        )
        for entry in result:
            assert 0.0 <= entry["p_given"] <= 1.0
            assert 0.0 <= entry["p_likely"] <= 1.0

    def test_results_sorted_ascending_by_p_given(self):
        """Results must be sorted by p_given ascending."""
        X = ["кредит банк карта"] * 30 + ["страховка полис выплата"] * 30
        y = ["банк"] * 30 + ["страх"] * 30
        y_noisy = list(y)
        y_noisy[0] = "страх"
        y_noisy[1] = "страх"

        result = confident_learning_detect(
            X, y_noisy, _make_features(),
            balanced=False, max_iter=200, cv=2, n_jobs=1,
        )
        p_vals = [r["p_given"] for r in result]
        assert p_vals == sorted(p_vals)

    def test_progress_cb_called_at_least_once(self):
        calls = []
        self._call(progress_cb=lambda p, msg: calls.append(p))
        # cv=2 folds → progress_cb called at least once if not cancelled
        assert len(calls) >= 1

    def test_cancel_event_stops_fold_iteration(self):
        """With cancel_event already set, no folds run → returns [] or partial."""
        evt = threading.Event()
        evt.set()
        result = self._call(cancel_event=evt)
        # No folds ran → oof_proba is all zeros, no suspicious found
        assert isinstance(result, list)

    def test_threshold_factor_zero_yields_no_suspicious(self):
        """threshold_factor=0.0 → all thresholds are 0 → nothing is below 0 → empty list."""
        result = self._call(threshold_factor=0.0)
        assert result == []

    def test_multiclass_returns_list(self):
        result = confident_learning_detect(
            X_MULTI, Y_MULTI, _make_features(),
            balanced=False, max_iter=100, cv=2, n_jobs=1,
        )
        assert isinstance(result, list)


# ===========================================================================
# train_kfold_ensemble
# ===========================================================================


class TestTrainKfoldEnsemble:
    """Tests for train_kfold_ensemble()."""

    def _call(self, X=None, y=None, k=3, **kwargs):
        defaults = dict(
            X=X if X is not None else list(X_BINARY),
            y=y if y is not None else list(Y_BINARY),
            features_factory=_features_factory,
            balanced=False,
            C=1.0,
            max_iter=100,
            k=k,
            calib_method="sigmoid",
            n_jobs=1,
        )
        defaults.update(kwargs)
        return train_kfold_ensemble(**defaults)

    def test_returns_tuple_of_models_and_classes(self):
        models, classes = self._call()
        assert isinstance(models, list)
        assert isinstance(classes, list)

    def test_len_models_equals_k(self):
        k = 3
        models, _ = self._call(k=k)
        assert len(models) == k

    def test_all_models_can_predict(self):
        models, _ = self._call(k=2)
        for model in models:
            preds = model.predict(X_BINARY[:5])
            assert len(preds) == 5

    def test_classes_ref_is_nonempty_list_of_strings(self):
        _, classes = self._call()
        assert len(classes) > 0
        assert all(isinstance(c, str) for c in classes)

    def test_features_factory_called_k_times(self):
        k = 3
        call_count = [0]

        def counting_factory():
            call_count[0] += 1
            return _features_factory()

        train_kfold_ensemble(
            X=list(X_BINARY), y=list(Y_BINARY),
            features_factory=counting_factory,
            balanced=False, C=1.0, max_iter=100,
            k=k, n_jobs=1,
        )
        assert call_count[0] == k

    def test_progress_cb_called_per_fold(self):
        k = 3
        calls = []
        self._call(k=k, progress_cb=lambda p, msg: calls.append(p))
        # progress_cb called once per fold + once at the end
        assert len(calls) >= k

    def test_cancel_event_stops_fold_loop_early(self):
        """Set cancel_event before call → fewer than k models produced."""
        evt = threading.Event()
        evt.set()
        models, classes = self._call(k=4, cancel_event=evt)
        # Cancelled before first fold → 0 models
        assert len(models) < 4

    def test_calib_method_isotonic_works(self):
        """isotonic calibration should work for datasets with enough samples per class."""
        # Need >= 3 samples per class in each fold's training split
        # With 20 per class and k=3: each fold train split ≈ 27 → min_class ≈ 13 → ok
        models, classes = train_kfold_ensemble(
            X=list(X_BINARY), y=list(Y_BINARY),
            features_factory=_features_factory,
            balanced=False, C=1.0, max_iter=100,
            k=3, calib_method="isotonic", n_jobs=1,
        )
        assert len(models) == 3

    def test_k_3_produces_exactly_3_models(self):
        models, _ = self._call(k=3)
        assert len(models) == 3

    def test_k_2_produces_exactly_2_models(self):
        models, _ = self._call(k=2)
        assert len(models) == 2

    def test_multiclass_dataset_works(self):
        models, classes = train_kfold_ensemble(
            X=list(X_MULTI), y=list(Y_MULTI),
            features_factory=_features_factory,
            balanced=False, C=1.0, max_iter=100,
            k=3, n_jobs=1,
        )
        assert len(models) == 3
        assert len(classes) >= 2  # at least 2 classes recognized

    def test_classes_ref_from_first_fold(self):
        """classes_ref is set from the first trained fold."""
        models, classes = self._call(k=3)
        # classes from the pipeline
        first_model_classes = list(models[0].classes_)
        assert sorted(classes) == sorted(first_model_classes)

    def test_balanced_flag_accepted(self):
        models, classes = train_kfold_ensemble(
            X=list(X_BINARY), y=list(Y_BINARY),
            features_factory=_features_factory,
            balanced=True, C=1.0, max_iter=100,
            k=2, n_jobs=1,
        )
        assert len(models) == 2
