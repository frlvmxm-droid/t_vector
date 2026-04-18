# -*- coding: utf-8 -*-
"""
Unit tests for ml_setfit.SetFitClassifier.

Heavy-path tests (actual fit/predict) are skipped when setfit is absent —
in CI/dev machines without torch+setfit we only verify:
  * is_available() matches the spec probe
  * __init__ stores hyperparameters
  * _resolve_device() logic (cpu fallback)
  * _default_cache_dir() logic
  * fit()/predict()/predict_proba() raise ImportError when setfit missing
  * pickle round-trip preserves hyperparameters and clears the torch model
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest

from ml_setfit import SetFitClassifier


HAS_SETFIT = importlib.util.find_spec("setfit") is not None


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_returns_bool(self):
        assert isinstance(SetFitClassifier.is_available(), bool)

    def test_matches_spec_lookup(self):
        assert SetFitClassifier.is_available() == HAS_SETFIT


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        clf = SetFitClassifier()
        assert clf.model_name == "deepvk/USER2-base"
        assert clf.num_iterations == 20
        assert clf.num_epochs == 3
        assert clf.batch_size == 8
        assert clf.device == "auto"
        assert clf._model is None
        assert clf.classes_ is None

    def test_custom_params(self):
        clf = SetFitClassifier(
            model_name="my/model",
            num_iterations=5,
            num_epochs=1,
            batch_size=4,
            fp16=False,
            device="cpu",
            progress_range=(10.0, 90.0),
        )
        assert clf.model_name == "my/model"
        assert clf.num_iterations == 5
        assert clf.num_epochs == 1
        assert clf.fp16 is False
        assert clf.device == "cpu"
        assert clf.progress_range == (10.0, 90.0)


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_explicit_device_passthrough(self):
        clf = SetFitClassifier(device="cpu")
        assert clf._resolve_device() == "cpu"

    def test_auto_returns_cpu_or_cuda(self):
        clf = SetFitClassifier(device="auto")
        assert clf._resolve_device() in {"cpu", "cuda"}


# ---------------------------------------------------------------------------
# _default_cache_dir
# ---------------------------------------------------------------------------

class TestDefaultCacheDir:
    def test_explicit_cache_dir(self, tmp_path):
        clf = SetFitClassifier(cache_dir=str(tmp_path))
        assert clf._default_cache_dir() == tmp_path

    def test_falls_back_to_module_sbert_models(self):
        clf = SetFitClassifier()
        cache = clf._default_cache_dir()
        assert isinstance(cache, Path)
        assert cache.name == "sbert_models"


# ---------------------------------------------------------------------------
# ImportError paths (when setfit is missing)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HAS_SETFIT, reason="setfit installed — cannot test missing-dep path")
class TestImportErrorPaths:
    def test_fit_raises_without_setfit(self):
        clf = SetFitClassifier()
        with pytest.raises(ImportError, match="setfit"):
            clf.fit(["hello"], ["A"])

    def test_predict_without_fit_raises(self):
        clf = SetFitClassifier()
        with pytest.raises((RuntimeError, ImportError)):
            clf.predict(["hello"])


# ---------------------------------------------------------------------------
# _ensure_model guard (no fit, no path)
# ---------------------------------------------------------------------------

class TestEnsureModelGuard:
    def test_raises_when_no_path_and_no_model(self):
        clf = SetFitClassifier()
        with pytest.raises((RuntimeError, ImportError)):
            clf._ensure_model()


# ---------------------------------------------------------------------------
# Pickle round-trip (serialization contract)
# ---------------------------------------------------------------------------

class TestPickleRoundTrip:
    def test_roundtrip_preserves_hyperparameters(self):
        clf = SetFitClassifier(
            model_name="my/model",
            num_iterations=7,
            num_epochs=2,
            batch_size=16,
            fp16=False,
            device="cpu",
        )
        clf.classes_ = ["A", "B"]
        clf._local_model_path = "/some/path"
        # simulate a fitted model field
        clf._model = object()

        blob = pickle.dumps(clf)
        restored = pickle.loads(blob)

        assert restored.model_name == "my/model"
        assert restored.num_iterations == 7
        assert restored.num_epochs == 2
        assert restored.batch_size == 16
        assert restored.fp16 is False
        assert restored.device == "cpu"
        assert restored.classes_ == ["A", "B"]
        assert restored._local_model_path == "/some/path"
        # Torch model is NOT pickled; lazy-loaded on next predict()
        assert restored._model is None

    def test_roundtrip_clears_callbacks(self):
        clf = SetFitClassifier(
            log_cb=lambda m: None, progress_cb=lambda p, s: None
        )
        restored = pickle.loads(pickle.dumps(clf))
        assert restored.log_cb is None
        assert restored.progress_cb is None


# ---------------------------------------------------------------------------
# Callback wiring
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_log_callback_invoked(self):
        logs = []
        clf = SetFitClassifier(log_cb=logs.append)
        clf._log("hello")
        assert logs == ["hello"]

    def test_log_no_cb_silent(self):
        clf = SetFitClassifier()
        clf._log("no-op")  # should not raise

    def test_progress_maps_range(self):
        recorded = []
        clf = SetFitClassifier(
            progress_cb=lambda pct, status: recorded.append((pct, status)),
            progress_range=(50.0, 90.0),
        )
        clf._prog(0.0, "start")
        clf._prog(100.0, "done")
        clf._prog(50.0, "mid")
        pcts = [r[0] for r in recorded]
        # 0% → 50 (low), 100% → 90 (high), 50% → 70 (middle)
        assert pcts[0] == pytest.approx(50.0)
        assert pcts[1] == pytest.approx(90.0)
        assert pcts[2] == pytest.approx(70.0)
