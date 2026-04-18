# -*- coding: utf-8 -*-
"""
Tests for ml_setfit.py, t5_summarizer.py, and bank_reason_trainer_gui.py.

All tests run without setfit, torch, or transformers installed.
Use: PYTHONPATH=. pytest tests/test_heavy_modules.py -q --tb=short

NOTE: ml_setfit.py requires numpy at module level.  The uv-isolated pytest
binary does not have the system site-packages on its path, so we inject them
here before any project imports.  This mirrors what ``PYTHONPATH=.`` does for
the project root, but also makes numpy/sklearn visible to the test runner.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — make system site-packages (numpy, sklearn, …) visible
# when tests are run via the uv-isolated pytest binary.
# ---------------------------------------------------------------------------

def _ensure_site_packages() -> None:
    """Add system dist-packages directories that contain numpy/sklearn."""
    candidates = [
        "/usr/local/lib/python3.11/dist-packages",
        "/usr/lib/python3/dist-packages",
        "/usr/lib/python3.11/dist-packages",
    ]
    for p in candidates:
        if p not in sys.path and Path(p).is_dir():
            sys.path.insert(0, p)


_ensure_site_packages()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent


# ===========================================================================
# MODULE 1: ml_setfit.py — SetFitClassifier
# ===========================================================================


class TestSetFitClassifierInit:
    """Constructor stores all parameters and initialises lazy attrs to None."""

    def setup_method(self):
        from ml_setfit import SetFitClassifier
        self.SetFitClassifier = SetFitClassifier

    def test_defaults_stored(self):
        clf = self.SetFitClassifier()
        assert clf.model_name == "deepvk/USER2-base"
        assert clf.num_iterations == 20
        assert clf.num_epochs == 3
        assert clf.batch_size == 8
        assert clf.fp16 is True
        assert clf.max_length == 256
        assert clf.device == "auto"

    def test_custom_params_stored(self):
        clf = self.SetFitClassifier(
            model_name="some/model",
            num_iterations=5,
            num_epochs=1,
            batch_size=4,
            fp16=False,
            max_length=128,
            device="cpu",
            cache_dir="/tmp/cache",
        )
        assert clf.model_name == "some/model"
        assert clf.num_iterations == 5
        assert clf.cache_dir == "/tmp/cache"
        assert clf.device == "cpu"

    def test_lazy_attrs_are_none(self):
        clf = self.SetFitClassifier()
        assert clf._model is None
        assert clf._local_model_path is None
        assert clf.classes_ is None

    def test_callbacks_default_to_none(self):
        clf = self.SetFitClassifier()
        assert clf.log_cb is None
        assert clf.progress_cb is None

    def test_progress_range_stored(self):
        clf = self.SetFitClassifier(progress_range=(10.0, 80.0))
        assert clf.progress_range == (10.0, 80.0)


class TestSetFitClassifierIsAvailable:
    """is_available() reflects whether setfit spec can be found."""

    def test_returns_bool(self):
        from ml_setfit import SetFitClassifier
        result = SetFitClassifier.is_available()
        assert isinstance(result, bool)

    def test_false_when_setfit_missing(self):
        from ml_setfit import SetFitClassifier
        # Patch the find_spec used inside ml_setfit's own importlib.util
        with patch("ml_setfit.importlib.util.find_spec", return_value=None):
            assert SetFitClassifier.is_available() is False

    def test_true_when_setfit_present(self):
        from ml_setfit import SetFitClassifier
        fake_spec = MagicMock()
        with patch("ml_setfit.importlib.util.find_spec", return_value=fake_spec):
            assert SetFitClassifier.is_available() is True


class TestSetFitClassifierLog:
    """_log() and _prog() delegate to callbacks when set."""

    def setup_method(self):
        from ml_setfit import SetFitClassifier
        self.SetFitClassifier = SetFitClassifier

    def test_log_calls_callback(self):
        cb = MagicMock()
        clf = self.SetFitClassifier(log_cb=cb)
        clf._log("hello")
        cb.assert_called_once_with("hello")

    def test_log_no_callback_no_error(self):
        clf = self.SetFitClassifier()
        clf._log("should not raise")  # must not throw

    def test_prog_maps_through_progress_range(self):
        cb = MagicMock()
        clf = self.SetFitClassifier(progress_cb=cb, progress_range=(50.0, 100.0))
        clf._prog(50.0, "halfway")
        # pct=50 → mapped = 50 + (100-50)*50/100 = 75
        cb.assert_called_once_with(75.0, "halfway")

    def test_prog_no_callback_no_error(self):
        clf = self.SetFitClassifier()
        clf._prog(100.0, "done")

    def test_prog_zero_maps_to_lo(self):
        cb = MagicMock()
        clf = self.SetFitClassifier(progress_cb=cb, progress_range=(20.0, 80.0))
        clf._prog(0.0, "start")
        cb.assert_called_once_with(20.0, "start")

    def test_prog_hundred_maps_to_hi(self):
        cb = MagicMock()
        clf = self.SetFitClassifier(progress_cb=cb, progress_range=(20.0, 80.0))
        clf._prog(100.0, "end")
        cb.assert_called_once_with(80.0, "end")


class TestSetFitClassifierResolveDevice:
    """_resolve_device() returns correct device string."""

    def setup_method(self):
        from ml_setfit import SetFitClassifier
        self.SetFitClassifier = SetFitClassifier

    def test_explicit_device_returned_as_is(self):
        clf = self.SetFitClassifier(device="cpu")
        assert clf._resolve_device() == "cpu"

    def test_explicit_cuda_returned_as_is(self):
        clf = self.SetFitClassifier(device="cuda")
        assert clf._resolve_device() == "cuda"

    def test_auto_falls_back_to_cpu_when_torch_missing(self):
        clf = self.SetFitClassifier(device="auto")
        with patch.dict(sys.modules, {"torch": None}):
            result = clf._resolve_device()
        assert result == "cpu"

    def test_auto_returns_cpu_when_cuda_unavailable(self):
        clf = self.SetFitClassifier(device="auto")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = clf._resolve_device()
        assert result == "cpu"

    def test_auto_returns_cuda_when_available(self):
        clf = self.SetFitClassifier(device="auto")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = clf._resolve_device()
        assert result == "cuda"


class TestSetFitClassifierDefaultCacheDir:
    """_default_cache_dir() resolves to the correct path."""

    def setup_method(self):
        from ml_setfit import SetFitClassifier
        self.SetFitClassifier = SetFitClassifier

    def test_explicit_cache_dir_used(self):
        clf = self.SetFitClassifier(cache_dir="/my/cache")
        assert clf._default_cache_dir() == Path("/my/cache")

    def test_default_cache_dir_is_sbert_models_next_to_module(self):
        clf = self.SetFitClassifier()
        result = clf._default_cache_dir()
        assert result.name == "sbert_models"
        assert result.parent == PROJECT_ROOT


class TestSetFitClassifierFit:
    """fit() raises ImportError when setfit is not installed."""

    def test_fit_raises_import_error_when_unavailable(self):
        from ml_setfit import SetFitClassifier
        clf = SetFitClassifier()
        with patch.object(SetFitClassifier, "is_available", return_value=False):
            with pytest.raises(ImportError, match="setfit"):
                clf.fit(["text"], ["label"])


class TestSetFitClassifierEnsureModel:
    """_ensure_model() raises appropriate errors before model is loaded."""

    def setup_method(self):
        from ml_setfit import SetFitClassifier
        self.SetFitClassifier = SetFitClassifier

    def test_raises_runtime_error_when_not_fitted(self):
        clf = self.SetFitClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf._ensure_model()

    def test_raises_import_error_when_path_set_but_setfit_missing(self):
        clf = self.SetFitClassifier()
        clf._local_model_path = "/some/path"
        with patch.object(self.SetFitClassifier, "is_available", return_value=False):
            with pytest.raises(ImportError, match="setfit"):
                clf._ensure_model()

    def test_no_error_when_model_already_loaded(self):
        clf = self.SetFitClassifier()
        clf._model = MagicMock()  # simulate already-loaded model
        clf._ensure_model()  # must not raise


class TestSetFitClassifierSerialization:
    """__getstate__ / __setstate__ handle pickle correctly."""

    def setup_method(self):
        from ml_setfit import SetFitClassifier
        self.SetFitClassifier = SetFitClassifier

    def test_getstate_excludes_model_and_callbacks(self):
        log_cb = MagicMock()
        prog_cb = MagicMock()
        clf = self.SetFitClassifier(log_cb=log_cb, progress_cb=prog_cb)
        clf._model = MagicMock()  # pretend a model is loaded

        state = clf.__getstate__()
        assert state["_model"] is None
        assert state["log_cb"] is None
        assert state["progress_cb"] is None

    def test_getstate_preserves_hyperparams(self):
        clf = self.SetFitClassifier(model_name="my/model", num_iterations=5)
        state = clf.__getstate__()
        assert state["model_name"] == "my/model"
        assert state["num_iterations"] == 5

    def test_setstate_sets_model_to_none(self):
        clf = self.SetFitClassifier()
        state = {
            "_model": object(),
            "_local_model_path": "/some/path",
            "model_name": "x",
            "num_iterations": 1,
            "num_epochs": 1,
            "batch_size": 8,
            "fp16": True,
            "max_length": 256,
            "device": "auto",
            "cache_dir": None,
            "log_cb": None,
            "progress_cb": None,
            "progress_range": (50.0, 92.0),
            "classes_": ["A", "B"],
        }
        clf.__setstate__(state)
        assert clf._model is None  # always reset on deserialisation
        assert clf._local_model_path == "/some/path"
        assert clf.classes_ == ["A", "B"]


# ---------------------------------------------------------------------------
# train_model_setfit — skip_val branch
# ---------------------------------------------------------------------------


class TestTrainModelSetfitSkipVal:
    """skip_val path returns 'ВАЛИДАЦИЯ ПРОПУЩЕНА' tuple without real training."""

    def _run(self, X, y, test_size=0.2):
        from ml_setfit import train_model_setfit, SetFitClassifier
        with patch.object(SetFitClassifier, "fit", return_value=None):
            with patch.object(SetFitClassifier, "is_available", return_value=True):
                return train_model_setfit(
                    X=X,
                    y=y,
                    model_name="fake/model",
                    test_size=test_size,
                )

    def test_test_size_zero_triggers_skip_val(self):
        X = ["text"] * 30
        y = ["A"] * 15 + ["B"] * 15
        clf, clf_type, report, labels, cm, extras = self._run(X, y, test_size=0)
        assert "ВАЛИДАЦИЯ ПРОПУЩЕНА" in report
        assert labels is None
        assert cm is None

    def test_single_class_triggers_skip_val(self):
        X = ["text"] * 25
        y = ["A"] * 25
        clf, clf_type, report, labels, cm, extras = self._run(X, y)
        assert "ВАЛИДАЦИЯ ПРОПУЩЕНА" in report

    def test_too_few_samples_triggers_skip_val(self):
        X = ["text"] * 10
        y = ["A"] * 5 + ["B"] * 5
        clf, clf_type, report, labels, cm, extras = self._run(X, y)
        assert "ВАЛИДАЦИЯ ПРОПУЩЕНА" in report

    def test_minority_class_single_sample_triggers_skip_val(self):
        X = ["text"] * 25
        y = ["A"] * 24 + ["B"] * 1
        clf, clf_type, report, labels, cm, extras = self._run(X, y)
        assert "ВАЛИДАЦИЯ ПРОПУЩЕНА" in report

    def test_clf_type_is_setfit(self):
        X = ["text"] * 10
        y = ["A"] * 10
        _, clf_type, _, _, _, _ = self._run(X, y)
        assert clf_type == "SetFit"

    def test_returns_six_tuple(self):
        X = ["text"] * 10
        y = ["A"] * 10
        result = self._run(X, y)
        assert len(result) == 6

    def test_extras_is_dict(self):
        X = ["text"] * 10
        y = ["A"] * 10
        *_, extras = self._run(X, y)
        assert isinstance(extras, dict)


# ===========================================================================
# MODULE 2: t5_summarizer.py
# ===========================================================================


class TestT5RussianSummarizerInit:
    """Constructor stores all parameters correctly."""

    def setup_method(self):
        from t5_summarizer import T5RussianSummarizer, DEFAULT_T5_MODEL, T5_CACHE_DIR
        self.T5RussianSummarizer = T5RussianSummarizer
        self.DEFAULT_T5_MODEL = DEFAULT_T5_MODEL
        self.T5_CACHE_DIR = T5_CACHE_DIR

    def test_default_model_name(self):
        s = self.T5RussianSummarizer()
        assert s.model_name == self.DEFAULT_T5_MODEL

    def test_cache_dir_defaults_to_t5_models(self):
        s = self.T5RussianSummarizer()
        assert s.cache_dir == str(self.T5_CACHE_DIR)

    def test_custom_cache_dir(self):
        s = self.T5RussianSummarizer(cache_dir=Path("/tmp/mymodels"))
        assert s.cache_dir == "/tmp/mymodels"

    def test_lazy_attrs_are_none(self):
        s = self.T5RussianSummarizer()
        assert s._tokenizer is None
        assert s._model is None
        assert s._device_obj is None

    def test_custom_params(self):
        s = self.T5RussianSummarizer(
            max_input_length=256,
            max_target_length=64,
            batch_size=2,
            device="cpu",
        )
        assert s.max_input_length == 256
        assert s.max_target_length == 64
        assert s.batch_size == 2
        assert s.device == "cpu"

    def test_load_lock_is_created(self):
        import threading
        s = self.T5RussianSummarizer()
        # threading.Lock() returns an instance of _thread.lock — check protocol
        assert hasattr(s._load_lock, "acquire") and hasattr(s._load_lock, "release")


class TestT5RussianSummarizerModuleConstants:
    """Module-level constants are sensible."""

    def test_default_model_name_value(self):
        from t5_summarizer import DEFAULT_T5_MODEL
        assert "t5" in DEFAULT_T5_MODEL.lower() or "urukhan" in DEFAULT_T5_MODEL.lower()

    def test_t5_cache_dir_is_path(self):
        from t5_summarizer import T5_CACHE_DIR
        assert isinstance(T5_CACHE_DIR, Path)
        assert T5_CACHE_DIR.name == "t5_models"

    def test_t5_cache_dir_next_to_app(self):
        from t5_summarizer import T5_CACHE_DIR
        assert T5_CACHE_DIR.parent == PROJECT_ROOT


class TestT5RussianSummarizerLoad:
    """load() raises ImportError when transformers/torch are absent."""

    def test_load_raises_import_error_without_transformers(self):
        from t5_summarizer import T5RussianSummarizer
        s = T5RussianSummarizer()
        # In the test environment torch/transformers are not installed
        with pytest.raises(ImportError, match="transformers"):
            s.load()

    def test_load_import_error_mentions_pip_install(self):
        from t5_summarizer import T5RussianSummarizer
        s = T5RussianSummarizer()
        with pytest.raises(ImportError) as exc_info:
            s.load()
        assert "pip install" in str(exc_info.value)


class TestT5RussianSummarizerHelpers:
    """_log() and _prog() delegate to callbacks."""

    def setup_method(self):
        from t5_summarizer import T5RussianSummarizer
        self.T5RussianSummarizer = T5RussianSummarizer

    def test_log_calls_callback(self):
        cb = MagicMock()
        s = self.T5RussianSummarizer(log_cb=cb)
        s._log("test message")
        cb.assert_called_once_with("test message")

    def test_log_silent_without_callback(self):
        s = self.T5RussianSummarizer()
        s._log("no error please")  # must not raise

    def test_prog_clamps_to_zero_one(self):
        cb = MagicMock()
        s = self.T5RussianSummarizer(progress_cb=cb)
        s._prog(1.5, "over")
        args = cb.call_args[0]
        assert args[0] == 1.0  # clamped to 1.0

    def test_prog_clamps_negative(self):
        cb = MagicMock()
        s = self.T5RussianSummarizer(progress_cb=cb)
        s._prog(-0.5, "under")
        args = cb.call_args[0]
        assert args[0] == 0.0  # clamped to 0.0

    def test_prog_passes_status_string(self):
        cb = MagicMock()
        s = self.T5RussianSummarizer(progress_cb=cb)
        s._prog(0.5, "halfway")
        cb.assert_called_once_with(0.5, "halfway")


# ===========================================================================
# MODULE 3: bank_reason_trainer_gui.py (legacy entry point)
# ===========================================================================


class TestBankReasonTrainerGui:
    """Legacy entry point can be imported safely; contains guard block."""

    def test_module_imports_without_error(self):
        """Importing the module at top level must not raise anything."""
        sys.modules.pop("bank_reason_trainer_gui", None)
        mod = importlib.import_module("bank_reason_trainer_gui")
        assert mod is not None

    def test_module_has_docstring_with_deprecated_reference(self):
        import bank_reason_trainer_gui as mod
        doc = mod.__doc__ or ""
        assert "deprecated" in doc.lower()

    def test_module_source_contains_main_guard(self):
        """The __main__ guard must be present in source."""
        src_path = PROJECT_ROOT / "bank_reason_trainer_gui.py"
        source = src_path.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__"' in source

    def test_module_source_contains_deprecation_warning(self):
        src_path = PROJECT_ROOT / "bank_reason_trainer_gui.py"
        source = src_path.read_text(encoding="utf-8")
        assert "DeprecationWarning" in source
        assert "warnings.warn" in source

    def test_module_source_mentions_bootstrap_run(self):
        """Deprecation message should point users to bootstrap_run.py."""
        src_path = PROJECT_ROOT / "bank_reason_trainer_gui.py"
        source = src_path.read_text(encoding="utf-8")
        assert "bootstrap_run" in source
