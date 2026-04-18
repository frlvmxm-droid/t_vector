# -*- coding: utf-8 -*-
"""
Tests for bootstrap_run.py, setup_env.py, download_sbert_models.py,
app_deps.py, app_cluster_workflow.py, and experiment_history_dialog.py
(non-tkinter logic only).
"""
from __future__ import annotations

import sys
import pathlib
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out tkinter and any heavy C-extension / optional modules before any
# module-level import that transitively pulls them in.  This is required
# because pytest may run in an isolated environment (e.g. uv tool) that does
# not have tkinter, joblib, sklearn, etc. installed.
# ---------------------------------------------------------------------------
def _make_mock(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = MagicMock()
    return m


import importlib.util as _ilu

def _needs_mock(name: str) -> bool:
    """Only mock modules that are truly not installed (avoid clobbering real packages)."""
    root = name.split(".")[0]
    return _ilu.find_spec(root) is None

_EARLY_MOCKS = [
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "ui_theme",
    "ui_widgets",
    # These are only mocked when not actually installed
    "joblib",
    "numpy",
    "scipy",
    "scipy.sparse",
    "scipy.stats",
    "pandas",
    "sklearn",
    "sklearn.pipeline",
    "sklearn.preprocessing",
    "sklearn.feature_extraction",
    "sklearn.feature_extraction.text",
    "sklearn.svm",
    "sklearn.calibration",
    "sklearn.base",
    "sklearn.utils",
    "sklearn.metrics",
    "sklearn.linear_model",
    "sklearn.decomposition",
    "sklearn.neighbors",
    "ml_vectorizers",
]

for _mod_name in _EARLY_MOCKS:
    if _mod_name not in sys.modules and _needs_mock(_mod_name):
        sys.modules[_mod_name] = _make_mock(_mod_name)

# ml_core needs SBERT_LOCAL_DIR attribute — only mock if not already present
if "ml_core" not in sys.modules and _needs_mock("ml_core"):
    _ml_core_mock = _make_mock("ml_core")
    _ml_core_mock.SBERT_LOCAL_DIR = pathlib.Path("/tmp/sbert_models")
    sys.modules["ml_core"] = _ml_core_mock
elif "ml_core" in sys.modules and not hasattr(sys.modules["ml_core"], "SBERT_LOCAL_DIR"):
    sys.modules["ml_core"].SBERT_LOCAL_DIR = pathlib.Path("/tmp/sbert_models")

# ---------------------------------------------------------------------------
# MODULE 1: bootstrap_run.py
# ---------------------------------------------------------------------------

import bootstrap_run
from bootstrap_run import (
    MIN_PYTHON,
    _banner,
    _log_error,
    _get_distro_id,
    _check_version,
    _check_python_version,
)


class TestMinPython:
    def test_min_python_value(self):
        assert MIN_PYTHON == (3, 9)


class TestBanner:
    def test_banner_prints_three_lines_with_text(self, capsys):
        _banner("hello")
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.strip()]
        assert any("hello" in line for line in lines)
        assert len(lines) == 3

    def test_banner_contains_equals_separators(self, capsys):
        _banner("test text")
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.strip()]
        # First and last lines should be separator lines
        assert "=" * 10 in lines[0]
        assert "=" * 10 in lines[2]


class TestLogError:
    def test_log_error_creates_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        with patch.object(bootstrap_run, "LOG_FILE", log_file):
            _log_error("test error message")
        assert log_file.exists()

    def test_log_error_writes_text(self, tmp_path):
        log_file = tmp_path / "test.log"
        with patch.object(bootstrap_run, "LOG_FILE", log_file):
            _log_error("test error message")
        content = log_file.read_text(encoding="utf-8")
        assert "test error message" in content

    def test_log_error_appends(self, tmp_path):
        log_file = tmp_path / "test.log"
        with patch.object(bootstrap_run, "LOG_FILE", log_file):
            _log_error("first error")
            _log_error("second error")
        content = log_file.read_text(encoding="utf-8")
        assert "first error" in content
        assert "second error" in content

    def test_log_error_includes_timestamp(self, tmp_path):
        log_file = tmp_path / "test.log"
        with patch.object(bootstrap_run, "LOG_FILE", log_file):
            _log_error("timestamped message")
        content = log_file.read_text(encoding="utf-8")
        # Timestamp format is [YYYY-MM-DD HH:MM:SS]
        assert "[20" in content  # year starts with 20xx


class TestGetDistroId:
    def test_returns_str(self):
        result = _get_distro_id()
        assert isinstance(result, str)

    def test_returns_lowercase(self):
        result = _get_distro_id()
        assert result == result.lower()

    def test_handles_missing_freedesktop(self):
        """Falls back gracefully on platforms without freedesktop_os_release."""
        import platform
        with patch.object(platform, "freedesktop_os_release", side_effect=AttributeError):
            result = _get_distro_id()
        assert isinstance(result, str)


class TestCheckVersion:
    def test_installed_package_pandas(self):
        # pandas >= 2.1 is installed per requirements
        assert _check_version("pandas", (1, 0)) is True

    def test_nonexistent_package_returns_true(self):
        # Fail-safe: unknown package must NOT block startup
        assert _check_version("nonexistent_xyz_pkg", (1, 0)) is True

    def test_mocked_version_below_min_returns_false(self):
        # _check_version uses `from importlib.metadata import version as pkg_version`
        # inside the function body, so patch at importlib.metadata.version
        import importlib.metadata
        with patch.object(importlib.metadata, "version", return_value="1.5.0"):
            result = _check_version("some_pkg", (2, 0))
        assert result is False

    def test_mocked_version_equal_min_returns_true(self):
        import importlib.metadata
        with patch.object(importlib.metadata, "version", return_value="2.0.0"):
            result = _check_version("some_pkg", (2, 0))
        assert result is True

    def test_mocked_version_above_min_returns_true(self):
        import importlib.metadata
        with patch.object(importlib.metadata, "version", return_value="2.1.0"):
            result = _check_version("some_pkg", (2, 0))
        assert result is True

    def test_exception_in_version_lookup_returns_true(self):
        import importlib.metadata
        with patch.object(importlib.metadata, "version", side_effect=Exception("oops")):
            result = _check_version("some_pkg", (1, 0))
        assert result is True


class TestCheckPythonVersion:
    def test_current_python_does_not_exit(self):
        """Running Python is >= 3.9, so _check_python_version must not exit."""
        # Ensure sys.version_info is high enough (it is, since we're running tests)
        assert sys.version_info >= (3, 9)
        # Must complete without calling sys.exit
        with patch("bootstrap_run._wait_and_exit") as mock_exit:
            _check_python_version()
        mock_exit.assert_not_called()

    def test_old_python_calls_wait_and_exit(self):
        with patch("sys.version_info", (3, 7, 0, "final", 0)):
            with patch("bootstrap_run._wait_and_exit") as mock_exit:
                _check_python_version()
        mock_exit.assert_called_once()


# ---------------------------------------------------------------------------
# MODULE 2: setup_env.py
# ---------------------------------------------------------------------------

from setup_env import (
    DEFAULT_SBERT_MODEL,
    REQUIRED,
    _ask,
    _ok,
    _info,
    _warn,
    _err,
    _sep,
    _header,
)


class TestSetupEnvConstants:
    def test_default_sbert_model(self):
        assert DEFAULT_SBERT_MODEL == "cointegrated/rubert-tiny2"

    def test_required_is_list(self):
        assert isinstance(REQUIRED, list)

    def test_required_has_at_least_five_items(self):
        assert len(REQUIRED) >= 5

    def test_required_items_have_three_elements(self):
        for item in REQUIRED:
            assert len(item) == 3, f"Item {item!r} does not have 3 elements"

    def test_required_items_third_element_is_tuple(self):
        for item in REQUIRED:
            assert isinstance(item[2], tuple), f"Third element of {item!r} is not a tuple"


class TestAsk:
    def test_ask_y_returns_true(self):
        with patch("builtins.input", return_value="y"):
            assert _ask("question?") is True

    def test_ask_yes_returns_true(self):
        with patch("builtins.input", return_value="yes"):
            assert _ask("question?") is True

    def test_ask_da_returns_true(self):
        with patch("builtins.input", return_value="да"):
            assert _ask("question?") is True

    def test_ask_d_returns_true(self):
        with patch("builtins.input", return_value="д"):
            assert _ask("question?") is True

    def test_ask_n_returns_false(self):
        with patch("builtins.input", return_value="n"):
            assert _ask("question?") is False

    def test_ask_no_returns_false(self):
        with patch("builtins.input", return_value="no"):
            assert _ask("question?") is False

    def test_ask_empty_default_true(self):
        with patch("builtins.input", return_value=""):
            assert _ask("question?", default_yes=True) is True

    def test_ask_empty_default_false(self):
        with patch("builtins.input", return_value=""):
            assert _ask("question?", default_yes=False) is False

    def test_ask_eoferror_returns_default_yes_true(self):
        with patch("builtins.input", side_effect=EOFError):
            assert _ask("question?", default_yes=True) is True

    def test_ask_eoferror_returns_default_yes_false(self):
        with patch("builtins.input", side_effect=EOFError):
            assert _ask("question?", default_yes=False) is False


class TestPrintHelpers:
    def test_ok_output(self, capsys):
        _ok("msg")
        out = capsys.readouterr().out
        assert "  [OK]   msg" in out

    def test_info_output(self, capsys):
        _info("msg")
        out = capsys.readouterr().out
        assert "  [INFO] msg" in out

    def test_warn_output(self, capsys):
        _warn("msg")
        out = capsys.readouterr().out
        assert "  [WARN] msg" in out

    def test_err_output(self, capsys):
        _err("msg")
        out = capsys.readouterr().out
        assert "  [ERR]  msg" in out

    def test_sep_default_prints_62_equals(self, capsys):
        _sep()
        out = capsys.readouterr().out.strip()
        assert out == "=" * 62

    def test_sep_custom_char_and_width(self, capsys):
        _sep("x", 5)
        out = capsys.readouterr().out.strip()
        assert out == "xxxxx"

    def test_header_prints_three_content_lines(self, capsys):
        _header("title")
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.strip()]
        assert len(lines) == 3
        assert any("title" in line for line in lines)


# ---------------------------------------------------------------------------
# MODULE 3: download_sbert_models.py
# ---------------------------------------------------------------------------

import download_sbert_models as dsm


class TestDownloadSbertModels:
    def test_models_is_dict(self):
        assert isinstance(dsm.MODELS, dict)

    def test_models_has_at_least_three_entries(self):
        assert len(dsm.MODELS) >= 3

    def test_all_keys_are_strings_with_slash(self):
        for key in dsm.MODELS:
            assert isinstance(key, str)
            assert "/" in key, f"Key {key!r} does not contain '/'"

    def test_tiny_models_is_list(self):
        assert isinstance(dsm.TINY_MODELS, list)

    def test_tiny_models_are_subset_of_models(self):
        for m in dsm.TINY_MODELS:
            assert m in dsm.MODELS

    def test_large_models_does_not_overlap_tiny(self):
        for m in dsm.LARGE_MODELS:
            assert m not in dsm.TINY_MODELS

    def test_check_installed_returns_bool(self):
        result = dsm._check_installed()
        assert isinstance(result, bool)

    def test_sbert_dir_is_path(self):
        assert isinstance(dsm.SBERT_DIR, pathlib.Path)

    def test_main_is_callable(self):
        assert callable(dsm.main)


# ---------------------------------------------------------------------------
# MODULE 4: app_deps.py — constants only (tkinter mocked)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_deps_module():
    mocks = {
        "tkinter": MagicMock(),
        "tkinter.ttk": MagicMock(),
        "ui_theme": MagicMock(),
        "ui_widgets": MagicMock(),
    }
    import_patch = {}
    for name, mock in mocks.items():
        if name not in sys.modules:
            import_patch[name] = mock

    with patch.dict("sys.modules", import_patch):
        import importlib
        # app_deps may already be cached — force fresh load
        if "app_deps" in sys.modules:
            ad = sys.modules["app_deps"]
        else:
            import app_deps as ad
        yield ad


class TestAppDepsConstants:
    def test_core_packages_contains_pandas(self, app_deps_module):
        pip_names = [entry[0] for entry in app_deps_module.CORE_PACKAGES]
        assert "pandas" in pip_names

    def test_optional_packages_contains_sentence_transformers(self, app_deps_module):
        pip_names = [entry[0] for entry in app_deps_module.OPTIONAL_PACKAGES]
        assert "sentence-transformers" in pip_names

    def test_model_sizes_rubert_tiny2(self, app_deps_module):
        assert app_deps_module._MODEL_SIZES["cointegrated/rubert-tiny2"] == "118 MB"

    def test_model_vram_rubert_tiny2(self, app_deps_module):
        assert app_deps_module._MODEL_VRAM["cointegrated/rubert-tiny2"] == "1 GB"

    def test_core_packages_each_entry_has_four_elements(self, app_deps_module):
        for entry in app_deps_module.CORE_PACKAGES:
            assert len(entry) == 4, f"Entry {entry!r} does not have 4 elements"

    def test_core_packages_install_args_is_list_of_strings(self, app_deps_module):
        for pip_name, import_name, install_args, desc in app_deps_module.CORE_PACKAGES:
            assert isinstance(install_args, list)
            for arg in install_args:
                assert isinstance(arg, str)

    def test_deps_tab_mixin_class_exists(self, app_deps_module):
        assert hasattr(app_deps_module, "DepsTabMixin")
        assert isinstance(app_deps_module.DepsTabMixin, type)


# ---------------------------------------------------------------------------
# MODULE 5: app_cluster_workflow.py
# ---------------------------------------------------------------------------

from app_cluster_workflow import (
    validate_cluster_preconditions,
    build_validated_cluster_snapshot,
)


class TestValidateClusterPreconditions:
    def test_with_files_returns_true(self):
        app = MagicMock()
        app.cluster_files = ["file.xlsx"]
        assert validate_cluster_preconditions(app) is True

    def test_no_files_calls_reject_start(self):
        app = MagicMock()
        app.cluster_files = []
        with patch("app_cluster_workflow.reject_start", return_value=False) as mock_reject:
            result = validate_cluster_preconditions(app)
        mock_reject.assert_called_once()
        assert result is False

    def test_no_files_result_is_false(self):
        app = MagicMock()
        app.cluster_files = []
        with patch("app_cluster_workflow.reject_start", return_value=False):
            result = validate_cluster_preconditions(app)
        assert result is False


class TestBuildValidatedClusterSnapshot:
    def test_returns_snapshot_dict(self):
        app = MagicMock()
        app._snap_params.return_value = {"llm_api_key": "secret123"}
        with patch("app_cluster_workflow.ClusterWorkflowConfig") as mock_cfg, \
             patch("app_cluster_workflow.encrypt_api_key_for_snapshot", return_value="enc"):
            mock_cfg.from_snapshot.return_value = MagicMock()
            result = build_validated_cluster_snapshot(app)
        assert result is not None

    def test_api_key_is_cleared(self):
        app = MagicMock()
        app._snap_params.return_value = {"llm_api_key": "secret123"}
        with patch("app_cluster_workflow.ClusterWorkflowConfig") as mock_cfg, \
             patch("app_cluster_workflow.encrypt_api_key_for_snapshot", return_value="enc"):
            mock_cfg.from_snapshot.return_value = MagicMock()
            result = build_validated_cluster_snapshot(app)
        assert result["llm_api_key"] == ""

    def test_encrypted_key_stored(self):
        app = MagicMock()
        app._snap_params.return_value = {"llm_api_key": "secret123"}
        with patch("app_cluster_workflow.ClusterWorkflowConfig") as mock_cfg, \
             patch("app_cluster_workflow.encrypt_api_key_for_snapshot", return_value="enc"):
            mock_cfg.from_snapshot.return_value = MagicMock()
            result = build_validated_cluster_snapshot(app)
        assert result["llm_api_key_encrypted"] == "enc"

    def test_invalid_config_returns_none(self):
        app = MagicMock()
        app._snap_params.return_value = {"llm_api_key": ""}
        with patch("app_cluster_workflow.ClusterWorkflowConfig") as mock_cfg, \
             patch("app_cluster_workflow.reject_start") as mock_reject:
            mock_cfg.from_snapshot.side_effect = ValueError("bad config")
            result = build_validated_cluster_snapshot(app)
        assert result is None
        mock_reject.assert_called_once()

    def test_invalid_config_calls_reject_start(self):
        app = MagicMock()
        app._snap_params.return_value = {"llm_api_key": ""}
        with patch("app_cluster_workflow.ClusterWorkflowConfig") as mock_cfg, \
             patch("app_cluster_workflow.reject_start") as mock_reject:
            mock_cfg.from_snapshot.side_effect = RuntimeError("unexpected")
            build_validated_cluster_snapshot(app)
        mock_reject.assert_called_once()


# ---------------------------------------------------------------------------
# MODULE 6: experiment_history_dialog.py — non-tkinter logic
# ---------------------------------------------------------------------------

from experiment_history_dialog import _sort_column, _refresh


class TestSortColumn:
    def _make_tree(self, values: dict):
        """Helper: build a mock Treeview with children whose .set() yields values."""
        tree = MagicMock()
        iids = list(values.keys())
        tree.get_children.return_value = iids
        tree.set.side_effect = lambda iid, col: values[iid]
        return tree

    def test_move_called_once_per_item(self):
        tree = self._make_tree({"id1": "0.75", "id2": "0.90", "id3": "0.65"})
        _sort_column(tree, "macro_f1", reverse=False)
        assert tree.move.call_count == 3

    def test_numeric_sort_ascending(self):
        tree = self._make_tree({"id1": "0.75", "id2": "0.90", "id3": "0.65"})
        _sort_column(tree, "macro_f1", reverse=False)
        # tree.move(iid, parent, index) — index is args[2]
        # After ascending sort: 0.65 (id3), 0.75 (id1), 0.90 (id2)
        moved_iids = [c.args[0] for c in tree.move.call_args_list]
        assert moved_iids == ["id3", "id1", "id2"]

    def test_numeric_sort_moves_all_items(self):
        tree = self._make_tree({"id1": "1.0", "id2": "2.0"})
        _sort_column(tree, "col", reverse=False)
        assert tree.move.call_count == 2

    def test_dash_values_fall_back_to_string_sort(self):
        """Values containing '—' cannot be float-parsed as-is;
        the function replaces '—' with '0' before parsing."""
        tree = self._make_tree({"id1": "—", "id2": "0.50", "id3": "—"})
        # Should not raise; move is still called once per item
        _sort_column(tree, "col", reverse=False)
        assert tree.move.call_count == 3

    def test_heading_rebind_called_after_sort(self):
        """_sort_column must call tree.heading to rebind the command
        with the opposite reverse flag."""
        tree = self._make_tree({"id1": "a", "id2": "b"})
        _sort_column(tree, "col", reverse=False)
        tree.heading.assert_called_once()
        _, kwargs = tree.heading.call_args
        # The new command should be a callable (partial / lambda)
        assert callable(kwargs.get("command") or tree.heading.call_args[0][1]
                        if len(tree.heading.call_args[0]) > 1
                        else tree.heading.call_args.kwargs.get("command"))


class TestRefresh:
    def _make_tree_for_refresh(self):
        tree = MagicMock()
        tree.get_children.return_value = ["existing_id"]
        return tree

    def test_refresh_calls_read_experiments(self):
        tree = self._make_tree_for_refresh()
        with patch("experiment_history_dialog.read_experiments", return_value=[]) as mock_read:
            _refresh(tree, [], ("timestamp",), 10)
        mock_read.assert_called_once_with(last_n=10)

    def test_refresh_deletes_existing_items(self):
        tree = self._make_tree_for_refresh()
        with patch("experiment_history_dialog.read_experiments", return_value=[]):
            _refresh(tree, [], ("timestamp",), 10)
        tree.delete.assert_called_once_with("existing_id")

    def test_refresh_inserts_fresh_records(self):
        tree = self._make_tree_for_refresh()
        records = [
            {
                "timestamp": "2024-01-01T10:00",
                "macro_f1": 0.85,
                "accuracy": 0.88,
                "n_train": 100,
                "n_test": 20,
                "model_file": "/path/to/model.joblib",
                "params": {
                    "train_mode": "tfidf",
                    "C": 1.0,
                    "use_smote": True,
                    "use_lemma": False,
                },
            }
        ]
        with patch("experiment_history_dialog.read_experiments", return_value=records):
            _refresh(tree, [], ("timestamp",), 10)
        tree.insert.assert_called_once()

    def test_refresh_inserts_correct_number_of_records(self):
        tree = self._make_tree_for_refresh()
        records = [
            {
                "timestamp": f"2024-01-0{i}T10:00",
                "macro_f1": 0.8 + i * 0.01,
                "accuracy": 0.85,
                "n_train": 100,
                "n_test": 20,
                "model_file": f"/path/model_{i}.joblib",
                "params": {"train_mode": "tfidf", "C": 1.0,
                           "use_smote": False, "use_lemma": False},
            }
            for i in range(1, 4)
        ]
        with patch("experiment_history_dialog.read_experiments", return_value=records):
            _refresh(tree, [], ("timestamp",), 10)
        assert tree.insert.call_count == len(records)
