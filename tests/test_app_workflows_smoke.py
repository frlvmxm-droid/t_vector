# -*- coding: utf-8 -*-
"""
Smoke tests for app_*_workflow helpers (train / apply / cluster).

These modules are the UI-free precondition validators and snapshot builders
used by TrainTabMixin / ApplyTabMixin / ClusterTabMixin. They are importable
without a Tk display and lend themselves to mock-based testing.

Coverage:
  * Valid snapshots pass precondition checks.
  * Missing files / columns / model paths trigger ``reject_start`` (returns False).
  * Invalid configs surface a ValueError from the contract layer.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
from unittest.mock import MagicMock


def _needs_mock(name: str) -> bool:
    return _ilu.find_spec(name.split(".")[0]) is None


def _stub(name: str) -> MagicMock:
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = MagicMock()
    return m


for _mod in (
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
    "customtkinter",
    "ui_theme",
    "ui_widgets",
):
    if _mod not in sys.modules and _needs_mock(_mod):
        sys.modules[_mod] = _stub(_mod)


import pytest

from app_train_workflow import (
    build_validated_train_snapshot,
    validate_train_preconditions,
)
from app_apply_workflow import (
    build_validated_apply_snapshot,
    validate_apply_preconditions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_var(value: str):
    var = MagicMock()
    var.get.return_value = value
    return var


def _mock_train_app(
    *,
    train_files=None,
    label_col: str = "label",
    train_mode: str = "tfidf",
    base_model_file: str = "",
    snap=None,
):
    app = MagicMock()
    app.train_files = train_files if train_files is not None else ["/tmp/x.xlsx"]
    app.label_col = _mock_var(label_col)
    app.train_mode = _mock_var(train_mode)
    app.base_model_file = _mock_var(base_model_file)
    app._snap_params.return_value = snap or {
        "train_mode": "tfidf",
        "C": 1.0,
        "max_iter": 1000,
        "test_size": 0.2,
    }
    return app


def _mock_apply_app(
    *,
    model_file: str = "/models/x.joblib",
    apply_file: str = "/data/x.xlsx",
    snap=None,
):
    app = MagicMock()
    app.model_file = _mock_var(model_file)
    app.apply_file = _mock_var(apply_file)
    app._snap_params.return_value = snap or {"pred_col": "cat"}
    return app


# ---------------------------------------------------------------------------
# Train workflow
# ---------------------------------------------------------------------------

class TestTrainWorkflow:
    def test_valid_preconditions(self, monkeypatch):
        monkeypatch.setattr(
            "app_train_workflow.reject_start", lambda *a, **kw: False
        )
        assert validate_train_preconditions(_mock_train_app()) is True

    def test_missing_train_files_rejected(self, monkeypatch):
        called = {}

        def _reject(app, **kw):
            called["reason"] = kw.get("msg")
            return False

        monkeypatch.setattr("app_train_workflow.reject_start", _reject)
        assert validate_train_preconditions(_mock_train_app(train_files=[])) is False
        assert "Excel" in called["reason"]

    def test_missing_label_col_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "app_train_workflow.reject_start", lambda *a, **kw: False
        )
        app = _mock_train_app(label_col="")
        assert validate_train_preconditions(app) is False

    def test_finetune_without_base_model_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "app_train_workflow.reject_start", lambda *a, **kw: False
        )
        app = _mock_train_app(train_mode="finetune", base_model_file="")
        assert validate_train_preconditions(app) is False

    def test_build_valid_snapshot(self, monkeypatch):
        monkeypatch.setattr(
            "app_train_workflow.reject_start", lambda *a, **kw: False
        )
        app = _mock_train_app()
        snap = build_validated_train_snapshot(app)
        assert snap is not None
        assert snap["train_mode"] == "tfidf"

    def test_invalid_snapshot_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "app_train_workflow.reject_start", lambda *a, **kw: False
        )
        app = _mock_train_app(
            snap={"train_mode": "tfidf", "C": -1.0, "max_iter": 100, "test_size": 0.2}
        )
        assert build_validated_train_snapshot(app) is None


# ---------------------------------------------------------------------------
# Apply workflow
# ---------------------------------------------------------------------------

class TestApplyWorkflow:
    def test_valid_preconditions(self, monkeypatch):
        monkeypatch.setattr(
            "app_apply_workflow.reject_start", lambda *a, **kw: False
        )
        assert validate_apply_preconditions(_mock_apply_app()) is True

    def test_missing_model_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "app_apply_workflow.reject_start", lambda *a, **kw: False
        )
        assert validate_apply_preconditions(_mock_apply_app(model_file="")) is False

    def test_missing_xlsx_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "app_apply_workflow.reject_start", lambda *a, **kw: False
        )
        assert validate_apply_preconditions(_mock_apply_app(apply_file="")) is False

    def test_build_snapshot_injects_paths(self, monkeypatch):
        monkeypatch.setattr(
            "app_apply_workflow.reject_start", lambda *a, **kw: False
        )
        snap = build_validated_apply_snapshot(
            _mock_apply_app(
                snap={"pred_col": "category"},
            )
        )
        assert snap is not None
        assert snap["model_file"] == "/models/x.joblib"
        assert snap["apply_file"] == "/data/x.xlsx"
        assert snap["pred_col"] == "category"

    def test_invalid_snapshot_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "app_apply_workflow.reject_start", lambda *a, **kw: False
        )
        # Non-string pred_col → contract rejects → None.
        app = _mock_apply_app(snap={"pred_col": 123})
        assert build_validated_apply_snapshot(app) is None
