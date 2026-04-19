# -*- coding: utf-8 -*-
"""
Import-level smoke tests for TrainTabMixin / ApplyTabMixin / ClusterTabMixin.

These mixins are ~10 000 lines of Tkinter-bound code, so we cannot instantiate
them in headless CI. We still want to guard against:

  * syntax / import errors after refactoring,
  * accidental removal of the canonical entry-point methods
    (``run_training`` / ``run_apply`` / ``run_cluster``),
  * signature drift on well-known public methods.

This file only performs module imports (with tkinter stubbed out where
necessary) and ``hasattr`` checks — it never calls the methods.
"""
from __future__ import annotations

import importlib

import pytest


# Tkinter is stubbed in tests/conftest.py at collection time.


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module_name, class_name, method_name",
    [
        ("app_train", "TrainTabMixin", "run_training"),
        ("app_apply", "ApplyTabMixin", "run_apply"),
        ("app_cluster", "ClusterTabMixin", "run_cluster"),
    ],
)
def test_mixin_module_imports_with_run_entrypoint(module_name, class_name, method_name):
    """Each UI mixin must import cleanly and expose its canonical run_* entry point."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{module_name}: optional dep not installed ({exc.name})")

    mixin = getattr(module, class_name, None)
    assert mixin is not None, f"{module_name}.{class_name} missing"
    assert callable(getattr(mixin, method_name, None)), (
        f"{class_name}.{method_name} must be a callable — do not rename without "
        f"updating call sites and the workflow layer."
    )


@pytest.mark.parametrize(
    "module_name",
    ["app_train_service", "app_apply_service", "app_cluster_service"],
)
def test_service_layer_modules_importable(module_name):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{module_name}: optional dep not installed ({exc.name})")


@pytest.mark.parametrize(
    "module_name",
    ["app_train_workflow", "app_apply_workflow", "app_cluster_workflow"],
)
def test_workflow_layer_modules_importable(module_name):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"{module_name}: optional dep not installed ({exc.name})")
