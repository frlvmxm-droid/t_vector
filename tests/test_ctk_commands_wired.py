"""Smoke-tests that the 5 CTk view buttons reference real methods on App.

Uses AST parsing (no display/tkinter needed) as a regression-guard against
silent button→method breakage discovered in the 2026-04 UI audit.
"""
from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _method_defined_in(filename: str, method_name: str) -> bool:
    src = (_ROOT / filename).read_text(encoding="utf-8")
    tree = ast.parse(src)
    return any(
        isinstance(n, ast.FunctionDef) and n.name == method_name
        for n in ast.walk(tree)
    )


def test_add_train_folder_exists():
    assert _method_defined_in("app_train.py", "add_train_folder"), (
        "Train tab «Добавить папку» button references missing TrainTabMixin.add_train_folder"
    )


def test_remove_train_file_exists():
    assert _method_defined_in("app_train.py", "remove_train_file"), (
        "Train tab «Удалить» button references missing TrainTabMixin.remove_train_file"
    )


def test_export_predictions_exists():
    assert _method_defined_in("app_apply.py", "export_predictions"), (
        "Apply tab «Экспорт» button references missing ApplyTabMixin.export_predictions"
    )


def test_export_cluster_results_exists():
    assert _method_defined_in("app_cluster.py", "export_cluster_results"), (
        "Cluster tab «Экспорт» button references missing ClusterTabMixin.export_cluster_results"
    )


def test_check_and_install_deps_exists():
    assert _method_defined_in("app_deps.py", "_check_and_install_deps"), (
        "Deps tab «Проверить и установить» button references missing "
        "DepsTabMixin._check_and_install_deps"
    )
