"""Regression test for ``ScrollableFrame`` widget lifecycle.

``ui_widgets`` was converted from a flat module to a package during the
Voilà web-UI port (``ui_widgets/`` now shadows ``ui_widgets.py``). The
Tk ``ScrollableFrame`` still lives in the legacy module file; this test
imports it via ``importlib`` so both layouts remain testable.
"""
from __future__ import annotations

import importlib.util
import pathlib
from unittest.mock import MagicMock

import pytest


def _load_scrollable_frame():
    legacy_path = pathlib.Path(__file__).resolve().parent.parent / "ui_widgets.py"
    if not legacy_path.is_file():
        pytest.skip("Legacy ui_widgets.py not present in this checkout")
    spec = importlib.util.spec_from_file_location("ui_widgets_legacy", legacy_path)
    if spec is None or spec.loader is None:
        pytest.skip("Could not build spec for legacy ui_widgets.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.skip(f"Legacy ui_widgets.py dependencies missing: {exc}")
    cls = getattr(module, "ScrollableFrame", None)
    if cls is None:
        pytest.skip("ScrollableFrame not exported by legacy ui_widgets.py")
    return cls


def test_scrollable_frame_create_destroy_recreate() -> None:
    if importlib.util.find_spec("tkinter") is None:
        pytest.skip("tkinter not installed — ScrollableFrame lifecycle cannot be exercised")
    import tkinter as tk

    if isinstance(tk, MagicMock):
        pytest.skip("tkinter is mocked — headless lifecycle check not meaningful")

    ScrollableFrame = _load_scrollable_frame()
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk not available in test environment")

    try:
        root.withdraw()
        f1 = ScrollableFrame(root)
        f1.pack()
        root.update_idletasks()
        f1.destroy()
        root.update_idletasks()

        f2 = ScrollableFrame(root)
        f2.pack()
        root.update_idletasks()
        f2.destroy()
        root.update_idletasks()
    finally:
        root.destroy()
