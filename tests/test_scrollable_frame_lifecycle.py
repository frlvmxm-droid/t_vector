"""Regression test for ``ScrollableFrame`` widget lifecycle.

The Tk ``ScrollableFrame`` lives in ``ui_widgets_tk.py`` (the legacy
flat Tk module, renamed from ``ui_widgets.py`` after the Voilà web-UI
port converted ``ui_widgets/`` into a package). This test imports it
directly — no ``spec_from_file_location`` needed now that there is no
name shadowing.
"""
from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import pytest


def _load_scrollable_frame():
    try:
        import ui_widgets_tk  # noqa: F401 — ensure module importable
    except ImportError as exc:
        pytest.skip(f"ui_widgets_tk dependencies missing: {exc}")
    cls = getattr(ui_widgets_tk, "ScrollableFrame", None)
    if cls is None:
        pytest.skip("ScrollableFrame not exported by ui_widgets_tk")
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
