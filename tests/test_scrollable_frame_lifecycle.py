import importlib.util
from unittest.mock import MagicMock

import tkinter as tk

import pytest

from ui_widgets import ScrollableFrame


def test_scrollable_frame_create_destroy_recreate():
    if importlib.util.find_spec("tkinter") is None or isinstance(tk, MagicMock):
        pytest.skip("tkinter not installed — ScrollableFrame lifecycle cannot be exercised")
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
