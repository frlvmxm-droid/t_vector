import tkinter as tk

import pytest

from ui_widgets import ScrollableFrame


def test_scrollable_frame_create_destroy_recreate():
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
