# -*- coding: utf-8 -*-
"""Общие guard/helper функции для workflow preconditions."""
from __future__ import annotations

from tkinter import messagebox


def reject_start(app, *, title: str, msg: str, kind: str = "warning") -> bool:
    """Единый отказ запуска workflow с корректным reset processing-флага."""
    with app._proc_lock:
        app._processing = False
    if kind == "error":
        messagebox.showerror(title, msg)
    else:
        messagebox.showwarning(title, msg)
    return False
