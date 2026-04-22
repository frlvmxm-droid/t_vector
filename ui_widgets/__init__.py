# -*- coding: utf-8 -*-
"""Voilà / Jupyter web-UI widgets for BankReasonTrainer.

Re-exports the top-level entry point so notebooks can do::

    from ui_widgets import build_app
    display(build_app())

The modules in this package assume ``ipywidgets`` is installed — which
is the case when the ``ui`` optional-dependency extra is installed.
See ``docs/JUPYTERHUB_UI.md`` for deployment details.

Legacy Tk widgets (``Tooltip``, ``ScrollableFrame``, …) live in
``ui_widgets_tk.py`` as a sibling top-level module. Desktop callers
should ``from ui_widgets_tk import Tooltip`` directly.
"""
from __future__ import annotations

from ui_widgets.notebook_app import build_app

__all__ = ["build_app"]
