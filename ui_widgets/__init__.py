# -*- coding: utf-8 -*-
"""Voilà / Jupyter web-UI widgets for BankReasonTrainer.

Re-exports the top-level entry point so notebooks can do::

    from ui_widgets import build_app
    display(build_app())

The modules in this package assume ``ipywidgets`` is installed — which
is the case when the ``ui`` optional-dependency extra is installed.
See ``docs/JUPYTERHUB_UI.md`` for deployment details.

Additionally re-exports the legacy Tk widgets from the sibling
``ui_widgets.py`` module (``Tooltip``, ``ScrollableFrame`` …) so desktop
code paths that do ``from ui_widgets import Tooltip`` keep working after
the package shadowed the flat module. Loaded via explicit
``spec_from_file_location`` because ``from ui_widgets import …`` would
recurse into the package itself.
"""
from __future__ import annotations

import importlib.util as _importlib_util
import pathlib as _pathlib

from ui_widgets.notebook_app import build_app

__all__ = ["build_app"]

_LEGACY_UI_WIDGETS = (
    _pathlib.Path(__file__).resolve().parent.parent / "ui_widgets.py"
)
_LEGACY_TK_NAMES = (
    "Tooltip",
    "ScrollableFrame",
    "ImageBackground",
    "RoundedButton",
    "PillTabBar",
    "ToggleSwitch",
    "RoundedCard",
    "CollapsibleSection",
)

if _LEGACY_UI_WIDGETS.is_file():
    try:
        _spec = _importlib_util.spec_from_file_location(
            "ui_widgets._legacy_tk", _LEGACY_UI_WIDGETS
        )
        if _spec is not None and _spec.loader is not None:
            _legacy = _importlib_util.module_from_spec(_spec)
            _spec.loader.exec_module(_legacy)
            for _name in _LEGACY_TK_NAMES:
                _obj = getattr(_legacy, _name, None)
                if _obj is not None:
                    globals()[_name] = _obj
    except ImportError:
        # tkinter / PIL / ui_theme absent — leave names unexported so the
        # Voilà-only env stays importable. Desktop code was already broken
        # in this environment.
        pass
