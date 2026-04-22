"""Read-only context dialogs (History / Artifacts / Settings).

Each ``build_*_dialog()`` returns an ``ipywidgets.VBox`` rendered inside
an overlay card in the main area; the wrapping ``Stack`` in
``notebook_app.build_app()`` handles show/hide.
"""
from __future__ import annotations

from ui_widgets.dialogs.artifacts import build_artifacts_dialog
from ui_widgets.dialogs.history import build_history_dialog
from ui_widgets.dialogs.settings import build_settings_dialog

__all__ = [
    "build_artifacts_dialog",
    "build_history_dialog",
    "build_settings_dialog",
]
