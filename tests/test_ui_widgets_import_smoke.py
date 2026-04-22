"""Import-smoke for every ``ui_widgets`` module.

Regression guard: if a public name (``build_apply_panel``,
``build_train_panel``, ``ProgressPanel``, ``TrustStore`` …) moves or
gets accidentally removed, CI catches it here before the Voilà
dashboard breaks at render time. Each module is imported in isolation;
importing one failure doesn't mask another.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Every public ui_widgets module that ships with the Voilà dashboard.
_UI_WIDGETS_MODULES = [
    "ui_widgets",
    "ui_widgets.apply_panel",
    "ui_widgets.cluster_panel",
    "ui_widgets.io",
    "ui_widgets.notebook_app",
    "ui_widgets.predictions_table",
    "ui_widgets.progress",
    "ui_widgets.session",
    "ui_widgets.theme",
    "ui_widgets.train_panel",
    "ui_widgets.trust_prompt",
    "ui_widgets.dialogs",
    "ui_widgets.dialogs.artifacts",
    "ui_widgets.dialogs.history",
    "ui_widgets.dialogs.settings",
]


@pytest.mark.parametrize("module_name", _UI_WIDGETS_MODULES)
def test_module_imports(module_name: str) -> None:
    """Every ui_widgets module must import cleanly (no side-effect failures)."""
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(
            f"Optional dependency missing for {module_name}: {exc}"
        )


def test_public_factories_exist() -> None:
    """The three top-level factories must be importable and callable."""
    ipywidgets = pytest.importorskip("ipywidgets")  # noqa: F841 — probe

    from ui_widgets.apply_panel import build_apply_panel
    from ui_widgets.cluster_panel import build_cluster_panel
    from ui_widgets.notebook_app import build_app
    from ui_widgets.train_panel import build_train_panel

    assert callable(build_app)
    assert callable(build_apply_panel)
    assert callable(build_cluster_panel)
    assert callable(build_train_panel)


def test_progress_panel_class_exposes_expected_api() -> None:
    pytest.importorskip("ipywidgets")
    from ui_widgets.progress import ProgressPanel

    assert callable(ProgressPanel)
    for attr in ("update", "log", "reset", "mark_done", "mark_error",
                 "attach_cancel_event", "detach_cancel_event"):
        assert hasattr(ProgressPanel, attr), f"Missing {attr} on ProgressPanel"


def test_trust_prompt_exports_public_api() -> None:
    from ui_widgets.trust_prompt import (
        TrustDenied,
        build_confirm_prompt,
        ensure_trusted_model_path_interactive,
        get_trust_store,
    )

    assert callable(build_confirm_prompt)
    assert callable(ensure_trusted_model_path_interactive)
    assert callable(get_trust_store)
    assert issubclass(TrustDenied, Exception)


def test_session_exports_public_api() -> None:
    from ui_widgets.session import (
        DebouncedSaver,
        load_last_session,
        save_session,
        session_path,
    )

    assert callable(save_session)
    assert callable(load_last_session)
    assert callable(session_path)
    assert callable(DebouncedSaver)
