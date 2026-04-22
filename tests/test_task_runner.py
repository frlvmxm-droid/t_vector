import threading

import pytest

from task_runner import (
    ErrorEnvelope,
    LongTaskUIContext,
    OperationLifecycle,
    prepare_long_task_ui,
)


class _Btn:
    def __init__(self):
        self.state = None
        self.text = None

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)
        self.text = kwargs.get("text", self.text)


class _Var:
    def __init__(self, value=None):
        self._value = value

    def set(self, value):
        self._value = value

    def get(self):
        return self._value


class _Owner:
    def __init__(self):
        self._processing = True
        self._proc_lock = threading.Lock()


def test_operation_lifecycle_failed_sets_ui_and_logs():
    logs = []
    owner = _Owner()
    run_btn = _Btn()
    stop_btn = _Btn()
    progress = {"value": None, "status": None}

    def _ui_prog(pct, status):
        progress["value"] = pct
        progress["status"] = status

    life = OperationLifecycle(
        owner=owner,
        run_button=run_btn,
        run_button_idle_text="▶ run",
        stop_button=stop_btn,
        log_fn=logs.append,
    )
    env = ErrorEnvelope.from_exception(
        RuntimeError("boom"),
        error_code="X",
        stage="stage",
        hint="hint",
    )
    life.failed(ui_prog=_ui_prog, status="Ошибка", envelope=env, traceback_text="tb")

    assert owner._processing is False
    assert run_btn.state == "normal"
    assert run_btn.text == "▶ run"
    assert stop_btn.state == "disabled"
    assert progress["status"] == "Ошибка"
    assert any("trace_id=" in m for m in logs)
    assert any("Traceback:\ntb" == m for m in logs)


# ---------------------------------------------------------------------------
# prepare_long_task_ui — bundled factory for (controller, ui_prog, lifecycle)
# ---------------------------------------------------------------------------


@pytest.fixture
def _ui_kit():
    """Набор Tk-stub-переменных + кнопок для стадии подготовки."""
    return {
        "progress_var": _Var(0.0),
        "status_var": _Var(""),
        "pct_var": _Var(""),
        "phase_var": _Var(""),
        "speed_var": _Var(""),
        "eta_var": _Var(""),
        "run_button": _Btn(),
        "stop_button": _Btn(),
    }


def test_prepare_long_task_ui_returns_context_bundle(_ui_kit):
    owner = _Owner()
    ctx = prepare_long_task_ui(
        owner=owner,
        log_fn=lambda _m: None,
        run_button_idle_text="▶ run",
        **_ui_kit,
    )
    assert isinstance(ctx, LongTaskUIContext)
    assert ctx.controller is not None
    assert callable(ctx.ui_prog)
    assert isinstance(ctx.lifecycle, OperationLifecycle)


def test_prepare_long_task_ui_ui_prog_updates_controller(_ui_kit):
    owner = _Owner()
    ctx = prepare_long_task_ui(
        owner=owner,
        log_fn=lambda _m: None,
        run_button_idle_text="▶ run",
        **_ui_kit,
    )
    ctx.ui_prog(42.5, "Фаза 1")
    assert _ui_kit["progress_var"].get() == pytest.approx(42.5)
    assert _ui_kit["status_var"].get() == "Фаза 1"


def test_prepare_long_task_ui_ui_prog_clamps_percent(_ui_kit):
    owner = _Owner()
    ctx = prepare_long_task_ui(
        owner=owner,
        log_fn=lambda _m: None,
        run_button_idle_text="▶ run",
        **_ui_kit,
    )
    ctx.ui_prog(150.0, "over")
    assert _ui_kit["progress_var"].get() == pytest.approx(100.0)
    ctx.ui_prog(-10.0, "under")
    assert _ui_kit["progress_var"].get() == pytest.approx(0.0)


def test_prepare_long_task_ui_lifecycle_complete_releases_owner(_ui_kit):
    owner = _Owner()
    assert owner._processing is True
    ctx = prepare_long_task_ui(
        owner=owner,
        log_fn=lambda _m: None,
        run_button_idle_text="▶ run",
        **_ui_kit,
    )
    ctx.lifecycle.complete()
    assert owner._processing is False
    assert _ui_kit["run_button"].state == "normal"
    assert _ui_kit["run_button"].text == "▶ run"
    assert _ui_kit["stop_button"].state == "disabled"


def test_prepare_long_task_ui_lifecycle_cancelled_logs_and_resets(_ui_kit):
    logs = []
    owner = _Owner()
    ctx = prepare_long_task_ui(
        owner=owner,
        log_fn=logs.append,
        run_button_idle_text="▶ run",
        **_ui_kit,
    )
    ctx.lifecycle.cancelled(ui_prog=ctx.ui_prog, log_message="Операция отменена")
    assert owner._processing is False
    assert _ui_kit["status_var"].get() == "Отменено"
    assert logs == ["Операция отменена"]


def test_prepare_long_task_ui_is_context_immutable(_ui_kit):
    owner = _Owner()
    ctx = prepare_long_task_ui(
        owner=owner,
        log_fn=lambda _m: None,
        run_button_idle_text="▶ run",
        **_ui_kit,
    )
    with pytest.raises(AttributeError):
        ctx.controller = None  # type: ignore[misc]
