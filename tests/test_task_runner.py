from task_runner import ErrorEnvelope, OperationLifecycle


class _Btn:
    def __init__(self):
        self.state = None
        self.text = None

    def configure(self, **kwargs):
        self.state = kwargs.get("state", self.state)
        self.text = kwargs.get("text", self.text)


class _Owner:
    def __init__(self):
        self._processing = True


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
