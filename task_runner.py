# -*- coding: utf-8 -*-
"""Единый lifecycle long-task и стандарт ошибок для UI-потоков."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ErrorEnvelope:
    """Стандартизованная форма ошибки для UI-лога."""

    error_code: str
    stage: str
    hint: str
    exc_type: str
    message: str
    debug_trace_id: str

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        error_code: str,
        stage: str,
        hint: str,
    ) -> "ErrorEnvelope":
        return cls(
            error_code=error_code,
            stage=stage,
            hint=hint,
            exc_type=type(exc).__name__,
            message=str(exc),
            debug_trace_id=uuid4().hex[:12],
        )

    def format_for_log(self) -> str:
        return (
            f"[error_code={self.error_code}] "
            f"stage={self.stage} "
            f"hint={self.hint} "
            f"trace_id={self.debug_trace_id} | "
            f"{self.exc_type}: {self.message}"
        )


def begin_long_task(
    *,
    cancel_event: Any,
    run_button: Any,
    run_button_busy_text: str,
    stop_button: Any,
    progress_var: Any,
    status_var: Any,
    pct_var: Any,
    phase_var: Any,
    speed_var: Any,
    eta_var: Any,
    start_log: Callable[[str], None],
    start_phase: str,
    clear_summary_var: Optional[Any] = None,
) -> None:
    """Унифицированный старт long-task lifecycle."""
    cancel_event.clear()
    run_button.configure(state="disabled", text=run_button_busy_text)
    stop_button.configure(state="normal")
    progress_var.set(0.0)
    status_var.set("Старт…")
    pct_var.set("0%")
    phase_var.set(start_phase)
    speed_var.set("")
    eta_var.set("")
    if clear_summary_var is not None:
        clear_summary_var.set("")
    start_log("\n==== TASK START ====")


def finalize_long_task(*, owner: Any, run_button: Any, run_button_idle_text: str, stop_button: Any) -> None:
    """Унифицированное завершение long-task lifecycle."""
    with owner._proc_lock:
        owner._processing = False
    run_button.configure(state="normal", text=run_button_idle_text)
    stop_button.configure(state="disabled")


@dataclass
class OperationLifecycle:
    """Хелпер для success/cancel/error веток в long-running UI операциях."""

    owner: Any
    run_button: Any
    run_button_idle_text: str
    stop_button: Any
    log_fn: Callable[[str], None]

    def finalize(self) -> None:
        finalize_long_task(
            owner=self.owner,
            run_button=self.run_button,
            run_button_idle_text=self.run_button_idle_text,
            stop_button=self.stop_button,
        )

    def complete(self) -> None:
        self.finalize()

    def cancelled(self, *, ui_prog: Callable[[float, str], None], log_message: str) -> None:
        ui_prog(0.0, "Отменено")
        self.log_fn(log_message)
        self.finalize()

    def failed(
        self,
        *,
        ui_prog: Callable[[float, str], None],
        status: str,
        envelope: ErrorEnvelope,
        traceback_text: str,
    ) -> None:
        ui_prog(0.0, status)
        self.log_fn(envelope.format_for_log())
        self.log_fn("Traceback:\n" + traceback_text)
        self.finalize()
