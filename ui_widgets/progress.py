"""Reusable progress-panel widget: bar + phase label + rolling log.

Phase 14: exposes an optional cancel button. Panels that pass a
``threading.Event`` via :meth:`attach_cancel_event` get a red ⏹
cancel button; clicking it calls ``event.set()`` so the service
layer breaks on the next ``_check_cancelled`` call.
"""
from __future__ import annotations

import threading
from typing import Any


class ProgressPanel:
    """Progress bar + phase label + scrollable log ``Output``.

    Suitable for passing ``self.update`` as a service-layer
    ``progress_cb(frac: float, msg: str)`` and ``self.log`` as a
    ``log_cb(line: str)``. Safe to call from a worker thread —
    ipywidgets pushes ``.value`` changes to the browser via COMM.
    """

    def __init__(self, *, log_height: str = "200px") -> None:
        import ipywidgets as w

        self._w = w
        self.bar = w.FloatProgress(
            value=0.0, min=0.0, max=100.0,
            bar_style="info",
            layout=w.Layout(width="60%"),
        )
        self.pct = w.Label(value="0%")
        self.phase = w.Label(value="—")
        self.cancel_btn = w.Button(
            description="⏹ Отмена",
            button_style="danger",
            disabled=True,
            layout=w.Layout(width="110px", display="none"),
        )
        self._cancel_event: threading.Event | None = None
        self.cancel_btn.on_click(self._on_cancel_click)
        self.output = w.Output(
            layout=w.Layout(
                height=log_height,
                overflow="auto",
                border="1px solid #ddd",
                padding="4px",
            ),
        )

    @property
    def widget(self) -> Any:
        w = self._w
        return w.VBox([
            w.HBox([self.bar, self.pct, self.cancel_btn]),
            self.phase,
            w.HTML("<b>Лог:</b>"),
            self.output,
        ])

    def attach_cancel_event(self, event: threading.Event) -> None:
        """Wire the cancel button to ``event.set()`` and make it visible."""
        self._cancel_event = event
        self.cancel_btn.disabled = False
        self.cancel_btn.layout.display = ""
        self.cancel_btn.description = "⏹ Отмена"

    def detach_cancel_event(self) -> None:
        """Hide the cancel button (call when the worker finishes)."""
        self._cancel_event = None
        self.cancel_btn.disabled = True
        self.cancel_btn.layout.display = "none"

    def _on_cancel_click(self, _btn: Any) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.cancel_btn.disabled = True
            self.cancel_btn.description = "⏹ Отмена запрошена…"

    def update(self, frac: float, msg: str = "") -> None:
        """Set progress ``0.0..1.0`` and (optionally) phase text."""
        frac = max(0.0, min(1.0, float(frac)))
        self.bar.value = frac * 100.0
        self.pct.value = f"{frac * 100:.0f}%"
        if msg:
            self.phase.value = msg

    def log(self, line: str) -> None:
        """Append a line to the rolling log output."""
        with self.output:
            print(line)

    def reset(self, phase: str = "—") -> None:
        self.bar.value = 0.0
        self.bar.bar_style = "info"
        self.pct.value = "0%"
        self.phase.value = phase
        self.output.clear_output()

    def mark_done(self, msg: str = "Готово ✅") -> None:
        self.bar.value = 100.0
        self.bar.bar_style = "success"
        self.pct.value = "100%"
        self.phase.value = msg
        self.detach_cancel_event()

    def mark_error(self, msg: str = "Ошибка ❌") -> None:
        self.bar.bar_style = "danger"
        self.phase.value = msg
        self.detach_cancel_event()
