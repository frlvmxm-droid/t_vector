# -*- coding: utf-8 -*-
"""Reusable progress-panel widget: bar + phase label + rolling log."""
from __future__ import annotations

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
            w.HBox([self.bar, self.pct]),
            self.phase,
            w.HTML("<b>Лог:</b>"),
            self.output,
        ])

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

    def mark_error(self, msg: str = "Ошибка ❌") -> None:
        self.bar.bar_style = "danger"
        self.phase.value = msg
