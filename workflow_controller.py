# -*- coding: utf-8 -*-
"""Тонкий controller-слой для обновления прогресса долгих операций."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkflowProgressController:
    progress_var: Any
    status_var: Any
    pct_var: Any
    phase_var: Any
    speed_var: Any
    eta_var: Any

    def update(self, pct: float, status: str) -> None:
        clamped_pct = max(0.0, min(100.0, pct))
        self.progress_var.set(clamped_pct)
        self.status_var.set(status)
        self.pct_var.set(f"{clamped_pct:.0f}%")
        parts = [p.strip() for p in status.split("|")]
        self.phase_var.set(parts[0] if parts else "")
        self.speed_var.set(parts[1] if len(parts) > 1 else "")
        self.eta_var.set(parts[2] if len(parts) > 2 else "")
