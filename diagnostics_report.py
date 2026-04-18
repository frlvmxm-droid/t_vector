# -*- coding: utf-8 -*-
"""Diagnostic mode report helpers."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json


@dataclass(frozen=True)
class DiagnosticReport:
    created_at_utc: str
    correlation_id: str
    event: str
    stage: str
    metrics: dict[str, Any]
    snapshot: dict[str, Any]
    error_class: str | None = None
    error_message: str | None = None


SENSITIVE_KEYS = {"api_key", "token", "password", "secret"}


def _sanitize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in snapshot.items():
        kl = str(k).lower()
        if any(s in kl for s in SENSITIVE_KEYS):
            out[str(k)] = "***"
        else:
            out[str(k)] = v
    return out


def build_diagnostic_report(
    *,
    correlation_id: str,
    event: str,
    stage: str,
    metrics: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    error: BaseException | None = None,
) -> DiagnosticReport:
    return DiagnosticReport(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        correlation_id=correlation_id,
        event=event,
        stage=stage,
        metrics=dict(metrics),
        snapshot=_sanitize_snapshot(snapshot),
        error_class=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
    )


def export_diagnostic_report(report: DiagnosticReport, out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)
