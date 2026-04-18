# -*- coding: utf-8 -*-
"""JSON run-report utilities (train/apply/cluster)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


def write_run_report(base_dir: str | Path, pipeline: str, payload: Dict[str, Any]) -> Path:
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{pipeline}_run_{ts}.json"
    data = {
        "pipeline": pipeline,
        "ts_utc": ts,
        **payload,
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def build_unified_run_report(
    *,
    pipeline: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    timings: Mapping[str, Any] | None = None,
    errors: list[Mapping[str, Any]] | None = None,
    status: str = "ok",
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Builds a normalized JSON-ready run-report payload for train/apply/cluster."""
    return {
        "schema_version": 1,
        "pipeline": pipeline,
        "status": status,
        "params": dict(params or {}),
        "metrics": dict(metrics or {}),
        "timings": dict(timings or {}),
        "errors": [dict(e) for e in (errors or [])],
        "metadata": dict(metadata or {}),
    }


def write_unified_run_report(
    base_dir: str | Path,
    *,
    pipeline: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    timings: Mapping[str, Any] | None = None,
    errors: list[Mapping[str, Any]] | None = None,
    status: str = "ok",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    payload = build_unified_run_report(
        pipeline=pipeline,
        params=params,
        metrics=metrics,
        timings=timings,
        errors=errors,
        status=status,
        metadata=metadata,
    )
    return write_run_report(base_dir, pipeline, payload)


def diff_run_reports(before: Dict[str, Any], after: Dict[str, Any], metric_keys: list[str]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for k in metric_keys:
        b = before.get(k)
        a = after.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            base = float(b)
            aft = float(a)
            pct = None if base == 0 else ((aft - base) / abs(base)) * 100.0
            diff[k] = {"before": b, "after": a, "delta": a - b, "delta_pct": pct}
    return diff


def compare_quality_speed(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    quality_higher_better: list[str],
    speed_lower_better: list[str],
    tolerance_pct: float = 5.0,
) -> Dict[str, Any]:
    quality_diff = diff_run_reports(dict(before), dict(after), quality_higher_better)
    speed_diff = diff_run_reports(dict(before), dict(after), speed_lower_better)
    regressions: list[Dict[str, Any]] = []

    for key in quality_higher_better:
        item = quality_diff.get(key)
        if not item:
            continue
        delta_pct = item.get("delta_pct")
        if isinstance(delta_pct, (int, float)) and delta_pct < -abs(float(tolerance_pct)):
            regressions.append({"metric": key, "kind": "quality", "delta_pct": delta_pct})

    for key in speed_lower_better:
        item = speed_diff.get(key)
        if not item:
            continue
        before_v = item["before"]
        after_v = item["after"]
        if isinstance(before_v, (int, float)) and isinstance(after_v, (int, float)):
            if float(before_v) > 0:
                slowdown_pct = ((float(after_v) - float(before_v)) / float(before_v)) * 100.0
                item["slowdown_pct"] = slowdown_pct
                if slowdown_pct > abs(float(tolerance_pct)):
                    regressions.append({"metric": key, "kind": "speed", "delta_pct": slowdown_pct})

    return {
        "tolerance_pct": float(tolerance_pct),
        "quality": quality_diff,
        "speed": speed_diff,
        "regressions": regressions,
        "ok": len(regressions) == 0,
    }


def extract_report_metrics(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Returns numeric-compatible metric dict from both unified and legacy report formats."""
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        return dict(metrics)
    return dict(report)
