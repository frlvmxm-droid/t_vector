# -*- coding: utf-8 -*-
"""Unified run observability helpers + diff CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from run_report import compare_quality_speed, diff_run_reports, extract_report_metrics


def load_report(path: str | Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Не удалось загрузить отчёт {path}: {exc}") from exc


def diff_reports_from_files(before_path: str | Path, after_path: str | Path, keys: list[str]) -> Dict[str, Any]:
    before = load_report(before_path)
    after = load_report(after_path)
    return diff_run_reports(extract_report_metrics(before), extract_report_metrics(after), keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--keys", default="macro_f1,throughput,noise_ratio")
    ap.add_argument("--quality-keys", default="")
    ap.add_argument("--speed-keys", default="")
    ap.add_argument("--tolerance-pct", type=float, default=5.0)
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    before = load_report(args.before)
    after = load_report(args.after)
    before_metrics = extract_report_metrics(before)
    after_metrics = extract_report_metrics(after)
    diff = diff_run_reports(before_metrics, after_metrics, keys)
    quality_keys = [k.strip() for k in args.quality_keys.split(",") if k.strip()]
    speed_keys = [k.strip() for k in args.speed_keys.split(",") if k.strip()]
    if quality_keys or speed_keys:
        envelope = compare_quality_speed(
            before_metrics,
            after_metrics,
            quality_higher_better=quality_keys,
            speed_lower_better=speed_keys,
            tolerance_pct=float(args.tolerance_pct),
        )
        print(json.dumps({"diff": diff, "comparison": envelope}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(diff, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
