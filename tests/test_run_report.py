from pathlib import Path

import pytest

from run_report import (
    write_run_report,
    diff_run_reports,
    compare_quality_speed,
    build_unified_run_report,
    extract_report_metrics,
    write_unified_run_report,
)


def test_write_run_report_creates_json(tmp_path: Path):
    p = write_run_report(tmp_path, "apply", {"macro_f1": 0.81, "throughput": 120.0})
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert '"pipeline": "apply"' in txt


def test_diff_run_reports_numeric_metrics_only():
    before = {"macro_f1": 0.7, "throughput": 100.0, "note": "old"}
    after = {"macro_f1": 0.8, "throughput": 90.0, "note": "new"}
    diff = diff_run_reports(before, after, ["macro_f1", "throughput", "note"])
    assert diff["macro_f1"]["delta"] == pytest.approx(0.1)
    assert diff["macro_f1"]["delta_pct"] == pytest.approx((0.1 / 0.7) * 100)
    assert diff["throughput"]["delta"] == pytest.approx(-10.0)
    assert "note" not in diff


def test_compare_quality_speed_flags_regression():
    before = {"macro_f1": 0.80, "latency_sec": 1.0}
    after = {"macro_f1": 0.72, "latency_sec": 1.2}
    out = compare_quality_speed(
        before,
        after,
        quality_higher_better=["macro_f1"],
        speed_lower_better=["latency_sec"],
        tolerance_pct=5.0,
    )
    assert out["ok"] is False
    assert len(out["regressions"]) == 2


def test_build_unified_run_report_structure():
    payload = build_unified_run_report(
        pipeline="train",
        params={"chunk": 2000},
        metrics={"macro_f1": 0.81},
        timings={"fit_sec": 12.5},
        errors=[{"error_code": "X", "message": "oops"}],
        status="warning",
    )
    assert payload["schema_version"] == 1
    assert payload["pipeline"] == "train"
    assert payload["params"]["chunk"] == 2000
    assert payload["metrics"]["macro_f1"] == pytest.approx(0.81)
    assert payload["timings"]["fit_sec"] == pytest.approx(12.5)
    assert payload["errors"][0]["error_code"] == "X"


def test_extract_report_metrics_supports_unified_format():
    report = {"metrics": {"throughput": 120.0}, "params": {"chunk": 1000}}
    out = extract_report_metrics(report)
    assert out["throughput"] == pytest.approx(120.0)


def test_write_unified_run_report_for_all_pipelines(tmp_path: Path):
    for pipeline in ("train", "apply", "cluster"):
        p = write_unified_run_report(
            tmp_path,
            pipeline=pipeline,
            params={"chunk": 1024},
            metrics={"macro_f1": 0.8},
            timings={"total_sec": 3.2},
            status="ok",
        )
        txt = p.read_text(encoding="utf-8")
        assert f'"pipeline": "{pipeline}"' in txt
        assert '"schema_version": 1' in txt
