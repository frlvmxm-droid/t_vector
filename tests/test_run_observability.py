import json
import sys
from pathlib import Path

import pytest

from run_observability import diff_reports_from_files, load_report, main


def test_diff_reports_from_files(tmp_path: Path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text('{"macro_f1": 0.7, "throughput": 100}', encoding="utf-8")
    after.write_text('{"macro_f1": 0.8, "throughput": 90}', encoding="utf-8")
    diff = diff_reports_from_files(before, after, ["macro_f1", "throughput"])
    assert diff["macro_f1"]["delta"] > 0
    assert diff["throughput"]["delta"] < 0


def test_diff_reports_from_files_unified_metrics(tmp_path: Path):
    before = tmp_path / "before_unified.json"
    after = tmp_path / "after_unified.json"
    before.write_text('{"metrics":{"macro_f1": 0.7, "throughput": 100}}', encoding="utf-8")
    after.write_text('{"metrics":{"macro_f1": 0.75, "throughput": 105}}', encoding="utf-8")
    diff = diff_reports_from_files(before, after, ["macro_f1", "throughput"])
    assert diff["macro_f1"]["delta"] > 0
    assert diff["throughput"]["delta"] > 0


def test_load_report_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"macro_f1": 0.82, "throughput": 1200}), encoding="utf-8")
    assert load_report(p) == {"macro_f1": 0.82, "throughput": 1200}


def test_load_report_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Не удалось загрузить"):
        load_report(tmp_path / "nope.json")


def test_load_report_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Не удалось загрузить"):
        load_report(p)


def test_main_diff_only(tmp_path: Path, capsys, monkeypatch) -> None:
    before = tmp_path / "b.json"
    before.write_text(json.dumps({"macro_f1": 0.80}), encoding="utf-8")
    after = tmp_path / "a.json"
    after.write_text(json.dumps({"macro_f1": 0.85}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["run_observability", str(before), str(after),
                         "--keys", "macro_f1"])
    rc = main()
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "macro_f1" in payload


def test_main_with_quality_speed_envelope(tmp_path: Path, capsys, monkeypatch) -> None:
    before = tmp_path / "b.json"
    before.write_text(json.dumps({"macro_f1": 0.80, "latency_ms": 50}), encoding="utf-8")
    after = tmp_path / "a.json"
    after.write_text(json.dumps({"macro_f1": 0.85, "latency_ms": 45}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_observability", str(before), str(after),
        "--keys", "macro_f1,latency_ms",
        "--quality-keys", "macro_f1",
        "--speed-keys", "latency_ms",
        "--tolerance-pct", "5.0",
    ])
    rc = main()
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "diff" in payload
    assert "comparison" in payload
