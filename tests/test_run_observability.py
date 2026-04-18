from pathlib import Path

from run_observability import diff_reports_from_files


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
