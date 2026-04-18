from diagnostics_report import build_diagnostic_report, export_diagnostic_report


def test_build_diagnostic_report_masks_sensitive_fields(tmp_path):
    rep = build_diagnostic_report(
        correlation_id="cid-1",
        event="train.run",
        stage="vectorize",
        metrics={"rows": 12},
        snapshot={"api_key": "secret", "model": "x"},
    )
    assert rep.snapshot["api_key"] == "***"
    assert rep.snapshot["model"] == "x"

    out = export_diagnostic_report(rep, str(tmp_path / "diag.json"))
    assert out.endswith("diag.json")
