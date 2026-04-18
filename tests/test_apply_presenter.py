from ui.tabs.apply_presenter import format_apply_autoprofile_log


def test_format_apply_autoprofile_log_contains_expected_fields():
    msg = format_apply_autoprofile_log({"input_size_gb": 1.5, "chunk": 3200, "sbert_batch": 48})
    assert "input=1.5 GB" in msg
    assert "chunk=3200" in msg
    assert "sbert_batch=48" in msg


def test_format_apply_autoprofile_log_handles_missing_keys():
    msg = format_apply_autoprofile_log({})
    assert "input=? GB" in msg
    assert "chunk=?" in msg
    assert "sbert_batch=?" in msg
