import pytest

pytest.importorskip("tools.perf_smoke", reason="tools/ not on PYTHONPATH in this environment")

from tools.perf_smoke import run_perf  # noqa: E402


def test_perf_smoke_runner_outputs_required_keys():
    out = run_perf(rounds=1)
    assert out["files_count"] == 500
    assert out["prepare_inputs_sec_median"] >= 0
    assert out["build_t5_source_text_sec_median"] >= 0
    assert out["peak_memory_bytes_max"] > 0
    assert out["rounds"] == 1
