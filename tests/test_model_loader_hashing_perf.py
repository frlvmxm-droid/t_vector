import os
import resource
import statistics
import time
from pathlib import Path

import pytest

from model_loader import file_sha256


@pytest.mark.skipif(os.getenv("RUN_PERF_HASH", "0") != "1", reason="set RUN_PERF_HASH=1 for perf run")
def test_model_loader_hashing_perf_large_artifact(tmp_path: Path):
    p = tmp_path / "artifact.joblib"
    size = 1024 * 1024 * 1024  # 1 GiB sparse file
    with p.open("wb") as f:
        f.seek(size - 1)
        f.write(b"\0")

    rounds = max(1, int(os.getenv("RUN_PERF_HASH_ROUNDS", "3")))
    elapsed_runs = []
    rss_peaks = []
    digest = ""
    for _ in range(rounds):
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        digest = file_sha256(p)
        elapsed = time.perf_counter() - t0
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        elapsed_runs.append(elapsed)
        rss_peaks.append(max(0, rss_after - rss_before))

    assert len(digest) == 64
    assert statistics.median(elapsed_runs) < 60
    assert max(rss_peaks) >= 0
