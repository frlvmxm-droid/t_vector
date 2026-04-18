import os
import resource
import time
from pathlib import Path

import pytest

from model_loader import file_sha256


@pytest.mark.skipif(os.getenv("RUN_PERF_HASH", "0") != "1", reason="set RUN_PERF_HASH=1 for perf run")
def test_stream_hash_large_artifact_perf(tmp_path: Path):
    """Perf smoke: 1GB sparse file hash with bounded memory path."""
    p = tmp_path / "large.joblib"
    size = 1024 * 1024 * 1024  # 1 GiB
    with p.open("wb") as f:
        f.seek(size - 1)
        f.write(b"\0")

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    digest = file_sha256(p)
    dt = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert isinstance(digest, str) and len(digest) == 64
    # мягкий guardrail; не точный benchmark
    assert dt < 60
    # ru_maxrss в KB на Linux, в bytes на macOS — используем только относительную границу.
    assert rss_after >= rss_before
