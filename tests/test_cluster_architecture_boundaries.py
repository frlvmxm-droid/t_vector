from pathlib import Path

from sklearn.cluster import MiniBatchKMeans

from cluster_algo_strategy import gpu_kmeans
from cluster_runtime_service import (
    try_mark_processing,
    clear_processing,
    tune_cluster_runtime_for_input,
)


def test_cluster_runtime_processing_guard():
    class _Owner:
        def __init__(self):
            import threading

            self._proc_lock = threading.Lock()
            self._processing = False

    owner = _Owner()
    assert try_mark_processing(owner) is True
    assert try_mark_processing(owner) is False
    clear_processing(owner)
    assert try_mark_processing(owner) is True


def test_cluster_runtime_tune_updates_snapshot(tmp_path: Path):
    f = tmp_path / "in.xlsx"
    f.write_bytes(b"x" * 1024)

    class _HW:
        sbert_batch = 64

    snap = {"streaming_chunk_size": 4000, "sbert_batch": 64, "kmeans_batch": 2048}
    out = tune_cluster_runtime_for_input(
        files_snapshot=[str(f)],
        snap=snap,
        hw=_HW(),
        log_fn=lambda _msg: None,
    )
    assert "chunk" in out
    assert snap["streaming_chunk_size"] == out["chunk"]


def test_cluster_algo_strategy_fallbacks_to_cpu_when_no_cuml(monkeypatch):
    monkeypatch.setattr("cluster_algo_strategy.cuml_kmeans_available", lambda: False)
    km = gpu_kmeans(n_clusters=3, random_state=42)
    assert isinstance(km, MiniBatchKMeans)


def test_app_cluster_uses_runtime_and_strategy_services():
    src = Path("app_cluster.py").read_text(encoding="utf-8")
    assert "from cluster_algo_strategy import (" in src
    assert "from cluster_runtime_service import (" in src
