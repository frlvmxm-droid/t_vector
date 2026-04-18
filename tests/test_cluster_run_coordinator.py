from types import SimpleNamespace

import cluster_run_coordinator


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_prepare_cluster_run_context_adjusts_ensemble_algo(monkeypatch):
    app = SimpleNamespace(
        _proc_lock=_Lock(),
        _processing=True,
        cluster_files=["a.xlsx"],
        cluster_algo=SimpleNamespace(set=lambda _v: None),
        log_cluster=lambda _m: None,
        txt_anchors=None,
    )
    monkeypatch.setattr(cluster_run_coordinator, "validate_cluster_preconditions", lambda _a: True)
    monkeypatch.setattr(
        cluster_run_coordinator,
        "build_cluster_runtime_snapshot",
        lambda _a: {"cluster_vec_mode": "ensemble", "cluster_algo": "hdbscan"},
    )

    snap, files = cluster_run_coordinator.prepare_cluster_run_context(app)
    assert snap["cluster_algo"] == "kmeans"
    assert files == ["a.xlsx"]
