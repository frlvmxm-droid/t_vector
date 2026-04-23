"""Tests for ``cluster_algo_strategy.gpu_kmeans`` — CPU/GPU branch selection.

The factory used to live at ``app_cluster._gpu_kmeans`` (pre-10.0.0);
after Sprint 3 of the web migration the helper was extracted to
``cluster_algo_strategy`` and renamed ``gpu_kmeans`` (no underscore).
These tests exercise both the CPU fallback and the cuML-available
branch without pulling in a real ``cuml`` install.
"""
import cluster_algo_strategy


def test_gpu_kmeans_cpu_branch_respects_params(monkeypatch):
    monkeypatch.setattr(cluster_algo_strategy, "_CUML_KMEANS", False)
    km = cluster_algo_strategy.gpu_kmeans(
        n_clusters=7,
        random_state=123,
        n_init=11,
        init="random",
        max_iter=77,
    )
    assert km.n_clusters == 7
    assert km.random_state == 123
    assert km.n_init == 11
    assert km.init == "random"
    assert km.max_iter == 77


def test_gpu_kmeans_gpu_branch_passes_core_kwargs(monkeypatch):
    captured = {}

    class _FakeKMeans:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeCluster:
        KMeans = _FakeKMeans

    monkeypatch.setattr(cluster_algo_strategy, "_CUML_KMEANS", True)
    monkeypatch.setattr(cluster_algo_strategy, "_cuml_cluster", _FakeCluster())

    _ = cluster_algo_strategy.gpu_kmeans(
        n_clusters=5,
        random_state=42,
        n_init=9,
        init="random",
        max_iter=123,
        unknown_arg=1,
    )
    assert captured["n_clusters"] == 5
    assert captured["random_state"] == 42
    assert captured["n_init"] == 9
    assert captured["init"] == "random"
    assert captured["max_iter"] == 123
