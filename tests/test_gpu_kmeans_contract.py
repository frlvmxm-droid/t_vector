import app_cluster


def test_gpu_kmeans_cpu_branch_respects_params(monkeypatch):
    monkeypatch.setattr(app_cluster, "_CUML_KMEANS", False)
    km = app_cluster._gpu_kmeans(
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

    monkeypatch.setattr(app_cluster, "_CUML_KMEANS", True)
    monkeypatch.setattr(app_cluster, "_cuml_cluster", _FakeCluster())

    _ = app_cluster._gpu_kmeans(
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
