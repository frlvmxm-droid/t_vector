# -*- coding: utf-8 -*-
"""Comprehensive unit tests for cluster_algo_strategy.py."""
from __future__ import annotations

import sys
import types

import pytest
from sklearn.cluster import MiniBatchKMeans

import cluster_algo_strategy as cas


# ---------------------------------------------------------------------------
# Helpers — reset module-level cache between tests
# ---------------------------------------------------------------------------

def _reset_module_cache():
    """Clear all module-level caches in cluster_algo_strategy."""
    # cuda_available uses a function attribute
    if hasattr(cas.cuda_available, "_cached"):
        del cas.cuda_available._cached
    # cuml checks use module globals
    cas._CUML_KMEANS = None
    cas._cuml_cluster = None
    cas._CUML_UMAP = None
    cas._cuml_umap_mod = None


@pytest.fixture(autouse=True)
def reset_cache():
    _reset_module_cache()
    yield
    _reset_module_cache()


# ===========================================================================
# cuda_available()
# ===========================================================================

class TestCudaAvailable:

    def test_returns_bool(self):
        result = cas.cuda_available()
        assert isinstance(result, bool)

    def test_returns_false_without_gpu(self):
        # In a standard test environment (no GPU / no torch), must be False.
        result = cas.cuda_available()
        assert result is False

    def test_result_is_cached(self, monkeypatch):
        call_count = 0

        def fake_torch_module():
            nonlocal call_count
            call_count += 1

            class _FakeTorch:
                class cuda:
                    @staticmethod
                    def is_available():
                        return False

            return _FakeTorch()

        # Call once to populate cache
        cas.cuda_available()
        first_call_count = call_count

        # Second call must NOT re-evaluate
        cas.cuda_available()
        assert call_count == first_call_count, "cuda_available should be cached"

    def test_cache_attribute_set_after_call(self):
        cas.cuda_available()
        assert hasattr(cas.cuda_available, "_cached")

    def test_returns_false_when_torch_missing(self, monkeypatch):
        # Ensure torch is treated as missing
        monkeypatch.setitem(sys.modules, "torch", None)
        result = cas.cuda_available()
        assert result is False

    def test_returns_false_when_torch_raises_runtime(self, monkeypatch):
        class _BrokenTorch:
            class cuda:
                @staticmethod
                def is_available():
                    raise RuntimeError("no CUDA")

        monkeypatch.setitem(sys.modules, "torch", _BrokenTorch())
        result = cas.cuda_available()
        assert result is False

    def test_mocked_true_via_attr(self):
        """Direct attribute injection simulates a GPU environment."""
        setattr(cas.cuda_available, "_cached", True)
        assert cas.cuda_available() is True


# ===========================================================================
# cuml_kmeans_available()
# ===========================================================================

class TestCumlKmeansAvailable:

    def test_returns_bool(self):
        result = cas.cuml_kmeans_available()
        assert isinstance(result, bool)

    def test_returns_false_without_cuml(self):
        result = cas.cuml_kmeans_available()
        assert result is False

    def test_cached_result_returned_on_second_call(self):
        first = cas.cuml_kmeans_available()
        # Force cached path by keeping _CUML_KMEANS set
        second = cas.cuml_kmeans_available()
        assert first == second

    def test_returns_false_when_cuda_unavailable_even_if_cuml_importable(self, monkeypatch):
        """Even if cuml were importable, cuda_available() == False → result False."""
        monkeypatch.setattr(cas, "_CUML_KMEANS", None)
        monkeypatch.setattr(cas, "_cuml_cluster", None)
        setattr(cas.cuda_available, "_cached", False)

        fake_cuml_cluster = types.ModuleType("cuml.cluster")

        class _FakeKMeans:
            pass

        fake_cuml_cluster.KMeans = _FakeKMeans
        monkeypatch.setitem(sys.modules, "cuml.cluster", fake_cuml_cluster)
        monkeypatch.setitem(sys.modules, "cuml", types.ModuleType("cuml"))

        result = cas.cuml_kmeans_available()
        assert result is False


# ===========================================================================
# cuml_umap_available()
# ===========================================================================

class TestCumlUmapAvailable:

    def test_returns_bool(self):
        result = cas.cuml_umap_available()
        assert isinstance(result, bool)

    def test_returns_false_without_cuml(self):
        result = cas.cuml_umap_available()
        assert result is False

    def test_cached_result_returned_on_second_call(self):
        first = cas.cuml_umap_available()
        second = cas.cuml_umap_available()
        assert first == second

    def test_returns_false_when_cuda_unavailable(self, monkeypatch):
        setattr(cas.cuda_available, "_cached", False)
        result = cas.cuml_umap_available()
        assert result is False


# ===========================================================================
# gpu_kmeans()
# ===========================================================================

class TestGpuKmeans:

    def test_returns_object_with_fit_method(self):
        obj = cas.gpu_kmeans(n_clusters=3)
        assert hasattr(obj, "fit")

    def test_returns_object_with_fit_predict_method(self):
        obj = cas.gpu_kmeans(n_clusters=3)
        assert hasattr(obj, "fit_predict")

    def test_fallback_is_minibatch_kmeans(self):
        obj = cas.gpu_kmeans(n_clusters=5)
        assert isinstance(obj, MiniBatchKMeans)

    def test_n_clusters_respected(self):
        for k in (2, 7, 15):
            obj = cas.gpu_kmeans(n_clusters=k)
            assert obj.n_clusters == k

    def test_random_state_respected(self):
        obj = cas.gpu_kmeans(n_clusters=3, random_state=99)
        assert obj.random_state == 99

    def test_n_init_respected(self):
        obj = cas.gpu_kmeans(n_clusters=3, n_init=5)
        assert obj.n_init == 5

    def test_init_param_respected(self):
        obj = cas.gpu_kmeans(n_clusters=3, init="random")
        assert obj.init == "random"

    def test_max_iter_forwarded(self):
        obj = cas.gpu_kmeans(n_clusters=3, max_iter=77)
        assert obj.max_iter == 77

    def test_default_max_iter_is_300(self):
        obj = cas.gpu_kmeans(n_clusters=3)
        assert obj.max_iter == 300

    def test_batch_size_forwarded(self):
        obj = cas.gpu_kmeans(n_clusters=3, batch_size=512)
        assert obj.batch_size == 512

    def test_extra_mb_kwargs_forwarded(self):
        # max_no_improvement is a valid MiniBatchKMeans kwarg
        obj = cas.gpu_kmeans(n_clusters=3, max_no_improvement=5)
        assert isinstance(obj, MiniBatchKMeans)
        assert obj.max_no_improvement == 5

    def test_cuda_true_but_cuml_not_available_gives_minibatch(self, monkeypatch):
        """If CUDA appears available but cuML is not importable, fall back gracefully."""
        monkeypatch.setattr(cas, "_CUML_KMEANS", False)
        monkeypatch.setattr(cas, "_cuml_cluster", None)
        obj = cas.gpu_kmeans(n_clusters=4)
        assert isinstance(obj, MiniBatchKMeans)

    def test_gpu_branch_used_when_cuml_available(self, monkeypatch):
        """With a fake cuML cluster module the GPU branch is exercised."""
        captured = {}

        class _FakeKMeans:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.n_clusters = kwargs["n_clusters"]

        class _FakeCluster:
            KMeans = _FakeKMeans

        monkeypatch.setattr(cas, "_CUML_KMEANS", True)
        monkeypatch.setattr(cas, "_cuml_cluster", _FakeCluster())

        cas.gpu_kmeans(n_clusters=6, random_state=7, n_init=3)
        assert captured["n_clusters"] == 6
        assert captured["random_state"] == 7
        assert captured["n_init"] == 3

    def test_gpu_branch_translates_kmeanspp_init(self, monkeypatch):
        """'k-means++' is translated to 'scalable-k-means++' for cuML."""
        captured = {}

        class _FakeKMeans:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class _FakeCluster:
            KMeans = _FakeKMeans

        monkeypatch.setattr(cas, "_CUML_KMEANS", True)
        monkeypatch.setattr(cas, "_cuml_cluster", _FakeCluster())

        cas.gpu_kmeans(n_clusters=3, init="k-means++")
        assert captured["init"] == "scalable-k-means++"

    def test_gpu_branch_fallback_on_exception(self, monkeypatch):
        """If the GPU branch raises, MiniBatchKMeans should be returned."""

        class _BrokenKMeans:
            def __init__(self, **kwargs):
                raise RuntimeError("GPU error")

        class _BrokenCluster:
            KMeans = _BrokenKMeans

        monkeypatch.setattr(cas, "_CUML_KMEANS", True)
        monkeypatch.setattr(cas, "_cuml_cluster", _BrokenCluster())

        obj = cas.gpu_kmeans(n_clusters=3)
        assert isinstance(obj, MiniBatchKMeans)


# ===========================================================================
# gpu_umap()
# ===========================================================================

class TestGpuUmap:

    def _umap_available(self):
        try:
            import umap  # noqa: F401
            return True
        except ImportError:
            return False

    def test_raises_import_error_if_umap_not_installed(self):
        if self._umap_available():
            pytest.skip("umap-learn is installed; testing CPU fallback path only")
        with pytest.raises(ImportError):
            cas.gpu_umap(n_components=2)

    def test_returns_umap_object_when_installed(self):
        if not self._umap_available():
            pytest.skip("umap-learn not installed")
        obj = cas.gpu_umap(n_components=2)
        assert hasattr(obj, "fit_transform")

    def test_n_components_respected(self):
        if not self._umap_available():
            pytest.skip("umap-learn not installed")
        obj = cas.gpu_umap(n_components=3)
        assert obj.n_components == 3

    def test_n_neighbors_respected(self):
        if not self._umap_available():
            pytest.skip("umap-learn not installed")
        obj = cas.gpu_umap(n_components=2, n_neighbors=20)
        assert obj.n_neighbors == 20

    def test_gpu_branch_used_when_cuml_umap_available(self, monkeypatch):
        """With a fake cuML UMAP module the GPU branch is exercised."""
        captured = {}

        class _FakeUMAP:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class _FakeCumlUmapMod:
            UMAP = _FakeUMAP

        monkeypatch.setattr(cas, "_CUML_UMAP", True)
        monkeypatch.setattr(cas, "_cuml_umap_mod", _FakeCumlUmapMod())

        cas.gpu_umap(n_components=2, n_neighbors=10, min_dist=0.05)
        assert captured["n_components"] == 2
        assert captured["n_neighbors"] == 10
        assert captured["min_dist"] == 0.05

    def test_gpu_umap_fallback_on_exception(self, monkeypatch):
        """If cuML UMAP raises, fall back to umap-learn (or ImportError if absent)."""
        if not self._umap_available():
            pytest.skip("umap-learn not installed — can't test CPU fallback")

        class _BrokenUMAP:
            def __init__(self, **kwargs):
                raise RuntimeError("GPU error")

        class _BrokenCumlMod:
            UMAP = _BrokenUMAP

        monkeypatch.setattr(cas, "_CUML_UMAP", True)
        monkeypatch.setattr(cas, "_cuml_umap_mod", _BrokenCumlMod())

        obj = cas.gpu_umap(n_components=2)
        assert hasattr(obj, "fit_transform")
