# -*- coding: utf-8 -*-
"""Cluster algorithm strategy (CPU/GPU selection + fallback)."""
from __future__ import annotations

import logging
from typing import Optional

from sklearn.cluster import MiniBatchKMeans

from config.ml_constants import KMEANS_BATCH_SIZE

_log = logging.getLogger(__name__)

_cuml_cluster = None
_CUML_KMEANS: Optional[bool] = None
_cuml_umap_mod = None
_CUML_UMAP: Optional[bool] = None


def cuda_available() -> bool:
    if hasattr(cuda_available, "_cached"):
        return bool(getattr(cuda_available, "_cached"))
    try:
        import torch

        ok = torch.cuda.is_available()
    except (ImportError, RuntimeError, AttributeError) as ex:
        _log.debug("cuda_available fallback: %s", ex)
        ok = False
    setattr(cuda_available, "_cached", bool(ok))
    return bool(ok)


def cuml_kmeans_available() -> bool:
    global _cuml_cluster, _CUML_KMEANS
    if _CUML_KMEANS is not None:
        return bool(_CUML_KMEANS)
    try:
        import cuml.cluster as _cuml_cluster_mod

        _cuml_cluster = _cuml_cluster_mod
        _CUML_KMEANS = cuda_available()
    except (ImportError, OSError, RuntimeError, AttributeError) as ex:
        _log.debug("cuML cluster unavailable: %s", ex)
        _cuml_cluster = None
        _CUML_KMEANS = False
    return bool(_CUML_KMEANS)


def cuml_umap_available() -> bool:
    global _cuml_umap_mod, _CUML_UMAP
    if _CUML_UMAP is not None:
        return bool(_CUML_UMAP)
    try:
        import cuml.manifold.umap as _cuml_umap_module

        _cuml_umap_mod = _cuml_umap_module
        _CUML_UMAP = cuda_available()
    except (ImportError, OSError, RuntimeError, AttributeError) as ex:
        _log.debug("cuML UMAP unavailable: %s", ex)
        _cuml_umap_mod = None
        _CUML_UMAP = False
    return bool(_CUML_UMAP)


def gpu_kmeans(
    n_clusters: int,
    random_state: int = 42,
    batch_size: int = KMEANS_BATCH_SIZE,
    n_init=10,
    init: str = "k-means++",
    **mb_kwargs,
) -> object:
    max_iter = int(mb_kwargs.pop("max_iter", 300))
    if cuml_kmeans_available() and _cuml_cluster is not None:
        try:
            gpu_kwargs = dict(
                n_clusters=n_clusters,
                random_state=random_state,
                init=init if init != "k-means++" else "scalable-k-means++",
                n_init=n_init,
                max_iter=max_iter,
            )
            if mb_kwargs:
                _log.debug("cuML KMeans ignores unsupported kwargs: %s", sorted(mb_kwargs.keys()))
            return _cuml_cluster.KMeans(**gpu_kwargs)
        except (ImportError, RuntimeError, AttributeError) as ex:
            _log.warning("cuML KMeans unavailable, falling back to MiniBatchKMeans: %s", ex)
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
        init=init,
        max_iter=max_iter,
        **mb_kwargs,
    )


def gpu_umap(
    n_components: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> object:
    if cuml_umap_available() and _cuml_umap_mod is not None:
        try:
            return _cuml_umap_mod.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
            )
        except (ImportError, RuntimeError, AttributeError) as ex:
            _log.warning("cuML UMAP unavailable, falling back to umap-learn: %s", ex)
    import umap as _umap_cpu

    return _umap_cpu.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
