"""Переприсвоение HDBSCAN-шума ближайшему центроиду.

Порт десктопного closure `_cluster_step_hdbscan_reclust` (app_cluster.py
≈2582). HDBSCAN помечает часть точек как шум (label = -1); этот сервис
заменяет их меткой ближайшего кластера по центроидному расстоянию.
"""
from __future__ import annotations

from typing import Any


def reassign_noise(
    labels: Any,
    vectors: Any,
    *,
    k_neighbors: int = 5,
) -> Any:
    """Reassign ``label == -1`` points to their nearest non-noise centroid.

    Parameters
    ----------
    labels:
        ``(N,)`` integer array with ``-1`` for noise.
    vectors:
        ``(N, D)`` dense or sparse matrix used during clustering.
    k_neighbors:
        Unused in the centroid approach — reserved so the API can switch
        to a k-NN strategy later without breaking callers.

    Returns
    -------
    ``(N,)`` ndarray of labels with no ``-1`` values (unless no valid
    cluster exists at all, in which case the input is returned as-is).
    """
    import numpy as np

    del k_neighbors  # reserved

    labels_arr = np.asarray(labels).copy()
    noise_mask = labels_arr < 0
    if not noise_mask.any():
        return labels_arr

    valid_mask = ~noise_mask
    if not valid_mask.any():
        return labels_arr

    dense = vectors.toarray() if hasattr(vectors, "toarray") else np.asarray(vectors)
    valid_labels = labels_arr[valid_mask]
    unique = np.unique(valid_labels)

    centroids = np.stack(
        [dense[valid_mask][valid_labels == cid].mean(axis=0) for cid in unique],
        axis=0,
    )

    noise_pts = dense[noise_mask]
    # Squared Euclidean distances to each centroid.
    diff = noise_pts[:, None, :] - centroids[None, :, :]
    dists = np.einsum("nkd,nkd->nk", diff, diff)
    nearest = np.argmin(dists, axis=1)
    labels_arr[noise_mask] = unique[nearest]
    return labels_arr
