"""Чистые метрики качества кластеризации (без tkinter).

Порт десктопного closure `_cluster_step_quality_metrics` из
`app_cluster.py` (строка ≈2453). Возвращает silhouette / Calinski–Harabasz
/ Davies–Bouldin; коэффициент тематической когерентности (`c_npmi`
через gensim) остаётся опциональным.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def compute_quality(
    vectors: Any,
    labels: Any,
    *,
    texts: Sequence[str] | None = None,
    sample_size: int = 5_000,
    random_state: int = 42,
) -> dict[str, float | None]:
    """Compute silhouette / DBI / CH for the clustering.

    Parameters
    ----------
    vectors:
        ``(N, D)`` dense or sparse matrix used during clustering.
    labels:
        ``(N,)`` integer labels. ``-1`` entries are treated as noise and
        ignored in the computation (matches the desktop closure).
    texts:
        Optional raw texts — reserved for future `c_npmi` coherence wiring
        (gensim); currently unused.
    sample_size:
        Cap for silhouette sampling (the full computation is O(N²)).
    random_state:
        Seed for silhouette's internal sampler.

    Returns
    -------
    dict with keys ``silhouette`` / ``calinski_harabasz`` / ``davies_bouldin``.
    Missing values are ``None`` (e.g. if only one cluster remains).
    """
    import numpy as np

    del texts  # reserved — no coherence backend wired yet

    labels_arr = np.asarray(labels)
    out: dict[str, float | None] = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    valid_mask = labels_arr >= 0
    if not valid_mask.any():
        return out

    lbl = labels_arr[valid_mask]
    unique = np.unique(lbl)
    if unique.size < 2:
        return out

    try:
        vec = vectors[valid_mask] if hasattr(vectors, "__getitem__") else vectors
    except Exception:
        vec = vectors

    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
    except ImportError:
        return out

    n = lbl.shape[0]
    sample = min(sample_size, n) if sample_size > 0 else n
    try:
        out["silhouette"] = float(
            silhouette_score(vec, lbl, sample_size=sample, random_state=random_state)
        )
    except Exception:
        out["silhouette"] = None

    try:
        dense = vec.toarray() if hasattr(vec, "toarray") else np.asarray(vec)
        out["calinski_harabasz"] = float(calinski_harabasz_score(dense, lbl))
    except Exception:
        out["calinski_harabasz"] = None

    try:
        dense = vec.toarray() if hasattr(vec, "toarray") else np.asarray(vec)
        out["davies_bouldin"] = float(davies_bouldin_score(dense, lbl))
    except Exception:
        out["davies_bouldin"] = None

    return out
