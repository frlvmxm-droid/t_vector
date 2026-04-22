"""Headless port of `app_cluster._cluster_step_autok` (auto-K selection).

Pure-Python helpers for picking the number of clusters K from a vector
matrix. Three score methods supported, mirroring the desktop closure at
``app_cluster.py:2783``:

* ``"silhouette"`` — sklearn ``silhouette_score`` on subsampled vectors.
* ``"calinski"`` — sklearn ``calinski_harabasz_score`` (variance ratio).
* ``"elbow"`` — KMeans inertia → ``ClusterElbowSelector.pick_elbow_k``.

The desktop closure mixes UI hooks (``self.after``, ``self.log_cluster``,
``self._cancel_event``) into the same loop. This module strips them out
and accepts an optional ``progress_cb(frac, msg)`` instead.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from app_logger import get_logger
from cluster_elbow import ClusterElbowSelector

_log = get_logger(__name__)

ProgressCB = Callable[[float, str], None]


def _to_dense(matrix):
    import numpy as _np
    import scipy.sparse as _sp

    return matrix.toarray() if _sp.issparse(matrix) else _np.asarray(matrix)


def _resolve_range(
    k_range: tuple[int, int], n_rows: int,
) -> list[int]:
    lo_req, hi_req = int(k_range[0]), int(k_range[1])
    lo = max(2, lo_req)
    hi = min(hi_req, n_rows - 1)
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _fit_minibatch_kmeans(matrix, k: int, random_state: int):
    """Fit MiniBatchKMeans with the same defaults as the desktop closure."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=1024,
        n_init=10,
        init="k-means++",
        max_iter=300,
        reassignment_ratio=0.01,
    )
    return km


def select_k_silhouette(
    vectors,
    k_range: tuple[int, int] = (2, 20),
    *,
    random_state: int = 42,
    sample_size: int = 5000,
    progress_cb: ProgressCB | None = None,
) -> int:
    """Pick K maximising silhouette score over ``range(*k_range)``.

    Returns the K with the highest silhouette score, or the lower bound
    of ``k_range`` if no candidate could be evaluated (degenerate input).
    """
    from sklearn.metrics import silhouette_score

    dense = _to_dense(vectors)
    n_rows = int(dense.shape[0])
    ks = _resolve_range(k_range, n_rows)
    if len(ks) < 2:
        return ks[0] if ks else max(2, k_range[0])

    sample = min(int(sample_size), n_rows)
    scores: list[float] = []
    for i, k in enumerate(ks):
        km = _fit_minibatch_kmeans(dense, k, random_state)
        labels = km.fit_predict(dense)
        try:
            score = float(
                silhouette_score(
                    dense, labels,
                    sample_size=sample,
                    random_state=random_state,
                )
            )
        except Exception as exc:  # noqa: BLE001 — degenerate clustering
            _log.debug("silhouette_score failed for K=%d: %s", k, exc)
            score = -1.0
        scores.append(score)
        if progress_cb is not None:
            frac = (i + 1) / len(ks)
            progress_cb(frac, f"silhouette K={k}: {score:.3f}")
    best_idx = int(_argmax(scores))
    return int(ks[best_idx])


def select_k_calinski(
    vectors,
    k_range: tuple[int, int] = (2, 20),
    *,
    random_state: int = 42,
    progress_cb: ProgressCB | None = None,
) -> int:
    """Pick K maximising Calinski-Harabasz (variance-ratio) score."""
    from sklearn.metrics import calinski_harabasz_score

    dense = _to_dense(vectors)
    n_rows = int(dense.shape[0])
    ks = _resolve_range(k_range, n_rows)
    if len(ks) < 2:
        return ks[0] if ks else max(2, k_range[0])

    scores: list[float] = []
    for i, k in enumerate(ks):
        km = _fit_minibatch_kmeans(dense, k, random_state)
        labels = km.fit_predict(dense)
        try:
            score = float(calinski_harabasz_score(dense, labels))
        except Exception as exc:  # noqa: BLE001 — degenerate clustering
            _log.debug("calinski_harabasz_score failed for K=%d: %s", k, exc)
            score = -1.0
        scores.append(score)
        if progress_cb is not None:
            frac = (i + 1) / len(ks)
            progress_cb(frac, f"calinski K={k}: {score:.1f}")
    best_idx = int(_argmax(scores))
    return int(ks[best_idx])


def select_k_elbow(
    vectors,
    k_range: tuple[int, int] = (2, 20),
    *,
    random_state: int = 42,
    progress_cb: ProgressCB | None = None,
) -> int:
    """Pick K via inertia elbow (delegates to ``ClusterElbowSelector``)."""
    n_rows = int(_to_dense(vectors).shape[0])
    ks = _resolve_range(k_range, n_rows)
    if len(ks) < 2:
        return ks[0] if ks else max(2, k_range[0])

    inertias: list[float] = []
    for i, k in enumerate(ks):
        km = _fit_minibatch_kmeans(vectors, k, random_state)
        km.fit(vectors)
        inertias.append(float(km.inertia_))
        if progress_cb is not None:
            frac = (i + 1) / len(ks)
            progress_cb(frac, f"elbow K={k}")
    return int(ClusterElbowSelector.pick_elbow_k(inertias, ks))


def select_k(
    vectors,
    k_range: tuple[int, int] = (2, 20),
    *,
    method: str = "silhouette",
    random_state: int = 42,
    sample_size: int = 5000,
    progress_cb: ProgressCB | None = None,
) -> int:
    """Dispatch to ``silhouette`` / ``calinski`` / ``elbow`` selector."""
    m = (method or "silhouette").lower()
    if m == "silhouette":
        return select_k_silhouette(
            vectors, k_range,
            random_state=random_state,
            sample_size=sample_size,
            progress_cb=progress_cb,
        )
    if m == "calinski":
        return select_k_calinski(
            vectors, k_range,
            random_state=random_state,
            progress_cb=progress_cb,
        )
    if m == "elbow":
        return select_k_elbow(
            vectors, k_range,
            random_state=random_state,
            progress_cb=progress_cb,
        )
    raise ValueError(
        f"Unknown auto-K method {method!r}; "
        "expected one of: silhouette, calinski, elbow"
    )


def _argmax(values: Iterable[float]) -> int:
    best_i = 0
    best_v = float("-inf")
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i
