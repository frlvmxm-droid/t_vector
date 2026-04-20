"""UMAP-редукция для кластеризации (без tkinter-зависимостей).

Порт десктопного closure `_cluster_step_umap` (app_cluster.py ≈2714).
UMAP — тяжёлая опциональная зависимость; если `umap-learn` не установлен
или данных мало (`n_rows < n_neighbors * 2`), сервис graceful-degrades:
возвращает исходную матрицу без падения.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def reduce_with_umap(
    vectors: Any,
    *,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    progress_cb: Callable[[float, str], None] | None = None,
) -> Any:
    """Run UMAP → return dense ``(N, n_components)`` matrix.

    Falls back to the input matrix when UMAP is unavailable, the data
    is too small (``N < n_neighbors * 2``), or any exception occurs.
    The caller can treat this as an opt-in optimisation.
    """
    import numpy as np

    if progress_cb is not None:
        try:
            progress_cb(0.0, "UMAP: подготовка…")
        except Exception:
            pass

    try:
        dense = vectors.toarray() if hasattr(vectors, "toarray") else np.asarray(vectors)
    except Exception:
        return vectors

    if dense.ndim != 2 or dense.shape[0] < 2:
        return vectors

    n_rows, n_feat = dense.shape
    effective_neighbors = max(2, min(n_neighbors, n_rows - 1))
    # UMAP needs a minimum density: n_rows ≥ 2 × n_neighbors for stable
    # manifold estimation. Fall through silently otherwise.
    if n_rows < effective_neighbors * 2:
        if progress_cb is not None:
            try:
                progress_cb(1.0, "UMAP пропущен: мало данных")
            except Exception:
                pass
        return dense

    effective_components = max(2, min(n_components, n_feat - 1, n_rows - 2))

    try:
        import umap  # type: ignore[import-not-found]
    except ImportError:
        if progress_cb is not None:
            try:
                progress_cb(1.0, "UMAP пропущен: umap-learn не установлен")
            except Exception:
                pass
        return dense

    if progress_cb is not None:
        try:
            progress_cb(0.4, "UMAP: fit_transform…")
        except Exception:
            pass

    try:
        reducer = umap.UMAP(
            n_components=effective_components,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric=metric,
            random_state=int(random_state),
        )
        reduced = reducer.fit_transform(dense)
    except Exception:
        return dense

    if progress_cb is not None:
        try:
            progress_cb(1.0, "UMAP готово")
        except Exception:
            pass
    return np.asarray(reduced)
