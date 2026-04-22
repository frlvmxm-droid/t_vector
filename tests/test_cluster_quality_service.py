"""Unit tests for cluster_quality_service.compute_quality."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from cluster_quality_service import compute_quality  # noqa: E402


def _three_blobs(seed: int = 0) -> tuple[object, object]:
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=(0, 0), scale=0.1, size=(30, 2))
    c2 = rng.normal(loc=(5, 5), scale=0.1, size=(30, 2))
    c3 = rng.normal(loc=(-5, 5), scale=0.1, size=(30, 2))
    vectors = np.vstack([c1, c2, c3])
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
    return vectors, labels


def test_compute_quality_returns_dict_keys() -> None:
    vectors, labels = _three_blobs()
    out = compute_quality(vectors, labels)
    assert set(out.keys()) == {"silhouette", "calinski_harabasz", "davies_bouldin"}


def test_compute_quality_well_separated_blobs() -> None:
    """Well-separated blobs → silhouette close to 1, DBI small, CH large."""
    vectors, labels = _three_blobs()
    out = compute_quality(vectors, labels)
    assert out["silhouette"] is not None and out["silhouette"] > 0.8
    assert out["davies_bouldin"] is not None and out["davies_bouldin"] < 0.5
    assert out["calinski_harabasz"] is not None and out["calinski_harabasz"] > 100.0


def test_compute_quality_single_cluster_returns_none() -> None:
    vectors, _ = _three_blobs()
    labels = np.zeros(vectors.shape[0], dtype=int)
    out = compute_quality(vectors, labels)
    assert out["silhouette"] is None
    assert out["calinski_harabasz"] is None
    assert out["davies_bouldin"] is None


def test_compute_quality_ignores_noise_labels() -> None:
    """``-1`` labels should be excluded from the metric computation."""
    vectors, labels = _three_blobs()
    # Mark a few points as noise; the remaining 3 clusters are still separable.
    labels = labels.copy()
    labels[:3] = -1
    out = compute_quality(vectors, labels)
    assert out["silhouette"] is not None and out["silhouette"] > 0.7


def test_compute_quality_all_noise_returns_none() -> None:
    vectors, _ = _three_blobs()
    labels = np.full(vectors.shape[0], -1, dtype=int)
    out = compute_quality(vectors, labels)
    assert out["silhouette"] is None
