"""Unit tests for cluster_umap_service.reduce_with_umap."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

np = pytest.importorskip("numpy")

from cluster_umap_service import reduce_with_umap  # noqa: E402


def test_small_input_falls_through() -> None:
    """``n_rows < n_neighbors * 2`` → returns densified input unchanged shape."""
    vectors = np.random.default_rng(0).random((10, 20))
    out = reduce_with_umap(
        vectors, n_components=5, n_neighbors=15, random_state=42,
    )
    # With only 10 rows and n_neighbors=15, UMAP is skipped → original shape.
    assert np.asarray(out).shape == vectors.shape


def test_reduces_shape_when_umap_available() -> None:
    umap = pytest.importorskip("umap")  # noqa: F841
    rng = np.random.default_rng(0)
    # 3 well-separated blobs in 50D → 4D embedding
    c1 = rng.normal(loc=0.0, scale=0.1, size=(30, 50))
    c2 = rng.normal(loc=5.0, scale=0.1, size=(30, 50))
    c3 = rng.normal(loc=-5.0, scale=0.1, size=(30, 50))
    vectors = np.vstack([c1, c2, c3])

    reduced = reduce_with_umap(
        vectors,
        n_components=4,
        n_neighbors=15,
        random_state=42,
    )
    assert np.asarray(reduced).shape == (90, 4)


def test_progress_cb_invoked() -> None:
    calls: list[tuple[float, str]] = []
    vectors = np.random.default_rng(0).random((5, 10))
    reduce_with_umap(
        vectors,
        n_components=3,
        n_neighbors=15,
        random_state=42,
        progress_cb=lambda f, m: calls.append((f, m)),
    )
    # At minimum: preparation + fallthrough.
    assert len(calls) >= 1
    assert calls[-1][0] == 1.0


def test_no_crash_on_nan_input() -> None:
    """NaN-heavy input should not raise — fall through to original matrix."""
    vectors = np.full((30, 8), np.nan)
    out = reduce_with_umap(vectors, n_components=3, n_neighbors=5, random_state=42)
    # Either skipped (original returned) or transformed — either way, no crash.
    assert out is not None


def test_sparse_input_handled() -> None:
    """Sparse input should densify inside the service."""
    sp = pytest.importorskip("scipy.sparse")
    vectors = sp.csr_matrix(np.eye(5))
    out = reduce_with_umap(vectors, n_components=2, n_neighbors=15)
    # Skipped → returns the dense version (np.eye(5).shape).
    assert np.asarray(out).shape[0] == 5
