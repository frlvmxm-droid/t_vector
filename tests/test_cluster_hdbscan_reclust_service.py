"""Unit tests for cluster_hdbscan_reclust_service.reassign_noise."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

np = pytest.importorskip("numpy")

from cluster_hdbscan_reclust_service import reassign_noise  # noqa: E402


def test_no_noise_returns_labels_unchanged() -> None:
    vectors = np.random.default_rng(0).random((10, 3))
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    out = reassign_noise(labels, vectors)
    assert np.array_equal(out, labels)


def test_noise_reassigned_to_nearest_centroid() -> None:
    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [0.05, 0.0],  # noise — closer to cluster 0
            [10.05, 10.0],  # noise — closer to cluster 1
        ]
    )
    labels = np.array([0, 0, 1, 1, -1, -1])
    out = reassign_noise(labels, vectors)
    assert (out >= 0).all()
    assert out[4] == 0
    assert out[5] == 1


def test_all_noise_returns_input() -> None:
    vectors = np.zeros((5, 3))
    labels = np.full(5, -1, dtype=int)
    out = reassign_noise(labels, vectors)
    assert np.array_equal(out, labels)


def test_sparse_input_densified_transparently() -> None:
    sp = pytest.importorskip("scipy.sparse")
    vectors = sp.csr_matrix([
        [0.0, 0.0],
        [0.1, 0.0],
        [10.0, 10.0],
        [0.05, 0.0],
    ])
    labels = np.array([0, 0, 1, -1])
    out = reassign_noise(labels, vectors)
    assert out[-1] == 0
    assert (out >= 0).all()


def test_original_labels_not_mutated() -> None:
    vectors = np.array([[0.0], [1.0], [10.0], [0.1]])
    labels = np.array([0, 0, 1, -1])
    before = labels.copy()
    _ = reassign_noise(labels, vectors)
    assert np.array_equal(labels, before)
