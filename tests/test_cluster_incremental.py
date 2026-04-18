# -*- coding: utf-8 -*-
"""
Unit tests for cluster_incremental_service.py.

Tests cover:
 - assign_labels_by_centers() — distance-based label assignment
 - IncrementalApplyResult    — frozen dataclass structure
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import sparse as sp

# Import only the parts that don't need heavy infra (no joblib model files).
from cluster_incremental_service import IncrementalApplyResult, assign_labels_by_centers


# ===========================================================================
# Helpers
# ===========================================================================

def _dense(rows):
    """Return a numpy float64 array from a list-of-lists."""
    return np.array(rows, dtype=np.float64)


# ===========================================================================
# assign_labels_by_centers()
# ===========================================================================

class TestAssignLabelsByCenters:

    def test_single_point_single_center(self):
        xv = _dense([[0.0, 0.0]])
        centers = _dense([[0.0, 0.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert labels.tolist() == [0]

    def test_two_points_two_centers(self):
        xv = _dense([[0.0, 0.0], [10.0, 10.0]])
        centers = _dense([[0.0, 0.0], [10.0, 10.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert labels.tolist() == [0, 1]

    def test_points_assigned_to_nearest_center(self):
        # Points: (1,0) and (9,0); Centers: (0,0) and (10,0)
        xv = _dense([[1.0, 0.0], [9.0, 0.0]])
        centers = _dense([[0.0, 0.0], [10.0, 0.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert labels[0] == 0  # (1,0) closer to (0,0)
        assert labels[1] == 1  # (9,0) closer to (10,0)

    def test_empty_input_returns_empty_array(self):
        xv = _dense([]).reshape(0, 2)
        centers = _dense([[0.0, 0.0], [1.0, 1.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert len(labels) == 0
        assert isinstance(labels, np.ndarray)

    def test_returns_numpy_array(self):
        xv = _dense([[0.0, 0.0]])
        centers = _dense([[0.0, 0.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert isinstance(labels, np.ndarray)

    def test_dtype_is_int(self):
        xv = _dense([[0.0, 0.0]])
        centers = _dense([[0.0, 0.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_labels_within_valid_range(self):
        n_centers = 5
        xv = _dense([[float(i), 0.0] for i in range(20)])
        centers = _dense([[float(i * 4), 0.0] for i in range(n_centers)])
        labels = assign_labels_by_centers(xv, centers)
        assert labels.min() >= 0
        assert labels.max() < n_centers

    def test_three_clusters_correct_assignment(self):
        centers = _dense([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        xv = _dense([[0.1, 0.0], [4.9, 0.0], [10.1, 0.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert labels[0] == 0
        assert labels[1] == 1
        assert labels[2] == 2

    def test_output_length_matches_input(self):
        n = 17
        xv = np.random.default_rng(0).standard_normal((n, 4))
        centers = np.random.default_rng(1).standard_normal((3, 4))
        labels = assign_labels_by_centers(xv, centers)
        assert len(labels) == n

    def test_batch_size_smaller_than_input_triggers_batching(self):
        """Force batching by using a very small batch_size and large max_distance_cells=0."""
        np.random.seed(42)
        xv = np.random.standard_normal((50, 4))
        centers = np.random.standard_normal((3, 4))
        # max_distance_cells set so small that batching is always triggered
        labels_batched = assign_labels_by_centers(
            xv, centers, batch_size=10, max_distance_cells=1
        )
        labels_direct = assign_labels_by_centers(
            xv, centers, batch_size=10, max_distance_cells=10_000_000
        )
        # Both paths should produce identical assignments
        np.testing.assert_array_equal(labels_batched, labels_direct)

    def test_batching_gives_same_result_as_direct(self):
        rng = np.random.default_rng(7)
        xv = rng.standard_normal((100, 8))
        centers = rng.standard_normal((5, 8))
        labels_direct = assign_labels_by_centers(xv, centers, max_distance_cells=10_000_000)
        labels_batched = assign_labels_by_centers(xv, centers, batch_size=20, max_distance_cells=1)
        np.testing.assert_array_equal(labels_direct, labels_batched)

    def test_sparse_input_handled_in_batch_path(self):
        """Sparse matrix triggers the tocsr() path in the batch branch."""
        rng = np.random.default_rng(3)
        dense = rng.standard_normal((30, 6))
        xv_sparse = sp.csr_matrix(dense)
        centers = rng.standard_normal((3, 6))

        labels_sparse = assign_labels_by_centers(
            xv_sparse, centers, batch_size=10, max_distance_cells=1
        )
        labels_dense = assign_labels_by_centers(
            dense, centers, max_distance_cells=10_000_000
        )
        np.testing.assert_array_equal(labels_sparse, labels_dense)

    def test_single_center_all_points_go_to_zero(self):
        xv = _dense([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
        centers = _dense([[0.0, 0.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert all(l == 0 for l in labels)

    def test_large_batch_size_does_not_crash(self):
        xv = _dense([[float(i), 0.0] for i in range(10)])
        centers = _dense([[0.0, 0.0], [5.0, 0.0]])
        labels = assign_labels_by_centers(xv, centers, batch_size=100_000)
        assert len(labels) == 10

    def test_equidistant_points_still_returns_valid_label(self):
        """A point equidistant from two centers must still get a valid (one of two) label."""
        xv = _dense([[5.0, 0.0]])
        centers = _dense([[0.0, 0.0], [10.0, 0.0]])
        labels = assign_labels_by_centers(xv, centers)
        assert labels[0] in (0, 1)


# ===========================================================================
# IncrementalApplyResult  — frozen dataclass
# ===========================================================================

class TestIncrementalApplyResult:

    def _make_result(self, **overrides):
        defaults = dict(
            vectorizer=object(),
            algo="kmeans",
            k_clusters=5,
            kw=["topic_a", "topic_b"],
            use_fastopic_kw_ready=False,
            xv=np.zeros((3, 4)),
            labels=np.array([0, 1, 2]),
        )
        defaults.update(overrides)
        return IncrementalApplyResult(**defaults)

    def test_is_dataclass(self):
        result = self._make_result()
        assert dataclasses.is_dataclass(result)

    def test_is_frozen(self):
        result = self._make_result()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.algo = "hdbscan"  # type: ignore[misc]

    def test_vectorizer_field_accessible(self):
        fake_vec = object()
        result = self._make_result(vectorizer=fake_vec)
        assert result.vectorizer is fake_vec

    def test_algo_field_accessible(self):
        result = self._make_result(algo="hdbscan")
        assert result.algo == "hdbscan"

    def test_k_clusters_field_accessible(self):
        result = self._make_result(k_clusters=12)
        assert result.k_clusters == 12

    def test_kw_field_accessible(self):
        kw = ["word1", "word2", "word3"]
        result = self._make_result(kw=kw)
        assert result.kw == kw

    def test_use_fastopic_kw_ready_true(self):
        result = self._make_result(use_fastopic_kw_ready=True)
        assert result.use_fastopic_kw_ready is True

    def test_use_fastopic_kw_ready_false(self):
        result = self._make_result(use_fastopic_kw_ready=False)
        assert result.use_fastopic_kw_ready is False

    def test_xv_field_accessible(self):
        xv = np.ones((5, 3))
        result = self._make_result(xv=xv)
        np.testing.assert_array_equal(result.xv, xv)

    def test_labels_field_accessible(self):
        labels = np.array([0, 1, 0, 2])
        result = self._make_result(labels=labels)
        np.testing.assert_array_equal(result.labels, labels)

    def test_all_fields_present(self):
        result = self._make_result()
        fields = {f.name for f in dataclasses.fields(result)}
        expected = {"vectorizer", "algo", "k_clusters", "kw", "use_fastopic_kw_ready", "xv", "labels"}
        assert expected.issubset(fields)
