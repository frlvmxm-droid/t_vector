# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for app_apply_service.py — EnsemblePredictor.

Covers:
  - align_probabilities: same order, different order, extra class in proba2,
    missing class (fills with 0)
  - blend: w1=1.0, w1=0.0, intermediate weights, row sums, zero-row handling
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app_apply_service import EnsemblePredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(n_rows: int, n_cols: int) -> np.ndarray:
    """Return a row-normalised probability array with uniform values."""
    arr = np.ones((n_rows, n_cols), dtype=np.float64)
    return arr / n_cols


def _assert_rows_sum_to_one(arr: np.ndarray, tol: float = 1e-9) -> None:
    sums = arr.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=tol), f"Row sums not 1: {sums}"


# ===========================================================================
# align_probabilities
# ===========================================================================

class TestAlignProbabilities:

    def test_same_order_identity(self):
        """When class lists are identical the output equals the input."""
        classes = ["A", "B", "C"]
        proba = np.array([[0.1, 0.6, 0.3], [0.4, 0.4, 0.2]], dtype=np.float64)
        aligned = EnsemblePredictor.align_probabilities(proba, classes, classes)
        np.testing.assert_array_almost_equal(aligned, proba)

    def test_different_order_reordered(self):
        """Columns are reordered to match classes1 order."""
        classes1 = ["A", "B", "C"]
        classes2 = ["C", "A", "B"]
        proba2 = np.array([[0.3, 0.1, 0.6]], dtype=np.float64)  # C=0.3, A=0.1, B=0.6
        aligned = EnsemblePredictor.align_probabilities(proba2, classes1, classes2)
        # Expected: A=0.1, B=0.6, C=0.3
        expected = np.array([[0.1, 0.6, 0.3]])
        np.testing.assert_array_almost_equal(aligned, expected)

    def test_extra_class_in_proba2_is_ignored(self):
        """Extra classes present in classes2 but not classes1 are ignored."""
        classes1 = ["A", "B"]
        classes2 = ["A", "B", "EXTRA"]
        proba2 = np.array([[0.2, 0.5, 0.3]], dtype=np.float64)
        aligned = EnsemblePredictor.align_probabilities(proba2, classes1, classes2)
        expected = np.array([[0.2, 0.5]])
        np.testing.assert_array_almost_equal(aligned, expected)

    def test_missing_class_fills_with_zero(self):
        """A class present in classes1 but absent in classes2 gets probability 0."""
        classes1 = ["A", "B", "MISSING"]
        classes2 = ["A", "B"]
        proba2 = np.array([[0.4, 0.6]], dtype=np.float64)
        aligned = EnsemblePredictor.align_probabilities(proba2, classes1, classes2)
        assert aligned.shape == (1, 3)
        assert aligned[0, 2] == pytest.approx(0.0)
        assert aligned[0, 0] == pytest.approx(0.4)
        assert aligned[0, 1] == pytest.approx(0.6)

    def test_output_shape_matches_classes1_length(self):
        classes1 = ["X", "Y", "Z"]
        classes2 = ["X", "Y"]
        proba2 = np.ones((5, 2), dtype=np.float64) * 0.5
        aligned = EnsemblePredictor.align_probabilities(proba2, classes1, classes2)
        assert aligned.shape == (5, 3)

    def test_single_class_same(self):
        classes = ["ONLY"]
        proba = np.array([[1.0]], dtype=np.float64)
        aligned = EnsemblePredictor.align_probabilities(proba, classes, classes)
        np.testing.assert_array_almost_equal(aligned, proba)

    def test_single_class_missing(self):
        classes1 = ["A"]
        classes2 = ["B"]
        proba2 = np.array([[1.0]], dtype=np.float64)
        aligned = EnsemblePredictor.align_probabilities(proba2, classes1, classes2)
        assert aligned[0, 0] == pytest.approx(0.0)

    def test_preserves_dtype(self):
        classes = ["A", "B"]
        proba = np.array([[0.3, 0.7]], dtype=np.float32)
        aligned = EnsemblePredictor.align_probabilities(proba, classes, classes)
        assert aligned.dtype == np.float32

    def test_multiple_rows_different_order(self):
        classes1 = ["cat", "dog"]
        classes2 = ["dog", "cat"]
        proba2 = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        aligned = EnsemblePredictor.align_probabilities(proba2, classes1, classes2)
        # cat column should come from index 1 of proba2
        np.testing.assert_array_almost_equal(aligned[:, 0], [0.2, 0.7])
        np.testing.assert_array_almost_equal(aligned[:, 1], [0.8, 0.3])

    def test_all_classes_missing_gives_zeros(self):
        classes1 = ["A", "B"]
        classes2 = ["X", "Y"]
        proba2 = np.array([[0.5, 0.5]], dtype=np.float64)
        aligned = EnsemblePredictor.align_probabilities(proba2, classes1, classes2)
        np.testing.assert_array_almost_equal(aligned, np.zeros((1, 2)))


# ===========================================================================
# blend
# ===========================================================================

class TestBlend:

    def _make_pair(self, n_rows: int = 3, n_cols: int = 3):
        rng = np.random.default_rng(42)
        p1 = rng.dirichlet(np.ones(n_cols), size=n_rows)
        p2 = rng.dirichlet(np.ones(n_cols), size=n_rows)
        return p1, p2

    def test_w1_one_returns_proba1(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=1.0)
        np.testing.assert_array_almost_equal(result, p1)

    def test_w1_zero_returns_proba2(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=0.0)
        np.testing.assert_array_almost_equal(result, p2)

    def test_intermediate_weight_rows_sum_to_one(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=0.6)
        _assert_rows_sum_to_one(result)

    def test_w1_half_rows_sum_to_one(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=0.5)
        _assert_rows_sum_to_one(result)

    def test_w1_one_rows_sum_to_one(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=1.0)
        _assert_rows_sum_to_one(result)

    def test_w1_zero_rows_sum_to_one(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=0.0)
        _assert_rows_sum_to_one(result)

    def test_zero_row_no_division_by_zero(self):
        """A row of all zeros should not raise and should remain [0,...,0]."""
        p1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        p2 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = EnsemblePredictor.blend(p1, p2, w1=0.5)
        # Should not raise; result is all zeros (or whatever the impl does safely)
        assert result.shape == (1, 3)
        assert not np.any(np.isnan(result))

    def test_blend_output_shape(self):
        p1 = np.ones((4, 5)) / 5.0
        p2 = np.ones((4, 5)) / 5.0
        result = EnsemblePredictor.blend(p1, p2, w1=0.7)
        assert result.shape == (4, 5)

    def test_intermediate_blend_values(self):
        """Manual check: w1=0.5 averages the two distributions."""
        p1 = np.array([[0.8, 0.2]], dtype=np.float64)
        p2 = np.array([[0.4, 0.6]], dtype=np.float64)
        result = EnsemblePredictor.blend(p1, p2, w1=0.5)
        expected = np.array([[0.6, 0.4]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_blend_w1_quarter(self):
        p1 = np.array([[1.0, 0.0]], dtype=np.float64)
        p2 = np.array([[0.0, 1.0]], dtype=np.float64)
        result = EnsemblePredictor.blend(p1, p2, w1=0.25)
        # 0.25*[1,0] + 0.75*[0,1] = [0.25, 0.75]
        np.testing.assert_array_almost_equal(result, [[0.25, 0.75]])

    def test_non_negative_probabilities(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=0.3)
        assert np.all(result >= 0.0)

    def test_probabilities_at_most_one(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=0.3)
        assert np.all(result <= 1.0 + 1e-9)

    def test_many_rows_all_sum_to_one(self):
        rng = np.random.default_rng(99)
        p1 = rng.dirichlet(np.ones(10), size=100)
        p2 = rng.dirichlet(np.ones(10), size=100)
        result = EnsemblePredictor.blend(p1, p2, w1=0.4)
        _assert_rows_sum_to_one(result)

    def test_single_row_blend(self):
        p1 = np.array([[0.5, 0.5]], dtype=np.float64)
        p2 = np.array([[0.5, 0.5]], dtype=np.float64)
        result = EnsemblePredictor.blend(p1, p2, w1=0.5)
        np.testing.assert_array_almost_equal(result, [[0.5, 0.5]])

    def test_returns_ndarray(self):
        p1, p2 = self._make_pair()
        result = EnsemblePredictor.blend(p1, p2, w1=0.5)
        assert isinstance(result, np.ndarray)
