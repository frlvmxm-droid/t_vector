"""Unit tests for `auto_k_service` (Phase 11 port of `_cluster_step_autok`)."""
from __future__ import annotations

import numpy as np
import pytest

from auto_k_service import (
    select_k,
    select_k_calinski,
    select_k_elbow,
    select_k_silhouette,
)


def _three_blob_matrix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]])
    blobs = []
    for c in centers:
        blobs.append(rng.normal(loc=c, scale=0.3, size=(40, 2)))
    return np.vstack(blobs).astype(np.float64)


def test_select_k_silhouette_recovers_three_blobs():
    X = _three_blob_matrix()
    k = select_k_silhouette(X, k_range=(2, 6), random_state=0, sample_size=120)
    assert k == 3


def test_select_k_calinski_recovers_three_blobs():
    X = _three_blob_matrix(seed=1)
    k = select_k_calinski(X, k_range=(2, 6), random_state=0)
    assert k == 3


def test_select_k_elbow_returns_value_in_range():
    X = _three_blob_matrix(seed=2)
    k = select_k_elbow(X, k_range=(2, 8), random_state=0)
    assert 2 <= k <= 8


def test_select_k_dispatch():
    X = _three_blob_matrix(seed=3)
    assert select_k(X, k_range=(2, 6), method="silhouette", random_state=0) == 3
    assert select_k(X, k_range=(2, 6), method="calinski", random_state=0) == 3
    k_elbow = select_k(X, k_range=(2, 6), method="elbow", random_state=0)
    assert 2 <= k_elbow <= 6


def test_select_k_unknown_method_raises():
    X = _three_blob_matrix(seed=4)
    with pytest.raises(ValueError):
        select_k(X, k_range=(2, 4), method="bogus")


def test_select_k_progress_callback_invoked():
    X = _three_blob_matrix(seed=5)
    calls = []

    def cb(frac: float, msg: str) -> None:
        calls.append((frac, msg))

    select_k_silhouette(X, k_range=(2, 4), random_state=0, sample_size=80, progress_cb=cb)
    assert len(calls) == 3
    assert all(0.0 < f <= 1.0 for f, _ in calls)


def test_select_k_handles_tiny_input():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    # n_rows-1 == 1 → range collapses; selector returns the lower bound.
    k = select_k_silhouette(X, k_range=(2, 5), random_state=0)
    assert k >= 2
