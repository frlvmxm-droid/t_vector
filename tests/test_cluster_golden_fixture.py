# -*- coding: utf-8 -*-
"""Golden-fixture regression for the clustering math migration (Wave 3a).

Purpose:
    Before porting `build_vectors` + `run_clustering` from the 969-LOC
    `run_cluster()` into pure functions in `app_cluster_pipeline.py`,
    we need a safety net that detects silent regressions in cluster
    composition. Line-for-line equivalence is impossible (cluster IDs
    are arbitrary), so we use **Hungarian matching** to align predicted
    labels to ground truth, then assert purity + size distribution.

Design:
    • Synthetic Gaussian blobs (seeded numpy) — deterministic, no binary
      fixtures committed, portable across OSes.
    • `_hungarian_match_labels` uses `scipy.optimize.linear_sum_assignment`
      on the confusion-matrix cost (−count → maximise agreement).
    • The test uses sklearn's `KMeans` directly today as a reference.
      When W3a migrates `run_clustering`, the test will switch to call
      `app_cluster_pipeline.run_clustering(vectors, snap)` instead, and
      the same assertions will catch any math drift.

If `BRT_CLUSTER_GOLDEN_RECORD=1` is set, the test prints the observed
sizes + purity to stderr — used to (re)generate fixture ground truth.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import List, Tuple

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

np = pytest.importorskip("numpy")
sklearn_cluster = pytest.importorskip("sklearn.cluster")
pytest.importorskip("scipy.optimize")

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "clustering"


# ---------------------------------------------------------------------------
# Hungarian label matching
# ---------------------------------------------------------------------------

def _hungarian_match_labels(y_true, y_pred) -> Tuple[dict, float]:
    """Align predicted cluster IDs to ground truth via Hungarian matching.

    Returns (mapping pred_id -> true_id, purity in [0,1]).
    """
    from scipy.optimize import linear_sum_assignment

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    true_labels = sorted(set(int(x) for x in y_true))
    pred_labels = sorted(set(int(x) for x in y_pred))
    size = max(len(true_labels), len(pred_labels))

    idx_true = {lab: i for i, lab in enumerate(true_labels)}
    idx_pred = {lab: i for i, lab in enumerate(pred_labels)}

    # Confusion matrix: rows = predicted, cols = true. Pad to square so that
    # linear_sum_assignment always has a solution.
    cost = np.zeros((size, size), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        cost[idx_pred[int(p)], idx_true[int(t)]] -= 1  # maximise match → minimise −count

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: dict = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(pred_labels) and c < len(true_labels):
            mapping[pred_labels[r]] = true_labels[c]

    n_correct = sum(1 for p, t in zip(y_pred, y_true) if mapping.get(int(p)) == int(t))
    purity = n_correct / len(y_true)
    return mapping, purity


# ---------------------------------------------------------------------------
# Synthetic fixture generation
# ---------------------------------------------------------------------------

def _generate_blobs(seed: int, n_points: int, n_features: int,
                    n_clusters: int, centers_scale: float,
                    blob_std: float) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic Gaussian blobs with equal sizes."""
    rng = np.random.default_rng(seed)
    # Place centers uniformly on a sphere of radius centers_scale
    centers = rng.standard_normal((n_clusters, n_features))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= centers_scale

    points_per = n_points // n_clusters
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for k in range(n_clusters):
        pts = centers[k] + rng.standard_normal((points_per, n_features)) * blob_std
        X_parts.append(pts)
        y_parts.append(np.full(points_per, k, dtype=np.int64))

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)

    # Shuffle deterministically — otherwise points are grouped by class.
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def three_blobs_spec():
    with open(FIXTURE_DIR / "three_blobs_k3.json", encoding="utf-8") as f:
        return json.load(f)


def test_hungarian_matches_perfect_predictions():
    """Identity labels → purity 1.0 and identity mapping."""
    y = np.array([0, 1, 2, 0, 1, 2])
    mapping, purity = _hungarian_match_labels(y, y)
    assert purity == 1.0
    assert mapping == {0: 0, 1: 1, 2: 2}


def test_hungarian_recovers_permuted_labels():
    """If pred = relabel(true), Hungarian should undo the permutation."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([2, 2, 0, 0, 1, 1])  # perm 0→2, 1→0, 2→1
    mapping, purity = _hungarian_match_labels(y_true, y_pred)
    assert purity == 1.0
    assert mapping[2] == 0
    assert mapping[0] == 1
    assert mapping[1] == 2


def test_hungarian_detects_drift():
    """If half the predictions are wrong, purity ≤ 0.5 (approx)."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])   # random coin flip
    _, purity = _hungarian_match_labels(y_true, y_pred)
    assert purity == pytest.approx(0.5, abs=0.01)


def test_kmeans_on_three_blobs_meets_fixture(three_blobs_spec):
    """KMeans on the three-blobs fixture must recover >= expected purity.

    This is the canonical math-regression anchor. When Wave 3a swaps the
    reference implementation for `app_cluster_pipeline.run_clustering`,
    this test must continue to pass unchanged — any drop in purity is
    a silent math regression.
    """
    from sklearn.cluster import KMeans

    X, y_true = _generate_blobs(
        seed=three_blobs_spec["seed"],
        n_points=three_blobs_spec["n_points"],
        n_features=three_blobs_spec["n_features"],
        n_clusters=three_blobs_spec["n_clusters"],
        centers_scale=three_blobs_spec["centers_scale"],
        blob_std=three_blobs_spec["blob_std"],
    )

    km = KMeans(
        n_clusters=three_blobs_spec["n_clusters"],
        n_init=three_blobs_spec["kmeans_n_init"],
        max_iter=three_blobs_spec["kmeans_max_iter"],
        random_state=three_blobs_spec["seed"],
    )
    y_pred = km.fit_predict(X)

    # Size distribution (sorted descending) should match the fixture up to a
    # small tolerance. With perfect clusters in 16-D these are exact.
    sizes_pred = sorted((int((y_pred == k).sum()) for k in range(three_blobs_spec["n_clusters"])),
                        reverse=True)
    sizes_expected = sorted(three_blobs_spec["expected_cluster_sizes_sorted"], reverse=True)

    _, purity = _hungarian_match_labels(y_true, y_pred)

    if os.environ.get("BRT_CLUSTER_GOLDEN_RECORD") == "1":
        # Record mode: print observed values for fixture authoring.
        sys.stderr.write(
            f"\n[GOLDEN RECORD] name={three_blobs_spec['name']} "
            f"sizes={sizes_pred} purity={purity:.4f}\n"
        )

    assert sizes_pred == sizes_expected, (
        f"Cluster-size distribution drift: got {sizes_pred}, "
        f"fixture {sizes_expected}. Possible math regression in Wave 3a migration."
    )
    assert purity >= three_blobs_spec["expected_purity_min"], (
        f"Cluster purity dropped to {purity:.4f}, fixture requires "
        f"≥ {three_blobs_spec['expected_purity_min']}. See "
        f"docs/adr/0002-pipeline-stages-and-snapshots.md."
    )
