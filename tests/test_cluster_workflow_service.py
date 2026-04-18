# -*- coding: utf-8 -*-
"""Unit tests for cluster_workflow_service.py.

Tests verify the orchestration logic and data contracts without running
a full clustering pipeline (which requires large files and optional deps).
Uses mocks for the pipeline stages.
"""
from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from cluster_workflow_service import ClusteringWorkflow, ClusterRunResult


# ---------------------------------------------------------------------------
# Helpers — mock pipeline stage results
# ---------------------------------------------------------------------------

def _make_mock_stages():
    """Return mock objects for each pipeline stage."""
    mock_prepared = MagicMock(name="PreparedInputs")
    mock_vectors = MagicMock(name="VectorPack")
    mock_cluster = MagicMock(name="ClusterResult")
    mock_cluster.labels = None  # no real labels
    mock_post = MagicMock(name="PostprocessResult")
    mock_export = MagicMock(name="ExportSummary")
    return mock_prepared, mock_vectors, mock_cluster, mock_post, mock_export


# ---------------------------------------------------------------------------
# ClusteringWorkflow.run
# ---------------------------------------------------------------------------

def test_run_calls_all_four_stages():
    """All four pipeline stages should be called in order."""
    prepared, vectors, cluster, post, export = _make_mock_stages()

    with patch("cluster_workflow_service.prepare_inputs", return_value=prepared) as p1, \
         patch("cluster_workflow_service.build_vectors", return_value=vectors) as p2, \
         patch("cluster_workflow_service.run_clustering", return_value=cluster) as p3, \
         patch("cluster_workflow_service.postprocess_clusters", return_value=post) as p4, \
         patch("cluster_workflow_service.export_cluster_outputs", return_value=export) as p5:

        result = ClusteringWorkflow.run(
            files_snapshot=["/some/file.xlsx"],
            snap={"cluster_algo": "kmeans", "k_clusters": 5},
        )

    p1.assert_called_once()
    p2.assert_called_once()
    p3.assert_called_once()
    p4.assert_called_once()
    p5.assert_called_once()
    assert isinstance(result, ClusterRunResult)
    assert result.export is export


def test_run_passes_snap_to_stages():
    """snap dict should be forwarded to build_vectors and run_clustering."""
    snap = {"cluster_algo": "hdbscan", "cluster_vec_mode": "tfidf"}
    prepared, vectors, cluster, post, export = _make_mock_stages()

    with patch("cluster_workflow_service.prepare_inputs", return_value=prepared), \
         patch("cluster_workflow_service.build_vectors", return_value=vectors) as p_bv, \
         patch("cluster_workflow_service.run_clustering", return_value=cluster) as p_rc, \
         patch("cluster_workflow_service.postprocess_clusters", return_value=post), \
         patch("cluster_workflow_service.export_cluster_outputs", return_value=export):

        ClusteringWorkflow.run(files_snapshot=[], snap=snap)

    # snap is passed as second positional arg to build_vectors and run_clustering
    assert p_bv.call_args.args[1] == snap
    assert p_rc.call_args.args[1] == snap


def test_run_log_callback_called():
    """log_cb should receive at least one message per stage."""
    prepared, vectors, cluster, post, export = _make_mock_stages()
    messages = []

    with patch("cluster_workflow_service.prepare_inputs", return_value=prepared), \
         patch("cluster_workflow_service.build_vectors", return_value=vectors), \
         patch("cluster_workflow_service.run_clustering", return_value=cluster), \
         patch("cluster_workflow_service.postprocess_clusters", return_value=post), \
         patch("cluster_workflow_service.export_cluster_outputs", return_value=export):

        ClusteringWorkflow.run([], {}, log_cb=messages.append)

    assert len(messages) >= 4  # at least one per stage


def test_run_returns_cluster_run_result():
    prepared, vectors, cluster, post, export = _make_mock_stages()

    with patch("cluster_workflow_service.prepare_inputs", return_value=prepared), \
         patch("cluster_workflow_service.build_vectors", return_value=vectors), \
         patch("cluster_workflow_service.run_clustering", return_value=cluster), \
         patch("cluster_workflow_service.postprocess_clusters", return_value=post), \
         patch("cluster_workflow_service.export_cluster_outputs", return_value=export):

        result = ClusteringWorkflow.run([], {})

    assert isinstance(result, ClusterRunResult)
    assert result.n_clusters == 0   # cluster.labels is None → 0
    assert result.n_noise == 0
    assert result.labels is None


def test_run_computes_n_clusters_from_labels():
    """n_clusters and n_noise should be computed from labels array when numpy available."""
    pytest.importorskip("numpy")
    import numpy as np

    prepared, vectors, cluster, post, export = _make_mock_stages()
    cluster.labels = np.array([0, 0, 1, 1, 2, -1, -1])  # 3 clusters, 2 noise

    with patch("cluster_workflow_service.prepare_inputs", return_value=prepared), \
         patch("cluster_workflow_service.build_vectors", return_value=vectors), \
         patch("cluster_workflow_service.run_clustering", return_value=cluster), \
         patch("cluster_workflow_service.postprocess_clusters", return_value=post), \
         patch("cluster_workflow_service.export_cluster_outputs", return_value=export):

        result = ClusteringWorkflow.run([], {})

    assert result.n_clusters == 3
    assert result.n_noise == 2


# ---------------------------------------------------------------------------
# ClusteringWorkflow.prepare_only
# ---------------------------------------------------------------------------

def test_prepare_only_calls_prepare_inputs():
    prepared = MagicMock(name="PreparedInputs")
    snap = {"cluster_algo": "kmeans"}

    with patch("cluster_workflow_service.prepare_inputs", return_value=prepared) as mock_prep:
        result = ClusteringWorkflow.prepare_only(["/f.xlsx"], snap)

    mock_prep.assert_called_once_with(["/f.xlsx"], snap)
    assert result is prepared
