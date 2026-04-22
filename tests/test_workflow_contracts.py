# -*- coding: utf-8 -*-
"""
Contract tests for workflow_contracts.py.

Validates that Pydantic (and manual-fallback) validation accepts valid
snapshots and rejects invalid ones for train/apply/cluster configs.
"""
from __future__ import annotations

import pytest

from workflow_contracts import (
    ApplyWorkflowConfig,
    ClusterWorkflowConfig,
    TrainWorkflowConfig,
)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

VALID_TRAIN = {
    "train_mode": "tfidf",
    "C": 1.0,
    "max_iter": 1000,
    "test_size": 0.2,
}


class TestTrainConfig:
    def test_valid_snapshot(self):
        cfg = TrainWorkflowConfig.from_snapshot(VALID_TRAIN)
        assert cfg.train_mode == "tfidf"
        assert cfg.c_value == 1.0
        assert cfg.use_smote is True
        assert cfg.oversample_strategy == "augment_light"

    def test_missing_required_field_rejected(self):
        payload = dict(VALID_TRAIN)
        del payload["train_mode"]
        with pytest.raises(ValueError):
            TrainWorkflowConfig.from_snapshot(payload)

    def test_negative_c_rejected(self):
        payload = dict(VALID_TRAIN, C=-1.0)
        with pytest.raises(ValueError):
            TrainWorkflowConfig.from_snapshot(payload)

    def test_test_size_above_limit_rejected(self):
        payload = dict(VALID_TRAIN, test_size=0.99)
        with pytest.raises(ValueError):
            TrainWorkflowConfig.from_snapshot(payload)

    def test_non_finite_c_rejected(self):
        payload = dict(VALID_TRAIN, C=float("inf"))
        with pytest.raises(ValueError):
            TrainWorkflowConfig.from_snapshot(payload)

    def test_frozen_instance(self):
        cfg = TrainWorkflowConfig.from_snapshot(VALID_TRAIN)
        with pytest.raises((AttributeError, Exception)):
            cfg.train_mode = "sbert"  # type: ignore[misc]

    def test_optional_defaults_applied(self):
        cfg = TrainWorkflowConfig.from_snapshot(VALID_TRAIN)
        assert cfg.diagnostic_mode is False
        assert cfg.use_smote is True


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

VALID_APPLY = {
    "model_file": "/models/x.joblib",
    "apply_file": "/data/x.xlsx",
    "pred_col": "category",
}


class TestApplyConfig:
    def test_valid_snapshot(self):
        cfg = ApplyWorkflowConfig.from_snapshot(VALID_APPLY)
        assert cfg.model_file == "/models/x.joblib"
        assert cfg.pred_col == "category"
        assert cfg.use_ensemble is False

    def test_empty_model_file_rejected(self):
        payload = dict(VALID_APPLY, model_file="")
        with pytest.raises(ValueError):
            ApplyWorkflowConfig.from_snapshot(payload)

    def test_non_string_model_file_rejected(self):
        payload = dict(VALID_APPLY, model_file=42)
        with pytest.raises(ValueError):
            ApplyWorkflowConfig.from_snapshot(payload)

    def test_missing_pred_col_rejected(self):
        payload = dict(VALID_APPLY)
        del payload["pred_col"]
        with pytest.raises(ValueError):
            ApplyWorkflowConfig.from_snapshot(payload)


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

VALID_CLUSTER = {
    "cluster_algo": "kmeans",
    "cluster_vec_mode": "tfidf",
    "k_clusters": 10,
    "n_init_cluster": 3,
    "cluster_min_df": 2,
    "use_umap": False,
}


class TestClusterConfig:
    def test_valid_snapshot(self):
        cfg = ClusterWorkflowConfig.from_snapshot(VALID_CLUSTER)
        assert cfg.algo == "kmeans"
        assert cfg.k_clusters == 10
        assert cfg.use_umap is False

    def test_unknown_algo_rejected(self):
        payload = dict(VALID_CLUSTER, cluster_algo="xyz")
        with pytest.raises(ValueError):
            ClusterWorkflowConfig.from_snapshot(payload)

    def test_unknown_vec_mode_rejected(self):
        payload = dict(VALID_CLUSTER, cluster_vec_mode="zzz")
        with pytest.raises(ValueError):
            ClusterWorkflowConfig.from_snapshot(payload)

    def test_k_clusters_too_small_rejected(self):
        payload = dict(VALID_CLUSTER, k_clusters=1)
        with pytest.raises(ValueError):
            ClusterWorkflowConfig.from_snapshot(payload)

    def test_k_clusters_too_large_rejected(self):
        payload = dict(VALID_CLUSTER, k_clusters=10_000)
        with pytest.raises(ValueError):
            ClusterWorkflowConfig.from_snapshot(payload)

    def test_defaults_applied(self):
        cfg = ClusterWorkflowConfig.from_snapshot(VALID_CLUSTER)
        assert cfg.use_llm_naming is False
        assert cfg.use_t5_summary is False
        assert cfg.diagnostic_mode is False

    def test_hdbscan_accepted(self):
        payload = dict(VALID_CLUSTER, cluster_algo="hdbscan")
        cfg = ClusterWorkflowConfig.from_snapshot(payload)
        assert cfg.algo == "hdbscan"

    def test_sbert_vec_mode_accepted(self):
        payload = dict(VALID_CLUSTER, cluster_vec_mode="sbert")
        cfg = ClusterWorkflowConfig.from_snapshot(payload)
        assert cfg.cluster_vec_mode == "sbert"
