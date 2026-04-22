# -*- coding: utf-8 -*-
"""
Unit tests for ml_diagnostics public pure functions.

Covered:
  * clean_training_data
  * dataset_health_checks
  * find_cluster_representative_texts
  * merge_similar_clusters
  * rank_for_active_learning
"""
from __future__ import annotations

import numpy as np
import pytest

from ml_diagnostics import (
    clean_training_data,
    dataset_health_checks,
    find_cluster_representative_texts,
    merge_similar_clusters,
    rank_for_active_learning,
)


# ---------------------------------------------------------------------------
# clean_training_data
# ---------------------------------------------------------------------------

class TestCleanTrainingData:
    def test_removes_duplicates(self):
        X = ["a", "a", "b", "c"]
        y = ["1", "1", "2", "3"]
        Xc, yc, report = clean_training_data(X, y, min_samples_per_class=1)
        assert report["n_duplicates"] == 1
        assert len(Xc) == 3

    def test_counts_conflicts(self):
        X = ["x", "x", "y"]
        y = ["A", "B", "C"]
        _Xc, _yc, report = clean_training_data(X, y, min_samples_per_class=1)
        assert report["n_conflicts"] == 1

    def test_drop_conflicts_removes_rows(self):
        X = ["x", "x", "y"]
        y = ["A", "B", "C"]
        Xc, yc, report = clean_training_data(
            X, y, min_samples_per_class=1, drop_conflicts=True
        )
        assert "x" not in Xc
        assert report["n_conflict_rows_dropped"] == 2

    def test_excludes_rare_classes(self):
        X = ["a", "b", "c", "d", "e"]
        y = ["big", "big", "big", "rare", "rare"]
        Xc, yc, report = clean_training_data(X, y, min_samples_per_class=3)
        assert set(yc) == {"big"}
        assert "rare" in report["excluded_classes"]

    def test_preserves_order(self):
        X = ["z", "y", "x"]
        y = ["A", "B", "C"]
        Xc, _, _ = clean_training_data(X, y, min_samples_per_class=1)
        assert Xc == ["z", "y", "x"]


# ---------------------------------------------------------------------------
# dataset_health_checks
# ---------------------------------------------------------------------------

class TestDatasetHealthChecks:
    def test_empty_dataset_is_fatal(self):
        fatal, _warn = dataset_health_checks({"rows_used": 0}, [])
        assert any("не осталось" in m for m in fatal)

    def test_single_class_is_fatal(self):
        fatal, _warn = dataset_health_checks({"rows_used": 10}, ["A"] * 10)
        assert any("разных классов" in m for m in fatal)

    def test_imbalance_warning(self):
        y = ["A"] * 80 + ["B"] * 20
        _fatal, warn = dataset_health_checks({"rows_used": 100}, y)
        assert any("Дисбаланс" in m for m in warn)

    def test_rare_class_warning(self):
        y = ["A", "A", "A", "B", "C"]
        _fatal, warn = dataset_health_checks({"rows_used": 5}, y)
        assert any("Редких классов" in m for m in warn)

    def test_healthy_dataset_has_no_fatal(self):
        y = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
        fatal, _warn = dataset_health_checks({"rows_used": 30}, y)
        assert fatal == []


# ---------------------------------------------------------------------------
# find_cluster_representative_texts
# ---------------------------------------------------------------------------

class TestFindClusterRepresentativeTexts:
    def test_returns_per_cluster_texts(self):
        texts = ["apple pie", "apple jam", "car fast", "car wheel"]
        vectors = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        )
        labels = np.array([0, 0, 1, 1])
        out = find_cluster_representative_texts(texts, labels, vectors, n_top=2)
        assert set(out.keys()) == {0, 1}
        assert len(out[0]) == 2

    def test_skips_noise_cluster(self):
        texts = ["a", "b", "c"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        labels = np.array([-1, 0, 0])
        out = find_cluster_representative_texts(texts, labels, vectors)
        assert -1 not in out
        assert 0 in out

    def test_truncates_long_texts(self):
        long_text = "x" * 1000
        out = find_cluster_representative_texts(
            [long_text], np.array([0]), np.array([[1.0]])
        )
        assert len(out[0][0]) == 300


# ---------------------------------------------------------------------------
# merge_similar_clusters
# ---------------------------------------------------------------------------

class TestMergeSimilarClusters:
    def test_no_merge_when_below_threshold(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.0, 0.9]])
        labels = np.array([0, 1, 0, 1])
        _new, info = merge_similar_clusters(labels, vectors, threshold=0.99)
        assert info["n_before"] == info["n_after"] == 2
        assert info["merges"] == []

    def test_merges_close_centroids(self):
        vectors = np.array(
            [[1.0, 0.0], [1.0, 0.01], [1.0, 0.02], [1.0, 0.03]]
        )
        labels = np.array([0, 0, 1, 1])
        _new, info = merge_similar_clusters(labels, vectors, threshold=0.95)
        assert info["n_after"] < info["n_before"]
        assert len(info["merges"]) >= 1

    def test_single_cluster_noop(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.1]])
        labels = np.array([0, 0])
        new, info = merge_similar_clusters(labels, vectors)
        assert info["n_before"] == 1
        assert (new == labels).all()

    def test_preserves_noise_label(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.01], [1.0, 0.0], [1.0, 0.02]])
        labels = np.array([0, 0, 1, -1])
        new, _info = merge_similar_clusters(labels, vectors, threshold=0.99)
        assert -1 in set(new.tolist())


# ---------------------------------------------------------------------------
# rank_for_active_learning
# ---------------------------------------------------------------------------

class TestRankForActiveLearning:
    def test_entropy_prefers_uncertain(self):
        proba = np.array([
            [0.99, 0.005, 0.005],
            [0.34, 0.33, 0.33],
            [0.90, 0.08, 0.02],
        ])
        texts = ["confident", "uncertain", "medium"]
        out = rank_for_active_learning(texts, proba, ["A", "B", "C"], top_n=3)
        assert out[0]["text"] == "uncertain"

    def test_least_confident_strategy(self):
        proba = np.array([[0.6, 0.4], [0.99, 0.01]])
        out = rank_for_active_learning(
            ["a", "b"], proba, ["A", "B"], strategy="least_confident", top_n=2
        )
        assert out[0]["text"] == "a"
        assert out[0]["strategy"] == "least_confident"

    def test_margin_strategy(self):
        proba = np.array([[0.5, 0.5], [0.9, 0.1]])
        out = rank_for_active_learning(
            ["tie", "clear"], proba, ["A", "B"], strategy="margin", top_n=2
        )
        assert out[0]["text"] == "tie"

    def test_top_n_limit(self):
        proba = np.array([[0.5, 0.5]] * 10)
        out = rank_for_active_learning(
            [f"t{i}" for i in range(10)], proba, ["A", "B"], top_n=3
        )
        assert len(out) == 3

    def test_per_class_quota(self):
        proba = np.array(
            [[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5
        )
        out = rank_for_active_learning(
            [f"t{i}" for i in range(10)],
            proba,
            ["A", "B"],
            top_n=10,
            per_class_quota=2,
        )
        labels = [r["best_label"] for r in out]
        assert labels.count("A") <= 2
        assert labels.count("B") <= 2

    def test_output_fields(self):
        proba = np.array([[0.7, 0.3]])
        out = rank_for_active_learning(["hello"], proba, ["A", "B"], top_n=1)
        assert out[0].keys() >= {
            "idx", "text", "best_label", "best_prob", "score", "strategy"
        }
