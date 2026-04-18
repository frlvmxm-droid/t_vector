import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import logging

from cluster_incremental_service import (
    apply_incremental_model,
    assign_labels_by_centers,
    load_and_apply_incremental_model,
)


def test_incremental_apply_happy_path_sets_kw_ready_and_labels():
    vec = TfidfVectorizer()
    vec.fit(["payments limits", "cards limits", "other query"])
    saved = {
        "vectorizer": vec,
        "algo": "kmeans",
        "K": 3,
        "centers": np.zeros((3, len(vec.vocabulary_))),
        "kw": {"0": "payments", 1: "limits", "2": "cards"},
    }
    out = apply_incremental_model(saved, ["payments limits", "cards limits"])
    assert out.vectorizer is vec
    assert out.algo == "kmeans"
    assert out.k_clusters == 3
    assert out.kw == ["payments", "limits", "cards"]
    assert out.use_fastopic_kw_ready is True
    assert len(out.labels) == 2


def test_loader_to_incremental_branch_happy_path(tmp_path: Path):
    vec = TfidfVectorizer()
    vec.fit(["payments limits", "cards limits", "other query"])
    bundle = {
        "schema_version": 1,
        "artifact_type": "cluster_model_bundle",
        "vectorizer": vec,
        "algo": "kmeans",
        "K": 3,
        "centers": np.zeros((3, len(vec.vocabulary_))),
        "kw": {"0": "payments", 1: "limits", "2": "cards"},
    }
    p = tmp_path / "inc_bundle.joblib"
    joblib.dump(bundle, p)
    out = load_and_apply_incremental_model(str(p), schema_version=1, texts=["payments limits"])
    assert out.use_fastopic_kw_ready is True
    assert len(out.labels) == 1


def test_assign_labels_by_centers_uses_batch_mode_for_large_matrices():
    xv = np.zeros((10, 4), dtype=float)
    centers = np.zeros((3, 4), dtype=float)
    labels = assign_labels_by_centers(xv, centers, batch_size=3, max_distance_cells=5)
    assert labels.shape == (10,)


def test_incremental_flow_emits_structured_logs_and_diagnostic_report(tmp_path: Path, caplog):
    vec = TfidfVectorizer()
    vec.fit(["payments limits", "cards limits", "other query"])
    bundle = {
        "schema_version": 1,
        "artifact_type": "cluster_model_bundle",
        "vectorizer": vec,
        "algo": "kmeans",
        "K": 3,
        "centers": np.zeros((3, len(vec.vocabulary_))),
        "kw": {"0": "payments", 1: "limits", "2": "cards"},
    }
    p = tmp_path / "inc_bundle_diag.joblib"
    joblib.dump(bundle, p)
    caplog.set_level(logging.INFO)
    report = tmp_path / "diag.json"
    logger = logging.getLogger("inc_diag_test")
    out = load_and_apply_incremental_model(
        str(p),
        schema_version=1,
        texts=["payments limits"],
        logger=logger,
        diagnostic_mode=True,
        diagnostic_report_path=str(report),
        correlation_id="cid-test-1",
    )
    assert len(out.labels) == 1
    assert report.exists()
    assert "event=cluster.incremental" in caplog.text
