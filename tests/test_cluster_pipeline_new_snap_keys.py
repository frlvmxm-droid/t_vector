"""Integration tests for the Phase 15 snap-key wiring in app_cluster_pipeline.

Each test drives the TF-IDF + KMeans slice end-to-end on a tmp CSV and
verifies that the snap-key-gated stages surface expected outputs:

  * ``compute_quality_metrics`` → ExportSummary carries quality dict.
  * ``merge_similar_clusters``  → PostprocessResult carries merge_info.
  * ``hdbscan_reclust``         → only fires for algo='hdbscan'; the
    TF-IDF + KMeans slice just verifies the snap is accepted.

UMAP is already covered by unit tests; wiring it to the TF-IDF path
requires vec_mode in {sbert, combo, ensemble}, so it's skipped here.
"""
from __future__ import annotations

import csv
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

pytest.importorskip("sklearn")
pytest.importorskip("numpy")

from app_cluster_pipeline import (  # noqa: E402
    build_vectors,
    export_cluster_outputs,
    postprocess_clusters,
    prepare_inputs,
    run_clustering,
)


def _three_topic_corpus(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a CSV with three well-separated text clusters."""
    path = tmp_path / "corpus.csv"
    rows = [
        ("text",),
        ("кредит карта платёж банк",),
        ("кредит кредит банк платёж",),
        ("кредит карта платёж платёж",),
        ("кредит карта кредит платёж",),
        ("кредит банк банк платёж",),
        ("вклад депозит процент процент",),
        ("вклад депозит ставка депозит",),
        ("вклад вклад процент депозит",),
        ("депозит депозит ставка вклад",),
        ("вклад процент ставка ставка",),
        ("валюта обмен курс доллар",),
        ("валюта доллар курс обмен",),
        ("обмен валюта курс евро",),
        ("валюта евро обмен курс",),
        ("доллар курс валюта обмен",),
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for r in rows:
            writer.writerow(r)
    return path


def _base_snap(input_path: pathlib.Path, output_path: pathlib.Path) -> dict:
    return {
        "cluster_algo": "kmeans",
        "cluster_vec_mode": "tfidf",
        "cluster_role_mode": "all",
        "ignore_chatbot_cluster": False,
        "text_col": "text",
        "call_col": "text",
        "chat_col": "",
        "k_clusters": 3,
        "n_init_cluster": 3,
        "cluster_min_df": 1,
        "cluster_max_features": 200,
        "random_state": 42,
        "output_path": str(output_path),
    }


def test_compute_quality_metrics_appears_in_export(tmp_path: pathlib.Path) -> None:
    input_path = _three_topic_corpus(tmp_path)
    output_path = tmp_path / "out.csv"
    snap = _base_snap(input_path, output_path) | {"compute_quality_metrics": True}

    prepared = prepare_inputs([str(input_path)], snap)
    vectors = build_vectors(prepared, snap)
    clustered = run_clustering(vectors, snap)
    post = postprocess_clusters(clustered, prepared, snap)
    export = export_cluster_outputs(post, snap)

    outputs = export.outputs or {}
    assert "quality_metrics" in outputs
    quality = outputs["quality_metrics"]
    assert set(quality.keys()) == {"silhouette", "calinski_harabasz", "davies_bouldin"}
    # Well-separated toy corpus → non-degenerate silhouette.
    assert quality["silhouette"] is not None


def test_merge_similar_clusters_emits_merge_info(tmp_path: pathlib.Path) -> None:
    input_path = _three_topic_corpus(tmp_path)
    output_path = tmp_path / "out.csv"
    # Force merge by demanding high K and a low threshold so similar
    # clusters collapse into their neighbours.
    snap = _base_snap(input_path, output_path) | {
        "k_clusters": 6,
        "merge_similar_clusters": True,
        "merge_threshold": 0.3,
    }

    prepared = prepare_inputs([str(input_path)], snap)
    vectors = build_vectors(prepared, snap)
    clustered = run_clustering(vectors, snap)
    post = postprocess_clusters(clustered, prepared, snap)
    export = export_cluster_outputs(post, snap)

    payload = post.payload or {}
    assert "merge_info" in payload
    merge_info = payload["merge_info"]
    # ml_diagnostics.merge_similar_clusters returns {merges, n_before, n_after}.
    assert "n_before" in merge_info
    assert "n_after" in merge_info
    assert merge_info["n_after"] <= merge_info["n_before"]

    outputs = export.outputs or {}
    assert "merge_info" in outputs


def test_hdbscan_reclust_snap_key_accepted(tmp_path: pathlib.Path) -> None:
    """snap['hdbscan_reclust'] is harmless for non-HDBSCAN algos."""
    input_path = _three_topic_corpus(tmp_path)
    output_path = tmp_path / "out.csv"
    snap = _base_snap(input_path, output_path) | {"hdbscan_reclust": True}

    prepared = prepare_inputs([str(input_path)], snap)
    vectors = build_vectors(prepared, snap)
    clustered = run_clustering(vectors, snap)
    post = postprocess_clusters(clustered, prepared, snap)
    export_cluster_outputs(post, snap)
    # No assertion on merge — the key is simply non-fatal here.
    assert clustered.labels is not None


def test_use_umap_snap_key_silently_skipped_for_tfidf(tmp_path: pathlib.Path) -> None:
    """UMAP wiring gates on vec_mode ∈ {sbert, combo, ensemble}; TF-IDF is untouched."""
    input_path = _three_topic_corpus(tmp_path)
    output_path = tmp_path / "out.csv"
    snap = _base_snap(input_path, output_path) | {"use_umap": True}

    prepared = prepare_inputs([str(input_path)], snap)
    vectors = build_vectors(prepared, snap)
    clustered = run_clustering(vectors, snap)
    post = postprocess_clusters(clustered, prepared, snap)
    export_cluster_outputs(post, snap)
    # The TF-IDF vectorisation should not carry the umap_applied marker.
    meta = vectors.meta or {}
    assert meta.get("umap_applied") is None
    assert clustered.labels is not None
