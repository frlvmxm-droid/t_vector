"""Smoke tests for the pipeline adapter module.

Two contracts are exercised here:

1. `prepare_inputs` is a real, pure function (no ML math) — it runs
   headlessly and returns a `PreparedInputs` dataclass.

2. After Wave 3a slice port, `build_vectors` / `run_clustering` /
   `postprocess_clusters` / `export_cluster_outputs` are real for the
   `tfidf` + `kmeans` combo. Other combinations still raise
   `NotImplementedError` (the slice port is intentionally narrow —
   ADR-0002 / ADR-0007 track the full migration).
"""
import pytest

from app_cluster_pipeline import (
    build_vectors,
    export_cluster_outputs,
    postprocess_clusters,
    prepare_inputs,
    run_clustering,
)


def test_prepare_inputs_is_real():
    snap = {
        "cluster_role_mode": "all",
        "ignore_chatbot_cluster": True,
        "call_col": "call",
        "chat_col": "chat",
    }
    files_snapshot = ["a.xlsx", "b.xlsx"]
    prepared = prepare_inputs(files_snapshot, snap)
    assert prepared.files_snapshot == files_snapshot
    assert prepared.snap.get("call_col") == "call"


_UNSUPPORTED_SNAP = {"cluster_vec_mode": "sbert", "cluster_algo": "hdbscan"}


def test_build_vectors_unsupported_combo_raises():
    with pytest.raises(NotImplementedError, match="Wave 3a slice"):
        build_vectors(prepared=None, snap=_UNSUPPORTED_SNAP)  # type: ignore[arg-type]


def test_run_clustering_unsupported_combo_raises():
    with pytest.raises(NotImplementedError, match="Wave 3a slice"):
        run_clustering(vectors=None, snap=_UNSUPPORTED_SNAP)  # type: ignore[arg-type]


def test_postprocess_clusters_unsupported_combo_raises():
    with pytest.raises(NotImplementedError, match="Wave 3a slice"):
        postprocess_clusters(  # type: ignore[arg-type]
            result=None, prepared=None, snap=_UNSUPPORTED_SNAP,
        )


def test_export_cluster_outputs_unsupported_combo_raises():
    with pytest.raises(NotImplementedError, match="Wave 3a slice"):
        export_cluster_outputs(postprocessed=None, snap=_UNSUPPORTED_SNAP)  # type: ignore[arg-type]
