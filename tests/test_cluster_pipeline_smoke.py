"""Smoke tests for the pipeline adapter module.

Two contracts are exercised here:

1. `prepare_inputs` is a real, pure function (no ML math) — it runs
   headlessly and returns a `PreparedInputs` dataclass.

2. `build_vectors` / `run_clustering` / `postprocess_clusters` /
   `export_cluster_outputs` are intentional scaffolding for the
   Wave 3a math migration. Until that wave lands, they must raise
   `NotImplementedError` rather than silently return `vectors=None`
   — the prior no-op mimicked the real shape of a call and hid the
   fact that no computation happens. See
   docs/adr/0002-pipeline-stages-and-snapshots.md.
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


def test_build_vectors_raises_not_implemented():
    """Wave 3a tracker: remove this test and replace with a real assertion
    once the port of run_cluster()'s Stage-2 block lands."""
    with pytest.raises(NotImplementedError, match="build_vectors"):
        build_vectors(prepared=None, snap={})  # type: ignore[arg-type]


def test_run_clustering_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="run_clustering"):
        run_clustering(vectors=None, snap={})  # type: ignore[arg-type]


def test_postprocess_clusters_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="postprocess_clusters"):
        postprocess_clusters(result=None, prepared=None, snap={})  # type: ignore[arg-type]


def test_export_cluster_outputs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="export_cluster_outputs"):
        export_cluster_outputs(postprocessed=None, snap={})  # type: ignore[arg-type]
