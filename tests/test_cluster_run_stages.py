# -*- coding: utf-8 -*-
"""Tests for cluster_run_stages frozen stage-scoped snapshots.

Validates that the per-stage view types:
  * correctly project a mutable ClusterRunState-like namespace onto
    each stage's subset of fields;
  * are truly frozen (attempts to mutate raise FrozenInstanceError);
  * survive round-trip via dataclasses.replace() — the only supported
    way to derive a new view with modified fields.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from cluster_run_stages import (
    Stage1InputSnapshot,
    Stage2VectorLabelSnapshot,
    Stage3PostprocessSnapshot,
    Stage4ExportSnapshot,
    snapshot_stage1,
    snapshot_stage2,
    snapshot_stage3,
    snapshot_stage4,
)


def _make_full_run_state() -> SimpleNamespace:
    """Builds a SimpleNamespace with every field ClusterRunState exposes."""
    return SimpleNamespace(
        # Stage 1
        in_paths=["a.xlsx", "b.xlsx"],
        total_rows=100,
        start_ts=1234.5,
        cluster_snap={"algo": "kmeans"},
        use_t5=True,
        X_all=["t1", "t2"],
        X_clean=["c1", "c2"],
        file_data=[("a", 1), ("b", 2)],
        raw_texts_all=["r1", "r2"],
        n_ok=2,
        # Stage 2
        Xv="vectors",
        Xv_tfidf="tfidf",
        labels=[0, 1],
        K=2,
        kw=[["k1"], ["k2"]],
        kw_final=[["kf1"], ["kf2"]],
        vec_kw="kw_vec",
        dedup_map={0: 0, 1: 1},
        X_clean_dd=["cd1", "cd2"],
        dedup_reverse_map="rev",
        labels_l1=[0, 0],
        noise_labels=[],
        km="km_model",
        use_hdbscan=True,
        use_lda=False,
        use_hier=False,
        use_bertopic=False,
        use_ensemble=False,
        use_gmm=False,
        use_fastopic=False,
        use_fastopic_kw_ready=False,
        stop_list=["stop"],
        inc_labels_done=True,
        hdbscan_proba=[[0.9, 0.1]],
        # Stage 3
        cluster_names={0: "A", 1: "B"},
        cluster_reasons={0: "reason"},
        cluster_quality={0: 0.8},
        t5_summaries=["s1", "s2"],
        cluster_name_map={0: "A"},
        cluster_reason_map={0: "r"},
        cluster_quality_map={0: 0.9},
        llm_feedback_map={0: "fb"},
        t5_summaries_all=["s1"],
        kw_dict={0: "kw"},
        # Stage 4
        stamp="20260418_120000",
        use_streaming=False,
        use_inc_model=True,
        done=95,
    )


# ---------------------------------------------------------------------------
# snapshot_* projections
# ---------------------------------------------------------------------------


def test_snapshot_stage1_captures_all_stage1_fields():
    rs = _make_full_run_state()
    s1 = snapshot_stage1(rs)
    assert isinstance(s1, Stage1InputSnapshot)
    assert s1.in_paths == ["a.xlsx", "b.xlsx"]
    assert s1.total_rows == 100
    assert s1.start_ts == pytest.approx(1234.5)
    assert s1.use_t5 is True
    assert s1.X_all == ["t1", "t2"]
    assert s1.X_clean == ["c1", "c2"]
    assert s1.file_data == [("a", 1), ("b", 2)]
    assert s1.raw_texts_all == ["r1", "r2"]
    assert s1.n_ok == 2
    assert s1.cluster_snap == {"algo": "kmeans"}


def test_snapshot_stage1_copies_mutable_collections():
    rs = _make_full_run_state()
    s1 = snapshot_stage1(rs)
    # Mutate the source after snapshotting — the snapshot should be unaffected.
    rs.X_all.append("t3")
    rs.in_paths.clear()
    assert s1.X_all == ["t1", "t2"]
    assert s1.in_paths == ["a.xlsx", "b.xlsx"]


def test_snapshot_stage2_captures_all_stage2_fields():
    rs = _make_full_run_state()
    s2 = snapshot_stage2(rs)
    assert isinstance(s2, Stage2VectorLabelSnapshot)
    assert s2.labels == [0, 1]
    assert s2.K == 2
    assert s2.kw == [["k1"], ["k2"]]
    assert s2.use_hdbscan is True
    assert s2.stop_list == ["stop"]
    assert s2.hdbscan_proba == [[0.9, 0.1]]
    assert s2.km == "km_model"


def test_snapshot_stage3_captures_all_stage3_fields():
    rs = _make_full_run_state()
    s3 = snapshot_stage3(rs)
    assert isinstance(s3, Stage3PostprocessSnapshot)
    assert s3.cluster_names == {0: "A", 1: "B"}
    assert s3.cluster_reasons == {0: "reason"}
    assert s3.t5_summaries == ["s1", "s2"]
    assert s3.kw_dict == {0: "kw"}


def test_snapshot_stage4_captures_export_metadata():
    rs = _make_full_run_state()
    s4 = snapshot_stage4(rs)
    assert isinstance(s4, Stage4ExportSnapshot)
    assert s4.stamp == "20260418_120000"
    assert s4.use_streaming is False
    assert s4.use_inc_model is True
    assert s4.done == 95


# ---------------------------------------------------------------------------
# Frozen semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "snapshot_cls",
    [
        Stage1InputSnapshot,
        Stage2VectorLabelSnapshot,
        Stage3PostprocessSnapshot,
        Stage4ExportSnapshot,
    ],
)
def test_snapshots_are_frozen(snapshot_cls):
    inst = snapshot_cls()
    with pytest.raises(FrozenInstanceError):
        # Any field will do — they all raise on assignment.
        next_field = next(iter(snapshot_cls.__dataclass_fields__))
        setattr(inst, next_field, "mutated")


def test_stage1_replace_produces_new_instance_with_override():
    rs = _make_full_run_state()
    s1 = snapshot_stage1(rs)
    s1_bumped = replace(s1, total_rows=500)
    assert s1.total_rows == 100
    assert s1_bumped.total_rows == 500
    # All other fields preserved.
    assert s1_bumped.in_paths == s1.in_paths
    assert s1_bumped.X_all == s1.X_all


# ---------------------------------------------------------------------------
# Default instantiation — used as empty-state sentinels in tests
# ---------------------------------------------------------------------------


def test_default_snapshots_are_empty_and_frozen():
    s1 = Stage1InputSnapshot()
    s2 = Stage2VectorLabelSnapshot()
    s3 = Stage3PostprocessSnapshot()
    s4 = Stage4ExportSnapshot()
    assert s1.total_rows == 0
    assert s1.X_all == []
    assert s2.K == 0
    assert s2.labels is None
    assert s3.cluster_names == {}
    assert s4.stamp == ""


def test_snapshot_stage2_kw_collections_are_copied():
    rs = _make_full_run_state()
    s2 = snapshot_stage2(rs)
    rs.kw.append(["extra"])
    rs.dedup_map[99] = 99
    assert s2.kw == [["k1"], ["k2"]]
    assert 99 not in s2.dedup_map


def test_all_four_snapshots_cover_disjoint_field_sets():
    """Sanity-check: no field appears in more than one snapshot class
    (except the 'use_*' flags that Stage2 owns; Stage4 has a separate set)."""
    f1 = set(Stage1InputSnapshot.__dataclass_fields__)
    f2 = set(Stage2VectorLabelSnapshot.__dataclass_fields__)
    f3 = set(Stage3PostprocessSnapshot.__dataclass_fields__)
    f4 = set(Stage4ExportSnapshot.__dataclass_fields__)
    assert f1.isdisjoint(f2)
    assert f1.isdisjoint(f3)
    assert f2.isdisjoint(f3)
    # Stage 4 is the export/meta bucket — must not overlap with any other.
    assert f4.isdisjoint(f1)
    assert f4.isdisjoint(f2)
    assert f4.isdisjoint(f3)
