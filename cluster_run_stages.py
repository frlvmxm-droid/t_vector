# -*- coding: utf-8 -*-
"""
cluster_run_stages — frozen stage-scoped views of ClusterRunState.

The mutable ``ClusterRunState`` dataclass in ``app_cluster.py`` aggregates 42
fields across four clustering pipeline stages. This module provides frozen
per-stage snapshot types that:

* make stage membership of each field explicit (types + field groups);
* give future service-layer refactors a typed handoff surface
  (Stage1 output  →  Stage2 input, etc.) without rewriting the existing
  worker in one big-bang pass;
* let tests and pipeline-level diagnostics consume a subset of state
  without depending on the mutable aggregate.

Design intent:
  * These are *views*, not replacements. ``ClusterRunState`` remains the
    single source of truth inside the worker thread. ``from_run_state()``
    projects it onto a frozen view at stage boundaries.
  * ``frozen=True`` means downstream code cannot accidentally mutate
    upstream results. Any modification requires ``dataclasses.replace``.
  * Deliberately loose typing (``Any`` for numpy / sklearn objects) to
    avoid a heavy mypy dependency on numpy stubs at this layer; typed
    pipeline stages already exist in ``app_cluster_pipeline.py``.

See ``app_cluster.ClusterRunState`` for the per-field write-stage comments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Stage1InputSnapshot:
    """Frozen view of Stage 1 (file I/O + text preparation) outputs.

    Produced by ``prepare_inputs`` / ``_cluster_prepare_data``; consumed by
    Stage 2 vectorization and, for final row emission, Stage 4 export.
    """

    in_paths: list = field(default_factory=list)
    total_rows: int = 0
    start_ts: float = 0.0
    cluster_snap: Any = None
    use_t5: bool = False
    X_all: list = field(default_factory=list)
    X_clean: list = field(default_factory=list)
    file_data: list = field(default_factory=list)
    raw_texts_all: list = field(default_factory=list)
    n_ok: int = 0


@dataclass(frozen=True)
class Stage2VectorLabelSnapshot:
    """Frozen view of Stage 2 (vectorization + clustering) outputs.

    Produced by ``build_vectors`` + ``run_clustering``. Carries the minimum
    state required by Stage 3 postprocessing: cluster assignments, keywords
    per cluster, and — where applicable — dedup / hierarchical / soft-membership
    structures that downstream naming consults.
    """

    Xv: Any = None
    Xv_tfidf: Any = None
    labels: Any = None
    K: int = 0
    kw: list = field(default_factory=list)
    kw_final: list = field(default_factory=list)
    vec_kw: Any = None
    dedup_map: dict = field(default_factory=dict)
    X_clean_dd: list = field(default_factory=list)
    dedup_reverse_map: Any = None
    labels_l1: Any = None
    noise_labels: Any = None
    km: Any = None
    # алгоритмические флаги, принятые на стадии 2 — неизменны для 3/4
    use_hdbscan: bool = False
    use_lda: bool = False
    use_hier: bool = False
    use_bertopic: bool = False
    use_ensemble: bool = False
    use_gmm: bool = False
    use_fastopic: bool = False
    use_fastopic_kw_ready: bool = False
    stop_list: Any = None
    inc_labels_done: bool = False
    hdbscan_proba: Any = None


@dataclass(frozen=True)
class Stage3PostprocessSnapshot:
    """Frozen view of Stage 3 (naming + summaries + quality) outputs.

    Produced by ``postprocess_clusters``; consumed by Stage 4 export to
    emit per-row labels, reasons, quality scores, and summaries.
    """

    cluster_names: dict = field(default_factory=dict)
    cluster_reasons: dict = field(default_factory=dict)
    cluster_quality: dict = field(default_factory=dict)
    t5_summaries: list = field(default_factory=list)
    cluster_name_map: dict = field(default_factory=dict)
    cluster_reason_map: dict = field(default_factory=dict)
    cluster_quality_map: dict = field(default_factory=dict)
    llm_feedback_map: dict = field(default_factory=dict)
    t5_summaries_all: list = field(default_factory=list)
    kw_dict: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Stage4ExportSnapshot:
    """Frozen view of Stage 4 (artifact / Excel export) metadata.

    Stage 4 is mostly side-effectful (writes to disk); this snapshot
    captures the metadata needed for experiment logs and the summary line.
    """

    stamp: str = ""
    use_streaming: bool = False
    use_inc_model: bool = False
    done: int = 0


def snapshot_stage1(run_state: Any) -> Stage1InputSnapshot:
    """Projects the running ``ClusterRunState`` onto a frozen Stage 1 view."""
    return Stage1InputSnapshot(
        in_paths=list(run_state.in_paths),
        total_rows=int(run_state.total_rows),
        start_ts=float(run_state.start_ts),
        cluster_snap=run_state.cluster_snap,
        use_t5=bool(run_state.use_t5),
        X_all=list(run_state.X_all),
        X_clean=list(run_state.X_clean),
        file_data=list(run_state.file_data),
        raw_texts_all=list(run_state.raw_texts_all),
        n_ok=int(run_state.n_ok),
    )


def snapshot_stage2(run_state: Any) -> Stage2VectorLabelSnapshot:
    """Projects the running ``ClusterRunState`` onto a frozen Stage 2 view."""
    return Stage2VectorLabelSnapshot(
        Xv=run_state.Xv,
        Xv_tfidf=run_state.Xv_tfidf,
        labels=run_state.labels,
        K=int(run_state.K),
        kw=list(run_state.kw),
        kw_final=list(run_state.kw_final),
        vec_kw=run_state.vec_kw,
        dedup_map=dict(run_state.dedup_map),
        X_clean_dd=list(run_state.X_clean_dd),
        dedup_reverse_map=run_state.dedup_reverse_map,
        labels_l1=run_state.labels_l1,
        noise_labels=run_state.noise_labels,
        km=run_state.km,
        use_hdbscan=bool(run_state.use_hdbscan),
        use_lda=bool(run_state.use_lda),
        use_hier=bool(run_state.use_hier),
        use_bertopic=bool(run_state.use_bertopic),
        use_ensemble=bool(run_state.use_ensemble),
        use_gmm=bool(run_state.use_gmm),
        use_fastopic=bool(run_state.use_fastopic),
        use_fastopic_kw_ready=bool(run_state.use_fastopic_kw_ready),
        stop_list=run_state.stop_list,
        inc_labels_done=bool(run_state.inc_labels_done),
        hdbscan_proba=run_state.hdbscan_proba,
    )


def snapshot_stage3(run_state: Any) -> Stage3PostprocessSnapshot:
    """Projects the running ``ClusterRunState`` onto a frozen Stage 3 view."""
    return Stage3PostprocessSnapshot(
        cluster_names=dict(run_state.cluster_names),
        cluster_reasons=dict(run_state.cluster_reasons),
        cluster_quality=dict(run_state.cluster_quality),
        t5_summaries=list(run_state.t5_summaries),
        cluster_name_map=dict(run_state.cluster_name_map),
        cluster_reason_map=dict(run_state.cluster_reason_map),
        cluster_quality_map=dict(run_state.cluster_quality_map),
        llm_feedback_map=dict(run_state.llm_feedback_map),
        t5_summaries_all=list(run_state.t5_summaries_all),
        kw_dict=dict(run_state.kw_dict),
    )


def snapshot_stage4(run_state: Any) -> Stage4ExportSnapshot:
    """Projects the running ``ClusterRunState`` onto a frozen Stage 4 view."""
    return Stage4ExportSnapshot(
        stamp=str(run_state.stamp),
        use_streaming=bool(run_state.use_streaming),
        use_inc_model=bool(run_state.use_inc_model),
        done=int(run_state.done),
    )
