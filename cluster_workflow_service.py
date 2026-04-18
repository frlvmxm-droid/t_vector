# -*- coding: utf-8 -*-
"""Сервисный слой кластеризации — чистый ML, без UI (без tkinter).

По образцу app_train_service.py. Предназначен для:
  - вызова из app_cluster.py (UI → snap → этот сервис → результат)
  - написания интеграционных тестов без Tkinter
  - использования из CLI / batch-скриптов

Весь ML-код сосредоточен в app_cluster_pipeline. Этот модуль —
тонкий orchestration-слой, который соединяет стадии и возвращает
структурированный результат.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app_cluster_pipeline import (
    PreparedInputs,
    ClusterResult,
    PostprocessResult,
    ExportSummary,
    prepare_inputs,
    build_vectors,
    run_clustering,
    postprocess_clusters,
    export_cluster_outputs,
)


@dataclass
class ClusterRunResult:
    """Итог полного запуска кластеризации (все стадии)."""
    export: ExportSummary
    n_clusters: int
    n_noise: int
    labels: Optional[Any]   # np.ndarray | None


class ClusteringWorkflow:
    """Orchestrates the four clustering stages without any UI dependency."""

    @staticmethod
    def run(
        files_snapshot: List[str],
        snap: Dict[str, Any],
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> ClusterRunResult:
        """Execute prepare → vectorize → cluster → postprocess → export.

        Parameters
        ----------
        files_snapshot:
            List of absolute file paths to cluster.
        snap:
            Parameter snapshot dict (same format as app.py _snap_params()).
        log_cb:
            Optional callback receiving log strings (e.g. print).
        progress_cb:
            Optional callback receiving (fraction, message) progress updates.

        Returns
        -------
        ClusterRunResult with export summary and top-level statistics.
        """
        if log_cb:
            log_cb("[ClusteringWorkflow] Подготовка данных…")
        prepared: PreparedInputs = prepare_inputs(files_snapshot, snap)

        if log_cb:
            log_cb("[ClusteringWorkflow] Векторизация…")
        vectors_stage = build_vectors(prepared, snap)

        if log_cb:
            log_cb("[ClusteringWorkflow] Кластеризация…")
        cluster_stage: ClusterResult = run_clustering(vectors_stage, snap)

        if log_cb:
            log_cb("[ClusteringWorkflow] Постобработка…")
        post_stage: PostprocessResult = postprocess_clusters(cluster_stage, prepared, snap)

        if log_cb:
            log_cb("[ClusteringWorkflow] Экспорт…")
        export: ExportSummary = export_cluster_outputs(post_stage, snap)

        labels = cluster_stage.labels if cluster_stage else None
        n_clusters = 0
        n_noise = 0
        if labels is not None:
            try:
                import numpy as _np
                arr = _np.asarray(labels)
                n_noise = int((arr < 0).sum())
                n_clusters = int(arr.max()) + 1 if (arr >= 0).any() else 0
            except Exception:
                pass

        if log_cb:
            log_cb(f"[ClusteringWorkflow] Готово: K={n_clusters}, шум={n_noise}")

        return ClusterRunResult(
            export=export,
            n_clusters=n_clusters,
            n_noise=n_noise,
            labels=labels,
        )

    @staticmethod
    def prepare_only(
        files_snapshot: List[str],
        snap: Dict[str, Any],
    ) -> PreparedInputs:
        """Run only the data-preparation stage (useful for testing)."""
        return prepare_inputs(files_snapshot, snap)
