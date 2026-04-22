"""Сервисный слой кластеризации — чистый ML, без UI (без tkinter).

По образцу app_train_service.py. Предназначен для:
  - вызова из app_cluster.py (UI → snap → этот сервис → результат)
  - написания интеграционных тестов без Tkinter
  - использования из CLI / batch-скриптов

Весь ML-код сосредоточен в app_cluster_pipeline. Этот модуль —
тонкий orchestration-слой, который соединяет стадии и возвращает
структурированный результат.

Phase 14: `progress_cb` теперь действительно вызывается на границе
каждой из пяти стадий (0.00/0.15/0.45/0.70/0.85/1.00), а
`cancel_event` (threading.Event | None) проверяется перед каждой
стадией и поднимает :class:`WorkflowCancelled`.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from app_cluster_pipeline import (
    ClusterResult,
    ExportSummary,
    PostprocessResult,
    PreparedInputs,
    build_vectors,
    export_cluster_outputs,
    postprocess_clusters,
    prepare_inputs,
    run_clustering,
)
from snap_utils import freeze_snap


class WorkflowCancelled(RuntimeError):
    """Raised when `cancel_event.is_set()` is observed between stages."""


@dataclass
class ClusterRunResult:
    """Итог полного запуска кластеризации (все стадии)."""
    export: ExportSummary
    n_clusters: int
    n_noise: int
    labels: Any | None   # np.ndarray | None


_STAGE_BOUNDARIES = (
    (0.00, "Подготовка данных…"),
    (0.15, "Векторизация…"),
    (0.45, "Кластеризация…"),
    (0.70, "Постобработка…"),
    (0.85, "Экспорт…"),
    (1.00, "Готово"),
)


def _check_cancelled(event: threading.Event | None) -> None:
    if event is not None and event.is_set():
        raise WorkflowCancelled("Операция отменена пользователем.")


class ClusteringWorkflow:
    """Orchestrates the four clustering stages without any UI dependency."""

    @staticmethod
    def run(
        files_snapshot: list[str],
        snap: dict[str, Any],
        log_cb: Callable[[str], None] | None = None,
        progress_cb: Callable[[float, str], None] | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> ClusterRunResult:
        """Execute prepare → vectorize → cluster → postprocess → export.

        Parameters
        ----------
        files_snapshot:
            List of absolute file paths to cluster.
        snap:
            Parameter snapshot dict (same shape the web UI serialises via
            ``ui_widgets/session.py:save_session()``).
        log_cb:
            Optional callback receiving log strings (e.g. print).
        progress_cb:
            Optional callback receiving (fraction, message) progress updates.
            Called at the boundary of each of the five stages.
        cancel_event:
            Optional ``threading.Event``. Checked before each stage; raises
            :class:`WorkflowCancelled` if set.
        """
        frozen_snap: Mapping[str, Any] = freeze_snap(snap)

        def _progress(idx: int) -> None:
            if progress_cb is None:
                return
            frac, msg = _STAGE_BOUNDARIES[idx]
            progress_cb(frac, msg)

        _check_cancelled(cancel_event)
        _progress(0)
        if log_cb:
            log_cb("[ClusteringWorkflow] Подготовка данных…")
        prepared: PreparedInputs = prepare_inputs(files_snapshot, frozen_snap)

        _check_cancelled(cancel_event)
        _progress(1)
        if log_cb:
            log_cb("[ClusteringWorkflow] Векторизация…")
        vectors_stage = build_vectors(prepared, frozen_snap)

        _check_cancelled(cancel_event)
        _progress(2)
        if log_cb:
            log_cb("[ClusteringWorkflow] Кластеризация…")
        cluster_stage: ClusterResult = run_clustering(vectors_stage, frozen_snap)

        _check_cancelled(cancel_event)
        _progress(3)
        if log_cb:
            log_cb("[ClusteringWorkflow] Постобработка…")
        post_stage: PostprocessResult = postprocess_clusters(cluster_stage, prepared, frozen_snap)

        _check_cancelled(cancel_event)
        _progress(4)
        if log_cb:
            log_cb("[ClusteringWorkflow] Экспорт…")
        export: ExportSummary = export_cluster_outputs(post_stage, frozen_snap)

        _progress(5)

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
        files_snapshot: list[str],
        snap: dict[str, Any],
    ) -> PreparedInputs:
        """Run only the data-preparation stage (useful for testing)."""
        frozen_snap: Mapping[str, Any] = freeze_snap(snap)
        return prepare_inputs(files_snapshot, frozen_snap)
