# -*- coding: utf-8 -*-
"""Сервис incremental-ветки кластеризации (без tkinter-зависимостей)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
from scipy import sparse as sp
from sklearn.metrics import pairwise_distances_argmin

from exceptions import ModelLoadError
from cluster_model_loader import load_cluster_model_bundle, normalize_incremental_bundle_payload
from diagnostics_report import build_diagnostic_report, export_diagnostic_report
from model_error_policy import classify_error, log_structured_event


@dataclass(frozen=True)
class IncrementalApplyResult:
    vectorizer: Any
    algo: str
    k_clusters: int
    kw: list[str]
    use_fastopic_kw_ready: bool
    xv: Any
    labels: np.ndarray


def _estimate_cells(xv: Any, centers: Any) -> int:
    n_samples = int(getattr(xv, "shape", [0, 0])[0])
    n_centers = int(getattr(centers, "shape", [0, 0])[0])
    return n_samples * n_centers


def assign_labels_by_centers(
    xv: Any,
    centers: Any,
    *,
    batch_size: int = 4096,
    max_distance_cells: int = 5_000_000,
) -> np.ndarray:
    """Assign labels with guardrails and batch mode for large matrices."""
    est_cells = _estimate_cells(xv, centers)
    if est_cells <= 0:
        return np.asarray([], dtype=int)

    if est_cells <= max_distance_cells:
        return np.asarray(pairwise_distances_argmin(xv, centers), dtype=int)

    labels: list[np.ndarray] = []
    n_samples = int(getattr(xv, "shape", [0, 0])[0])
    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        xb = xv[start:end]
        if sp.issparse(xb):
            xb = xb.tocsr()
        labels.append(np.asarray(pairwise_distances_argmin(xb, centers), dtype=int))
    return np.concatenate(labels, axis=0) if labels else np.asarray([], dtype=int)


def apply_incremental_model(saved: Mapping[str, Any], texts: Sequence[str]) -> IncrementalApplyResult:
    """Применяет сохранённую incremental-модель к очищенным текстам."""
    bundle = normalize_incremental_bundle_payload(saved)
    xv = bundle.vectorizer.transform(list(texts))

    if bundle.algo == "kmeans" and bundle.centers is not None:
        labels = assign_labels_by_centers(xv, bundle.centers)
    elif bundle.algo == "hdbscan" and bundle.model is not None:
        labels = np.asarray(bundle.model.predict(xv), dtype=int)
    elif bundle.centers is not None:
        labels = assign_labels_by_centers(xv, bundle.centers)
    else:
        raise ModelLoadError("Инкрементальная модель не содержит центров/предиктора для разметки.")

    return IncrementalApplyResult(
        vectorizer=bundle.vectorizer,
        algo=bundle.algo,
        k_clusters=bundle.k_clusters,
        kw=bundle.kw,
        use_fastopic_kw_ready=bundle.use_fastopic_kw_ready,
        xv=xv,
        labels=labels,
    )


def load_and_apply_incremental_model(
    model_path: str,
    *,
    schema_version: int,
    texts: Sequence[str],
    logger: Any = None,
    correlation_id: str | None = None,
    diagnostic_mode: bool = False,
    diagnostic_report_path: str | None = None,
    trusted_paths: Sequence[str] | None = None,
) -> IncrementalApplyResult:
    cid = correlation_id or uuid4().hex[:12]
    log_structured_event(
        logger,
        event="cluster.incremental",
        stage="load",
        file=model_path,
        rows=len(texts),
        duration_sec=None,
        error_class=None,
        correlation_id=cid,
    )
    try:
        saved = load_cluster_model_bundle(
            model_path,
            schema_version=schema_version,
            trusted_paths=trusted_paths,
            logger=logger,
        )
        out = apply_incremental_model(saved, texts)
        log_structured_event(
            logger,
            event="cluster.incremental",
            stage="apply",
            file=model_path,
            rows=len(texts),
            duration_sec=None,
            error_class=None,
            correlation_id=cid,
        )
        if diagnostic_mode and diagnostic_report_path:
            report = build_diagnostic_report(
                correlation_id=cid,
                event="cluster.incremental",
                stage="completed",
                metrics={"rows": len(texts), "k_clusters": out.k_clusters, "algo": out.algo},
                snapshot={"model_path": model_path, "schema_version": schema_version},
            )
            export_diagnostic_report(report, diagnostic_report_path)
        return out
    except Exception as exc:
        dec = classify_error(exc)
        log_structured_event(
            logger,
            event="cluster.incremental",
            stage="failed",
            file=model_path,
            rows=len(texts),
            duration_sec=None,
            error_class=type(exc).__name__,
            correlation_id=cid,
        )
        if diagnostic_mode and diagnostic_report_path:
            report = build_diagnostic_report(
                correlation_id=cid,
                event="cluster.incremental",
                stage=dec.category,
                metrics={"rows": len(texts), "recoverable": dec.recoverable},
                snapshot={"model_path": model_path, "schema_version": schema_version},
                error=exc,
            )
            export_diagnostic_report(report, diagnostic_report_path)
        raise
