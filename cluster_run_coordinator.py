# -*- coding: utf-8 -*-
"""Координатор preflight-этапа запуска кластеризации."""
from __future__ import annotations

from typing import Any

from cluster_state_adapter import build_cluster_runtime_snapshot
from app_cluster_workflow import validate_cluster_preconditions


def prepare_cluster_run_context(app: Any):
    """Готовит snapshot и preconditions для run_cluster.

    Возвращает (snap, files_snapshot) или (None, None) при ошибке.
    """
    if not validate_cluster_preconditions(app):
        with app._proc_lock:
            app._processing = False
        return None, None

    snap = build_cluster_runtime_snapshot(app)
    if snap is None:
        with app._proc_lock:
            app._processing = False
        return None, None

    if snap.get("cluster_vec_mode") == "ensemble" and snap.get("cluster_algo") != "kmeans":
        bad_algo = snap["cluster_algo"]
        snap["cluster_algo"] = "kmeans"
        app.cluster_algo.set("kmeans")
        app.log_cluster(
            f"⚠️  Ансамбль несовместим с алгоритмом «{bad_algo}».\n"
            f"   Алгоритм автоматически скорректирован → KMeans.\n"
            f"   Ансамбль выполняет три внутренние KMeans-кластеризации\n"
            f"   (TF-IDF / SBERT-1 / SBERT-2) и выбирает лучшую по Silhouette."
        )

    algo = snap.get("cluster_algo")
    vec_mode = snap.get("cluster_vec_mode")

    if algo == "lda" and snap.get("use_umap"):
        snap["use_umap"] = False
        app.log_cluster("⚠️  UMAP несовместим с LDA (нужна count-матрица) — отключён.")

    if algo in ("lda", "bertopic") and snap.get("use_cosine_cluster"):
        snap["use_cosine_cluster"] = False
        app.log_cluster(
            f"⚠️  Косинусная метрика неприменима к «{algo}» — отключена."
        )

    uses_own_vec = algo in ("lda", "bertopic", "fastopic")
    if snap.get("use_anchors"):
        if uses_own_vec or vec_mode not in ("sbert", "combo") or algo != "kmeans":
            snap["use_anchors"] = False
            app.log_cluster(
                "⚠️  Якоря работают только с SBERT/Combo + KMeans — отключены."
            )

    if not uses_own_vec and vec_mode in ("sbert", "combo", "ensemble") and snap.get("use_lemma_cluster"):
        snap["use_lemma_cluster"] = False
        app.log_cluster(
            "⚠️  Лемматизация бесполезна для SBERT/Combo/Ensemble — отключена."
        )

    return snap, list(app.cluster_files)
