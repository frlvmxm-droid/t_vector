# -*- coding: utf-8 -*-
"""Workflow-слой для вкладки кластеризации."""
from __future__ import annotations

from workflow_contracts import ClusterWorkflowConfig
from workflow_guard import reject_start
from app_cluster_service import encrypt_api_key_for_snapshot


def validate_cluster_preconditions(app) -> bool:
    """Проверяет preconditions перед запуском кластеризации."""
    if not app.cluster_files:
        return reject_start(app, title="Кластеризация", msg="Добавь хотя бы один файл для кластеризации.")
    return True


def build_validated_cluster_snapshot(app):
    """Считывает и валидирует snapshot параметров cluster workflow."""
    snap = app._snap_params()
    try:
        ClusterWorkflowConfig.from_snapshot(snap)
    except Exception as ex:
        reject_start(app, title="Кластеризация", msg=f"Некорректные параметры: {ex}", kind="error")
        return None
    raw_api_key = (snap.get("llm_api_key") or "").strip()
    snap["llm_api_key_encrypted"] = encrypt_api_key_for_snapshot(raw_api_key)
    snap["llm_api_key"] = ""
    return snap
