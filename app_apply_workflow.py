# -*- coding: utf-8 -*-
"""Workflow-слой для вкладки классификации."""
from __future__ import annotations

from workflow_contracts import ApplyWorkflowConfig
from workflow_guard import reject_start


def validate_apply_preconditions(app) -> bool:
    """Проверяет preconditions перед запуском классификации."""
    model = app.model_file.get().strip()
    xlsx = app.apply_file.get().strip()
    if not model:
        return reject_start(app, title="Классификация", msg="Выбери модель .joblib.")
    if not xlsx:
        return reject_start(app, title="Классификация", msg="Выбери Excel файл.")
    return True


def build_validated_apply_snapshot(app):
    """Считывает и валидирует snapshot параметров apply workflow."""
    snap = app._snap_params()
    # Apply-специфичные пути не входят в общий _snap_params — добавляем явно
    snap["model_file"] = app.model_file.get().strip()
    snap["apply_file"] = app.apply_file.get().strip()
    try:
        ApplyWorkflowConfig.from_snapshot(snap)
    except Exception as ex:
        reject_start(app, title="Классификация", msg=f"Некорректные параметры: {ex}", kind="error")
        return None
    return snap
