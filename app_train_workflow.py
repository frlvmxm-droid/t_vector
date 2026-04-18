# -*- coding: utf-8 -*-
"""Workflow-слой для вкладки обучения."""
from __future__ import annotations

from workflow_contracts import TrainWorkflowConfig
from workflow_guard import reject_start


def validate_train_preconditions(app) -> bool:
    """Проверяет preconditions перед запуском обучения."""
    if not app.train_files:
        return reject_start(app, title="Обучение", msg="Добавь хотя бы один Excel для обучения.")
    if not app.label_col.get().strip():
        return reject_start(app, title="Обучение", msg="Выбери колонку label/причина.")
    if app.train_mode.get() == "finetune" and not app.base_model_file.get().strip():
        return reject_start(app, title="Дообучение", msg="Выбери базовую модель .joblib.")
    return True


def build_validated_train_snapshot(app):
    """Считывает и валидирует snapshot параметров train workflow."""
    snap = app._snap_params()
    try:
        TrainWorkflowConfig.from_snapshot(snap)
    except Exception as ex:
        reject_start(app, title="Обучение", msg=f"Некорректные параметры: {ex}", kind="error")
        return None
    return snap
