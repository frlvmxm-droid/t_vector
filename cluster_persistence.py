# -*- coding: utf-8 -*-
"""Сохранение артефактов кластеризации."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib


class ClusterModelPersistence:
    """Сервис сохранения артефакта модели кластеризации (без UI-зависимостей)."""

    @staticmethod
    def normalize_model_path(model_path: str, default_path: str) -> str:
        resolved = (model_path or "").strip() or default_path
        if not resolved.lower().endswith(".joblib"):
            resolved = f"{resolved}.joblib"
        return resolved

    @staticmethod
    def save_bundle(bundle: Dict[str, Any], model_path: str) -> str:
        out = str(Path(model_path))
        joblib.dump(bundle, out, compress=3)
        return out
