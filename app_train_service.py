# -*- coding: utf-8 -*-
"""Сервисный слой обучения (без UI)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
from sklearn.pipeline import Pipeline

from ml_training import TrainingOptions, train_model


class TrainingWorkflow:
    """Тонкий orchestration-сервис для train pipeline."""

    def fit_and_evaluate(
        self,
        X: List[str],
        y: List[str],
        features: Any,
        C: float,
        max_iter: int,
        balanced: bool,
        test_size: float,
        random_state: int,
        *,
        options: TrainingOptions,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Pipeline, str, str, Optional[List[str]], Optional[Any], Dict[str, Any]]:
        return train_model(
            X=X,
            y=y,
            features=features,
            C=C,
            max_iter=max_iter,
            balanced=balanced,
            test_size=test_size,
            random_state=random_state,
            options=options,
            log_cb=log_cb,
            progress_cb=progress_cb,
        )

    @staticmethod
    def persist_artifact(payload: Dict[str, Any], model_path: str) -> None:
        joblib.dump(payload, model_path, compress=3)
