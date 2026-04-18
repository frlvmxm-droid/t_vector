# -*- coding: utf-8 -*-
"""Сервисный слой обучения (без UI)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
from sklearn.pipeline import Pipeline

from ml_training import train_model


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
        calib_method: str,
        test_size: float,
        random_state: int,
        use_smote: bool,
        oversample_strategy: str = "augment_light",
        max_dup_per_sample: int = 5,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        run_cv: bool = False,
        use_hard_negatives: bool = False,
        use_field_dropout: bool = False,
        field_dropout_prob: float = 0.15,
        field_dropout_copies: int = 2,
        use_label_smoothing: bool = False,
        label_smoothing_eps: float = 0.05,
    ) -> Tuple[Pipeline, str, str, Optional[List[str]], Optional[Any], Dict[str, Any]]:
        return train_model(
            X=X,
            y=y,
            features=features,
            C=C,
            max_iter=max_iter,
            balanced=balanced,
            calib_method=calib_method,
            test_size=test_size,
            random_state=random_state,
            use_smote=use_smote,
            oversample_strategy=oversample_strategy,
            max_dup_per_sample=max_dup_per_sample,
            log_cb=log_cb,
            progress_cb=progress_cb,
            run_cv=run_cv,
            use_hard_negatives=use_hard_negatives,
            use_field_dropout=use_field_dropout,
            field_dropout_prob=field_dropout_prob,
            field_dropout_copies=field_dropout_copies,
            use_label_smoothing=use_label_smoothing,
            label_smoothing_eps=label_smoothing_eps,
        )

    @staticmethod
    def persist_artifact(payload: Dict[str, Any], model_path: str) -> None:
        joblib.dump(payload, model_path, compress=3)
