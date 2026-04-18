# -*- coding: utf-8 -*-
"""Error classification and recovery policy for model loading/apply flows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exceptions import ModelLoadError, SchemaError


@dataclass(frozen=True)
class ErrorPolicyDecision:
    category: str
    recoverable: bool
    user_hint: str
    incident_marker: str


def classify_error(exc: BaseException) -> ErrorPolicyDecision:
    """Maps exceptions to unified recovery strategy buckets."""
    if isinstance(exc, SchemaError):
        return ErrorPolicyDecision(
            category="schema_error",
            recoverable=False,
            user_hint="Артефакт несовместим по схеме. Требуется миграция/переобучение модели.",
            incident_marker="MODEL_SCHEMA_HARD_FAIL",
        )
    if isinstance(exc, ModelLoadError):
        return ErrorPolicyDecision(
            category="model_load_error",
            recoverable=True,
            user_hint="Не удалось загрузить модель. Проверьте файл и попробуйте fallback-сценарий.",
            incident_marker="MODEL_LOAD_RECOVERABLE",
        )
    return ErrorPolicyDecision(
        category="unexpected_error",
        recoverable=False,
        user_hint="Непредвиденная ошибка. Передайте лог инженерам поддержки.",
        incident_marker="UNEXPECTED_INCIDENT",
    )


def log_structured_event(
    logger: Any,
    *,
    event: str,
    stage: str,
    file: str,
    rows: int | None,
    duration_sec: float | None,
    error_class: str | None,
    correlation_id: str | None,
) -> None:
    """Logs structured operational telemetry in key=value style."""
    if not logger:
        return
    logger.info(
        "event=%s stage=%s file=%s rows=%s duration_sec=%s error_class=%s correlation_id=%s",
        event,
        stage,
        file,
        rows,
        duration_sec,
        error_class,
        correlation_id,
    )
