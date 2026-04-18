# -*- coding: utf-8 -*-
"""Pure helpers for train dataset preparation without UI dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DatasetBuildResult:
    texts: list[str]
    labels: list[str]
    dropped_rows: int


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def build_training_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    label_col: str,
    text_cols: Sequence[str],
) -> DatasetBuildResult:
    """Builds normalized train dataset from row dictionaries."""
    texts: list[str] = []
    labels: list[str] = []
    dropped = 0

    for row in rows:
        label = _safe_text(row.get(label_col))
        if not label:
            dropped += 1
            continue

        parts = [_safe_text(row.get(col)) for col in text_cols]
        text = " ".join(p for p in parts if p).strip()
        if not text:
            dropped += 1
            continue

        texts.append(text)
        labels.append(label)

    return DatasetBuildResult(texts=texts, labels=labels, dropped_rows=dropped)
