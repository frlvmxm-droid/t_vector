# -*- coding: utf-8 -*-
"""Prediction helpers for apply flow (thresholding, contract checks)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np

from artifact_contracts import TRAIN_MODEL_ARTIFACT_TYPE, validate_bundle_schema


@dataclass
class ApplyRunState:
    """Mutable run-state bag for a single apply (classification) operation.

    Passed between ``_apply_load_models``, ``worker`` (flush_chunk), and
    ``_apply_postprocess`` so that each phase reads/writes a shared object
    rather than scattered local variables.
    """

    # ── Model & inference ────────────────────────────────────────────────
    pipe: Any = None
    pipe2: Any = None
    pkg: Dict = field(default_factory=dict)
    has_proba: bool = False
    has_proba2: bool = False
    classes: Any = None
    classes2: Any = None
    classes_map2: Any = None
    per_cls_thr: Dict = field(default_factory=dict)
    temperature: float = 1.0
    model_cfg: Dict = field(default_factory=dict)
    ens_w1: float = 1.0
    ens_w2: float = 0.0

    # ── File & I/O ───────────────────────────────────────────────────────
    in_path: Any = None
    out_path: Any = None
    wb_out: Any = None
    ws_out: Any = None
    header_in: List = field(default_factory=list)
    header_out: List = field(default_factory=list)
    idx_map: Dict = field(default_factory=dict)
    i_desc_in: Any = None
    pred_col: str = ""
    thr: float = 0.5

    # ── Accumulators ─────────────────────────────────────────────────────
    done: int = 0
    total_rows: int = 0
    CHUNK: int = 4000
    total_classified: int = 0
    excel_row_num: int = 2
    summary_dict: Dict = field(default_factory=dict)
    conf_bins: List = field(default_factory=lambda: [0] * 20)
    al_rows: List = field(default_factory=list)
    start_ts: float = 0.0
    snap: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class ApplyPredictionResult:
    labels: list[str]
    confidences: list[float]
    needs_review: list[int]
    is_ambiguous: list[bool]


def validate_apply_bundle(bundle: Mapping[str, Any], *, schema_version: int = 1) -> None:
    """Ensures apply bundle follows train artifact identity contract."""
    validate_bundle_schema(
        bundle,
        expected_artifact_type=TRAIN_MODEL_ARTIFACT_TYPE,
        supported_schema_version=schema_version,
        allow_missing_schema=True,
    )


def predict_with_thresholds(
    proba: np.ndarray,
    classes: Sequence[str],
    *,
    default_threshold: float = 0.5,
    per_class_thresholds: Mapping[str, float] | None = None,
    fallback_label: str | None = "other_label",
    threshold_mode: Literal["review_only", "strict"] | None = None,
    strict_threshold: bool = False,  # backward compatibility
    review_only: bool = False,  # backward compatibility
    ambiguity_threshold: float = 0.10,
) -> ApplyPredictionResult:
    """Returns labels with per-class thresholding and configurable fallback label.

    Если лучший класс не проходит порог:
    - threshold_mode=\"strict\": возвращается fallback_label (или best_cls при fallback_label=None).
    - threshold_mode=\"review_only\": label остаётся best_cls, но needs_review=1.
    Если threshold_mode не задан, используется migration-совместимость через legacy-флаги:
    review_only=True -> review_only, strict_threshold=True -> strict, иначе review_only.
    """
    if proba.ndim != 2:
        raise ValueError("proba must be 2-D")
    if proba.shape[1] != len(classes):
        raise ValueError("classes size must match proba columns")

    thresholds = dict(per_class_thresholds or {})
    if threshold_mode is None:
        mode: Literal["review_only", "strict"] = "review_only"
        if review_only:
            mode = "review_only"
        elif strict_threshold:
            mode = "strict"
    else:
        if threshold_mode not in {"review_only", "strict"}:
            raise ValueError("threshold_mode must be 'review_only' or 'strict'")
        mode = threshold_mode

    out_labels: list[str] = []
    out_conf: list[float] = []
    out_review: list[int] = []
    out_ambiguous: list[bool] = []

    for row in proba:
        best_idx = int(np.argmax(row))
        best_cls = str(classes[best_idx])
        best_score = float(row[best_idx])
        thr = float(thresholds.get(best_cls, default_threshold))
        if best_score >= thr:
            out_labels.append(best_cls)
            out_review.append(0)
        else:
            out_review.append(1)
            if mode == "review_only":
                out_labels.append(best_cls)
            else:
                out_labels.append(best_cls if fallback_label is None else str(fallback_label))
        out_conf.append(best_score)
        sorted_row = np.sort(row)[::-1]
        second_score = float(sorted_row[1]) if len(sorted_row) > 1 else 0.0
        out_ambiguous.append((best_score - second_score) < ambiguity_threshold)

    return ApplyPredictionResult(
        labels=out_labels,
        confidences=out_conf,
        needs_review=out_review,
        is_ambiguous=out_ambiguous,
    )
