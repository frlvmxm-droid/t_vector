# -*- coding: utf-8 -*-
"""Append-only JSONL log of training experiments.

Each call to log_experiment() appends one JSON line to
~/.classification_tool/experiments.jsonl so users can track
model quality over time without digging through logs.
"""
from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime
from typing import Any, Dict

EXPERIMENT_LOG = pathlib.Path.home() / ".classification_tool" / "experiments.jsonl"

_TRACKED_PARAMS = (
    "train_mode", "C", "max_iter", "test_size", "use_smote",
    "oversample_strategy", "use_lemma", "use_sbert", "use_svd",
    "use_per_field", "class_weight_balanced",
)


def log_experiment(model_path: str, snap: Dict[str, Any], eval_metrics: Dict[str, Any]) -> None:
    """Append one experiment record to the JSONL log (silent on errors)."""
    try:
        entry = {
            "timestamp":             datetime.now().isoformat(timespec="seconds"),
            "model_file":            str(model_path),
            "macro_f1":              eval_metrics.get("macro_f1"),
            "accuracy":              eval_metrics.get("accuracy"),
            "roc_auc":               eval_metrics.get("roc_auc"),
            "n_train":               eval_metrics.get("n_train"),
            "n_test":                eval_metrics.get("n_test"),
            "training_duration_sec": eval_metrics.get("training_duration_sec"),
            "model_size_bytes":      eval_metrics.get("model_size_bytes"),
            "ece":                   eval_metrics.get("ece"),
            "mce":                   eval_metrics.get("mce"),
            "temperature":           eval_metrics.get("temperature"),
            "per_class_f1":          eval_metrics.get("per_class_f1"),
            "per_class_precision":   eval_metrics.get("per_class_precision"),
            "per_class_recall":      eval_metrics.get("per_class_recall"),
            "per_class_thresholds":  eval_metrics.get("per_class_thresholds"),
            "params":                {k: snap.get(k) for k in _TRACKED_PARAMS},
        }
        EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass


def read_experiments(last_n: int = 20) -> list:
    """Read the last N experiment records from the log."""
    if not EXPERIMENT_LOG.exists():
        return []
    lines = EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records[-last_n:]
