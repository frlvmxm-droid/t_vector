# -*- coding: utf-8 -*-
"""Cluster runtime lifecycle helpers (start/cancel/cleanup/tuning)."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterable

from config.ml_constants import KMEANS_BATCH_SIZE
from hw_profile import tune_runtime_by_input_size


def try_mark_processing(owner: Any) -> bool:
    with owner._proc_lock:
        if owner._processing:
            return False
        owner._processing = True
    return True


def clear_processing(owner: Any) -> None:
    with owner._proc_lock:
        owner._processing = False


def tune_cluster_runtime_for_input(
    *,
    files_snapshot: Iterable[str],
    snap: dict,
    hw: Any,
    log_fn,
) -> dict:
    try:
        bytes_total = sum(Path(p).stat().st_size for p in files_snapshot if Path(p).exists())
        tuned = tune_runtime_by_input_size(
            input_bytes=bytes_total,
            chunk=int(snap.get("streaming_chunk_size", 5000)),
            sbert_batch=int(snap.get("sbert_batch", hw.sbert_batch)),
            kmeans_batch=int(snap.get("kmeans_batch", KMEANS_BATCH_SIZE)),
        )
        snap["streaming_chunk_size"] = tuned["chunk"]
        snap["sbert_batch"] = tuned["sbert_batch"]
        snap["kmeans_batch"] = tuned["kmeans_batch"]
        log_fn(
            f"[auto-profile] input={tuned['input_size_gb']} GB | "
            f"stream_chunk={tuned['chunk']} | sbert_batch={tuned['sbert_batch']} | "
            f"kmeans_batch={tuned['kmeans_batch']}"
        )
        return tuned
    except Exception:
        return {}


def cleanup_cluster_runtime(log_fn) -> None:
    try:
        gc.collect()
    except (RuntimeError, ValueError, TypeError) as ex:
        log_fn(f"cluster cleanup gc.collect failed: {ex}")
    try:
        import torch as _torch_mem

        if _torch_mem.cuda.is_available():
            _torch_mem.cuda.empty_cache()
    except (ImportError, RuntimeError, AttributeError) as ex:
        log_fn(f"cluster cleanup torch.cuda.empty_cache failed: {ex}")
