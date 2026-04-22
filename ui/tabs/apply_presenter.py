# -*- coding: utf-8 -*-
"""Presenter helpers for the Apply tab (UI-free formatting)."""
from __future__ import annotations

from typing import Any, Mapping


def format_apply_autoprofile_log(tuned: Mapping[str, Any]) -> str:
    """Render the runtime autoprofile tuning result as a one-line log entry.

    The message summarises the tuned values picked by
    ``hw_profile.tune_runtime_by_input_size`` so the user can see why the
    apply workflow is using the chunk / batch sizes it chose.
    """
    size = tuned.get("input_size_gb")
    chunk = tuned.get("chunk")
    sbert_batch = tuned.get("sbert_batch")
    size_str = f"{float(size):.1f}" if isinstance(size, (int, float)) else "?"
    chunk_str = str(chunk) if chunk is not None else "?"
    sbert_str = str(sbert_batch) if sbert_batch is not None else "?"
    return (
        f"[autoprofile] input={size_str} GB, "
        f"chunk={chunk_str}, sbert_batch={sbert_str}"
    )
