# -*- coding: utf-8 -*-
"""Адаптер состояния для preflight-этапа кластеризации."""
from __future__ import annotations

from typing import Any

from app_cluster_workflow import build_validated_cluster_snapshot


def build_cluster_runtime_snapshot(app: Any):
    """Собирает runtime-snapshot кластера из UI-state без запуска pipeline."""
    snap = build_validated_cluster_snapshot(app)
    if snap is None:
        return None

    anchor_raw = getattr(app, "txt_anchors", None)
    if anchor_raw is not None:
        anchor_lines = anchor_raw.get("1.0", "end").strip().splitlines()
        snap["anchor_phrases"] = [ln.strip() for ln in anchor_lines if ln.strip()]
    return snap
