# -*- coding: utf-8 -*-
"""UI-builder helpers for cluster tab composition."""
from __future__ import annotations

from typing import Any

from app_cluster_view import (
    build_cluster_algo_main_section,
    build_cluster_basic_settings_sections,
    build_cluster_files_card,
)


def build_cluster_primary_sections(app: Any, settings_frame: Any):
    """Builds core cluster-tab sections and returns files-card widget."""
    card = build_cluster_files_card(app, settings_frame)
    build_cluster_basic_settings_sections(app, settings_frame, card)
    build_cluster_algo_main_section(app, settings_frame)
    return card
