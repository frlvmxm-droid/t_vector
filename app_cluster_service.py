# -*- coding: utf-8 -*-
"""Совместимый фасад сервисов кластеризации."""
from __future__ import annotations

from cluster_elbow import ClusterElbowSelector
from cluster_persistence import ClusterModelPersistence
from cluster_reason_builder import ClusterReasonBuilder
from llm_client import LLMClient, _urlreq, random, time, is_safe_url
from llm_key_store import (
    LLMSnapshotDecryptError,
    decrypt_api_key_from_snapshot,
    encrypt_api_key_for_snapshot,
    resolve_api_key,
)

__all__ = [
    "LLMClient",
    "ClusterReasonBuilder",
    "ClusterModelPersistence",
    "ClusterElbowSelector",
    "encrypt_api_key_for_snapshot",
    "decrypt_api_key_from_snapshot",
    "resolve_api_key",
    "is_safe_url",
    "LLMSnapshotDecryptError",
    # test compatibility exports
    "_urlreq",
    "random",
    "time",
]
