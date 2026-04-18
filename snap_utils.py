"""Shared `freeze_snap` helper — single source of truth for snap immutability.

Before P3 there were three near-identical copies of the same three-line
freeze expression:

  1. `app_cluster.ClusterTabMixin._freeze_snap` (method form)
  2. `cluster_workflow_service.ClusteringWorkflow.run`  (inline)
  3. `cluster_workflow_service.ClusteringWorkflow.prepare_only` (inline)

Because (2) and (3) were inline, a caller could in principle bypass the
freeze by constructing a `ClusteringWorkflow` and reaching past the
stages into a mutable `snap`. Consolidating into this module makes the
freeze a named boundary — mypy / reviewers can grep for a single
import, and the UI, the service layer, and any future CLI batch driver
all share one implementation.

Semantic contract:
  * If `snap` is already a `MappingProxyType`, return it unchanged (no
    double-wrap, preserves identity for cache keys).
  * Otherwise, shallow-copy into a new `dict` and wrap in
    `MappingProxyType`. The copy protects against later mutation of
    the caller's dict leaking into stages via the proxy.
  * **Shallow only** by design: we do not deep-freeze nested lists,
    since the cluster snap only contains scalars + a couple of
    read-only filename lists that the pipeline treats as immutable by
    convention. If that convention ever breaks, promote this to a
    recursive freeze here rather than duplicating the copy logic at
    call sites again.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_snap(snap: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only view of *snap* safe to forward into pipeline stages."""
    if isinstance(snap, MappingProxyType):
        return snap
    return MappingProxyType(dict(snap))
