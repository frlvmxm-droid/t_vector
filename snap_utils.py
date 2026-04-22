"""Single freeze boundary for cluster-snap handoff into pipeline stages.

Semantic contract:
  * If `snap` is already a `MappingProxyType`, return it unchanged —
    preserves identity for cache keys and avoids double-wrap.
  * Otherwise shallow-copy into a new `dict` and wrap in
    `MappingProxyType`; the copy shields stages from later mutation
    of the caller's dict.

Shallow-only by design: the cluster snap is scalars + a couple of
read-only filename lists. Promote to a recursive freeze here if that
ever stops being true.
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
