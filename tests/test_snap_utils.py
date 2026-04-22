"""Contract tests for `snap_utils.freeze_snap`.

This helper is the single freeze boundary for cluster-snap handoff into
pure pipeline stages. Three callers rely on its exact semantics:
  * `ClusterTabMixin._freeze_snap` (UI → stages)
  * `ClusteringWorkflow.run` (service → stages)
  * `ClusteringWorkflow.prepare_only` (CLI → stages)

A silent behaviour change here (e.g. returning the caller's dict
directly, or deep-copying into a new dict) would break the "caller's
mutation cannot leak into a running stage" invariant documented in
ADR-0002.
"""
from __future__ import annotations

import pathlib
import sys
from types import MappingProxyType

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from snap_utils import freeze_snap


def test_returns_mapping_proxy():
    view = freeze_snap({"k": 1})
    assert isinstance(view, MappingProxyType)
    assert view["k"] == 1


def test_already_frozen_returned_unchanged():
    """Identity preservation matters for cache keys and avoids double-wrap."""
    original = MappingProxyType({"k": 1})
    view = freeze_snap(original)
    assert view is original


def test_view_rejects_mutation():
    view = freeze_snap({"k": 1})
    with pytest.raises(TypeError):
        view["k"] = 2  # type: ignore[index]


def test_caller_mutation_does_not_leak():
    """Shallow copy shields stages from later caller-side writes."""
    src = {"k": 1}
    view = freeze_snap(src)
    src["k"] = 999
    src["new"] = "added"
    assert view["k"] == 1
    assert "new" not in view
