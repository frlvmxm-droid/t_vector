"""Regression guard for `ClusterRunState` slot discipline.

`ClusterRunState` lives inside `ClusterTabMixin` (app_cluster.py:~1895)
and is decorated `@dataclass(slots=True)`. With `slots=True`, attempting
to assign an unknown attribute raises `AttributeError` instead of
silently creating a new instance-dict entry. That matters because the
967-LOC `run_cluster()` worker mutates this state across ~20 stage
boundaries — a typo like `_crs.labls = ...` would otherwise be an
un-trackable silent failure.

This test guards against accidental removal of `slots=True` (or
migration to a plain class). It intentionally does not exercise
anything tkinter-specific; it simply grabs the nested dataclass via
the mixin.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def cluster_run_state_cls():
    """Extract `ClusterRunState` from the mixin without instantiating Tk.

    Tkinter is stubbed in `tests/conftest.py` at collection time.
    """
    import app_cluster

    cls = app_cluster.ClusterTabMixin.ClusterRunState  # type: ignore[attr-defined]
    return cls


def test_cluster_run_state_has_slots(cluster_run_state_cls):
    """slots=True must produce a populated __slots__ attribute."""
    assert hasattr(cluster_run_state_cls, "__slots__"), (
        "ClusterRunState lost slots=True — typos like `_crs.labls = ...` "
        "would now silently succeed."
    )
    assert len(cluster_run_state_cls.__slots__) >= 30, (
        f"__slots__ unexpectedly short: {cluster_run_state_cls.__slots__!r}"
    )


def test_cluster_run_state_rejects_unknown_attr(cluster_run_state_cls):
    """Assigning an unknown field must raise AttributeError."""
    state = cluster_run_state_cls()
    with pytest.raises(AttributeError):
        state.this_field_does_not_exist = 42  # type: ignore[attr-defined]


def test_cluster_run_state_known_attrs_still_writable(cluster_run_state_cls):
    """Declared fields must remain writable (not frozen in this wave)."""
    state = cluster_run_state_cls()
    state.K = 7
    state.kw_dict = {0: "hello"}
    state.use_hdbscan = True
    assert state.K == 7
    assert state.kw_dict == {0: "hello"}
    assert state.use_hdbscan is True
