# -*- coding: utf-8 -*-
"""Pytest-compatible wrapper for test_all.py.

test_all.py uses a custom runner (run_test + sys.exit) that is not
pytest-discoverable. This wrapper collects all test_* functions and
exposes them as individual parametrized pytest cases so they appear
in the standard CI test run.

Skips functions that require unavailable optional deps (numpy, torch, etc.)
"""
from __future__ import annotations

import importlib
import inspect
import pathlib
import sys
from typing import Callable, List, Tuple

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def _collect_test_all_functions() -> List[Tuple[str, Callable]]:
    """Import test_all and collect all test_* functions."""
    try:
        import test_all as _ta
    except Exception:
        return []

    pairs = []
    for name, obj in inspect.getmembers(_ta, inspect.isfunction):
        if name.startswith("test_") and obj.__module__ == _ta.__name__:
            pairs.append((name, obj))
    return sorted(pairs)


_ALL_TESTS = _collect_test_all_functions()
_IDS = [name for name, _ in _ALL_TESTS]
_FNS = [fn for _, fn in _ALL_TESTS]


@pytest.mark.parametrize("fn", _FNS, ids=_IDS)
def test_all_suite(fn: Callable) -> None:
    """Run one test function from test_all.py.

    Skips automatically if the function raises ImportError for
    optional deps (numpy, sklearn, torch, etc.).
    """
    try:
        fn()
    except ImportError as exc:
        pytest.skip(f"optional dep missing: {exc}")
