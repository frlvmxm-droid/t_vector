"""Tests for ci/mutation_smoke.py wrapper.

Wave 8.1. Validates argparse + early-exit paths so a typo in the
nightly workflow surfaces as a CI failure on the same PR rather than
silently mis-running on the next 04:12 UTC trigger.

We do NOT actually run ``mutmut`` here — that would take minutes per
test. We monkey-patch ``shutil.which`` and ``subprocess.check_output``
to inject deterministic responses.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CI_DIR = ROOT / "ci"
if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))

mutation_smoke = importlib.import_module("mutation_smoke")


def test_missing_mutmut_returns_exit_2(monkeypatch, capsys):
    """If mutmut isn't installed, exit 2 (not crash, not 1)."""
    monkeypatch.setattr(mutation_smoke, "_resolve_mutmut", lambda: None)
    rc = mutation_smoke.main(["ml_distillation.py"])
    assert rc == 2
    assert "mutmut not installed" in capsys.readouterr().err


def test_missing_module_returns_exit_2(monkeypatch, capsys):
    """If the source module doesn't exist, exit 2."""
    monkeypatch.setattr(mutation_smoke, "_resolve_mutmut", lambda: "/usr/bin/mutmut")
    rc = mutation_smoke.main(["this_module_does_not_exist.py"])
    assert rc == 2
    assert "module not found" in capsys.readouterr().err


def test_score_above_threshold_returns_zero(monkeypatch, capsys, tmp_path):
    """killed/total ≥ threshold → exit 0 + JSON summary on stdout."""
    monkeypatch.setattr(mutation_smoke, "_resolve_mutmut", lambda: "/usr/bin/mutmut")
    monkeypatch.setattr(mutation_smoke, "_run_mutmut", lambda *a, **kw: 0)
    monkeypatch.setattr(mutation_smoke, "_read_score", lambda *a, **kw: (8, 10))
    rc = mutation_smoke.main([
        "ml_distillation.py", "--threshold", "0.75",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["killed"] == 8
    assert summary["total"] == 10
    assert summary["score"] == 0.8
    assert summary["threshold"] == 0.75


def test_score_below_threshold_returns_one(monkeypatch, capsys, tmp_path):
    """killed/total < threshold → exit 1."""
    monkeypatch.setattr(mutation_smoke, "_resolve_mutmut", lambda: "/usr/bin/mutmut")
    monkeypatch.setattr(mutation_smoke, "_run_mutmut", lambda *a, **kw: 0)
    monkeypatch.setattr(mutation_smoke, "_read_score", lambda *a, **kw: (5, 10))
    rc = mutation_smoke.main([
        "ml_distillation.py", "--threshold", "0.75",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAIL" in err


def test_zero_total_mutants_returns_two(monkeypatch, capsys, tmp_path):
    """No mutants reported (mutmut crashed or scope empty) → exit 2."""
    monkeypatch.setattr(mutation_smoke, "_resolve_mutmut", lambda: "/usr/bin/mutmut")
    monkeypatch.setattr(mutation_smoke, "_run_mutmut", lambda *a, **kw: 0)
    monkeypatch.setattr(mutation_smoke, "_read_score", lambda *a, **kw: (0, 0))
    rc = mutation_smoke.main([
        "ml_distillation.py",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert rc == 2
    assert "no mutants" in capsys.readouterr().err


def test_build_runner_with_test_files() -> None:
    runner = mutation_smoke._build_runner([
        "tests/test_a.py", "tests/test_b.py",
    ])
    assert "pytest" in runner
    assert "tests/test_a.py" in runner
    assert "tests/test_b.py" in runner


def test_build_runner_empty_tests_uses_full_suite() -> None:
    runner = mutation_smoke._build_runner([])
    assert "pytest" in runner
    assert "tests/" not in runner  # no path appended
