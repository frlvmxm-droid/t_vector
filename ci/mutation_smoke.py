#!/usr/bin/env python3
"""Per-module mutation testing wrapper.

Runs ``mutmut run`` against a single source module with a focused subset
of the test suite, then reports the kill score and gates on a
configurable threshold. Designed for two scenarios:

* **Local smoke** — a developer can iterate on test quality for one
  numerically critical module without paying the 6h cost of running
  mutmut against the whole ``ml_*`` tree::

      PYTHONPATH=. python ci/mutation_smoke.py ml_distillation \
          --tests tests/test_distill_numerics.py \
                  tests/test_property_invariants.py \
          --threshold 0.75

* **Nightly CI** — ``.github/workflows/mutation-score.yml`` invokes
  this script per target module so a slow mutant in one module does
  not gate the others. Each invocation owns its own ``.mutmut-cache``
  via ``--cache-dir``; CI archives the resulting JSON summary as an
  artefact for trend tracking (ADR-0006 §3).

Exit codes:
  0  — score ≥ threshold
  1  — score < threshold
  2  — mutmut not installed / module missing / unrecoverable error
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent


def _resolve_mutmut() -> Optional[str]:
    """Return path to mutmut executable, or None if not installed."""
    return shutil.which("mutmut")


def _build_runner(test_files: Sequence[str]) -> str:
    """Build a -x -q pytest runner string for mutmut."""
    if not test_files:
        return "python -m pytest -x -q --tb=no"
    return f"python -m pytest -x -q --tb=no {' '.join(test_files)}"


def _run_mutmut(
    mutmut: str,
    *,
    module: str,
    runner: str,
    cache_dir: Path,
) -> int:
    """Invoke ``mutmut run`` once. Returns process exit code."""
    cmd = [
        mutmut, "run",
        "--paths-to-mutate", module,
        "--runner", runner,
        # mutmut's exit code is non-zero when ANY mutant survives — that's
        # not what we gate on; we compute the kill score ourselves below.
    ]
    env_overrides: dict[str, str] = {"MUTMUT_CACHE_PATH": str(cache_dir)}
    print(f"[mutation_smoke] running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(
        cmd, cwd=ROOT, env={**__import__("os").environ, **env_overrides},
        check=False,
    )
    return proc.returncode


def _read_score(mutmut: str, *, cache_dir: Path) -> tuple[int, int]:
    """Returns (killed, total) by parsing ``mutmut results --json``.

    Older mutmut versions emit a different format; we tolerate both by
    falling back to text parsing of ``mutmut results``.
    """
    env = {**__import__("os").environ, "MUTMUT_CACHE_PATH": str(cache_dir)}
    try:
        out = subprocess.check_output(
            [mutmut, "results", "--json"], cwd=ROOT, env=env, text=True,
        )
        data = json.loads(out)
        killed = sum(1 for m in data if m.get("status") == "killed")
        total = len(data)
        return killed, total
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        pass

    out = subprocess.check_output(
        [mutmut, "results"], cwd=ROOT, env=env, text=True,
    )
    killed = total = 0
    for line in out.splitlines():
        s = line.strip().lower()
        if s.startswith("killed"):
            killed += s.count(",") + 1 if ":" not in s else _safe_count_after_colon(s)
        elif s.startswith("survived") or s.startswith("timeout") or s.startswith("suspicious"):
            pass
        if "total" in s and ":" in s:
            try:
                total = int(s.split(":", 1)[1].strip())
            except ValueError:
                continue
    return killed, total


def _safe_count_after_colon(line: str) -> int:
    try:
        rhs = line.split(":", 1)[1].strip()
        return len([t for t in rhs.split(",") if t.strip()])
    except (IndexError, ValueError):
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Per-module mutation testing wrapper.")
    p.add_argument("module", help="Python source file to mutate (e.g. ml_distillation.py).")
    p.add_argument(
        "--tests", nargs="+", default=[],
        help="Test files to run for each mutant. Empty = full suite (slow).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.75,
        help="Required kill ratio (default: 0.75 per ADR-0006).",
    )
    p.add_argument(
        "--cache-dir", default=".mutmut-cache",
        help="Cache dir; nightly CI sets this per-module to isolate runs.",
    )
    args = p.parse_args(argv)

    mutmut = _resolve_mutmut()
    if mutmut is None:
        print("[mutation_smoke] mutmut not installed; pip install mutmut>=2.4", file=sys.stderr)
        return 2

    module_path = ROOT / args.module
    if not module_path.is_file():
        print(f"[mutation_smoke] module not found: {args.module}", file=sys.stderr)
        return 2

    cache_dir = (ROOT / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    runner = _build_runner(args.tests)
    _run_mutmut(mutmut, module=args.module, runner=runner, cache_dir=cache_dir)

    killed, total = _read_score(mutmut, cache_dir=cache_dir)
    if total <= 0:
        print("[mutation_smoke] no mutants reported; nothing to score", file=sys.stderr)
        return 2
    score = killed / total
    summary = {
        "module": args.module,
        "killed": killed,
        "total": total,
        "score": round(score, 4),
        "threshold": args.threshold,
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if score < args.threshold:
        print(
            f"[mutation_smoke] FAIL: {killed}/{total} = {score:.2%} < threshold {args.threshold:.2%}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[mutation_smoke] OK: {killed}/{total} = {score:.2%} ≥ threshold {args.threshold:.2%}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
