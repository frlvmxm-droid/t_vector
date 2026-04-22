"""Smoke tests for the reproducible-build contract (ADR-0008).

These do not actually build the image or invoke uv — they just guard
the three drift modes that motivated Wave 8.2:

1. ``uv.lock`` exists and is well-formed TOML.
2. Every "Обязательные" entry in ``requirements.txt`` is also declared
   in ``pyproject.toml`` ``[project.dependencies]``. Drift here is what
   broke the bootstrap launcher in Wave 6 incident #4.
3. The Dockerfile still installs from the lock — a future
   ``pip install <name>`` shortcut would silently leak un-pinned wheels
   into the release image.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — runtime is pinned >=3.11 via pyproject
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent


def _normalise_pkg(name: str) -> str:
    """PEP 503 normalisation: lowercase, dash-collapse."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _direct_deps_from_pyproject() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    out: set[str] = set()
    for spec in deps:
        # "scikit-learn>=1.4" → "scikit-learn"
        head = re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0]
        out.add(_normalise_pkg(head))
    return out


def _required_block_from_requirements_txt() -> set[str]:
    """Parse only the Обязательные block (above the first 'Опциональные')."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    out: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Stop at the optional-deps banner.
        if "Опциональные" in line:
            break
        if line.startswith("#"):
            continue
        head = re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0]
        if head:
            out.add(_normalise_pkg(head))
    return out


def test_uv_lock_exists_and_parses() -> None:
    lock = ROOT / "uv.lock"
    assert lock.is_file(), "uv.lock missing — run `uv lock` (ADR-0008)"
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    assert data.get("version") == 1
    assert "package" in data
    # Spot-check: at least the core ML wheels are pinned.
    names = {_normalise_pkg(p["name"]) for p in data["package"]}
    for must in ("numpy", "scikit-learn", "scipy", "joblib"):
        assert must in names, f"{must} missing from uv.lock"


def test_pyproject_deps_cover_requirements_txt() -> None:
    """Every Обязательные line in requirements.txt → pyproject dependency."""
    pyproject = _direct_deps_from_pyproject()
    required = _required_block_from_requirements_txt()
    missing = required - pyproject
    assert not missing, (
        f"requirements.txt declares {missing!r} but pyproject.toml does not. "
        "Add to [project.dependencies] (ADR-0008)."
    )


def test_dockerfile_uses_uv_sync_not_pip_install() -> None:
    """Guards against the `pip install <name>` shortcut regression."""
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "uv sync --frozen" in df, "Dockerfile must install via uv sync --frozen"
    # `pip install --no-cache-dir <name>` is the regression we're guarding
    # against. `pip install -r requirements.txt` is also forbidden in the
    # image — bootstrap_run.py uses it on the desktop, not in CI.
    for forbidden in ("pip install -r", "pip install --no-cache-dir scikit-learn"):
        assert forbidden not in df, (
            f"Dockerfile must not contain {forbidden!r} (ADR-0008)"
        )
