"""UI-driven cluster E2E — behavioural baseline for Wave 7 refactor.

Unlike ``test_ui_smoke.py`` (which only verifies ``App()`` boots), this
test drives the full ``run_cluster()`` path end-to-end through Xvfb:

  1. Tiny CSV written to ``tmp_path``.
  2. ``App()`` constructed; ``cluster_files`` + key tk.Var fields set to
     exercise the ``tfidf`` + ``kmeans`` slice (matches the pipeline
     path already shipped in ``app_cluster_pipeline.py``).
  3. ``app.run_cluster()`` invoked directly; Tk event loop pumped via
     ``update()`` until ``self._processing`` flips back to ``False``
     or the timeout fires.
  4. Output XLSX asserted to exist in the monkey-patched
     ``CLUST_DIR``.

This is the safety net ADR-0007 requires before extracting stages 2–3
from ``run_cluster()``. A refactor that silently breaks the UI wiring,
threading hand-off, or the "put the output next to CLUST_DIR" contract
fails this test, which the pipeline-only smoke suite cannot catch.

Skipped when the UI stack is unavailable (same pattern as
``test_ui_smoke.py``). Runs in CI via the ``ui-smoke`` job.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ui_stack_available() -> tuple[bool, str]:
    if os.environ.get("DISPLAY", "") == "":
        return False, "no DISPLAY (run via xvfb-run -a)"
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        return False, f"tkinter не установлен: {exc}"
    try:
        import customtkinter  # noqa: F401
    except ImportError as exc:
        return False, f"customtkinter не установлен: {exc}"
    return True, ""


_OK, _REASON = _ui_stack_available()

# A 10-row Russian corpus, two obvious topics (перевод / карта).
# Small enough to cluster in <2s on the CI runner, balanced enough
# that k=2 reliably separates the two topics.
_FIXTURE_ROWS = [
    "перевод денег срочно прошёл",
    "блокировка карты сегодня утром",
    "не могу сделать перевод на счёт",
    "карта заблокирована вчера вечером",
    "нужно перевести деньги поскорее",
    "разблокировать карту помогите",
    "отправить перевод не получается",
    "карта блок не снимается",
    "перевод завис в обработке",
    "карту заморозили без предупреждения",
]


def _write_fixture(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text"])
        for row in _FIXTURE_ROWS:
            w.writerow([row])


def _wait_for_completion(app, timeout_sec: float = 120.0) -> None:
    """Pump the Tk event loop until _processing flips to False."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        app.update()
        if not app._processing:
            # Drain any pending .after(0, ...) callbacks.
            for _ in range(5):
                app.update()
            return
        time.sleep(0.1)
    raise TimeoutError(
        f"run_cluster() did not finish within {timeout_sec}s; "
        f"_processing still True"
    )


@pytest.mark.slow
@pytest.mark.skipif(not _OK, reason=_REASON)
def test_run_cluster_end_to_end_produces_xlsx(tmp_path, monkeypatch):
    """Full run_cluster() drive: CSV in, XLSX out in CLUST_DIR."""
    # Redirect clustering output to tmp_path so we don't pollute
    # the real APP_ROOT/clustering directory on the CI runner.
    clust_dir = tmp_path / "clustering"
    clust_dir.mkdir()
    monkeypatch.setattr("app_cluster.CLUST_DIR", clust_dir, raising=True)

    # Suppress any message-box popups that would block the Tk loop.
    from unittest.mock import MagicMock
    monkeypatch.setattr("tkinter.messagebox.showerror", MagicMock(), raising=False)
    monkeypatch.setattr("tkinter.messagebox.showwarning", MagicMock(), raising=False)
    monkeypatch.setattr("tkinter.messagebox.showinfo", MagicMock(), raising=False)

    csv_path = tmp_path / "cluster_input.csv"
    _write_fixture(csv_path)

    from app import App

    app = App()
    try:
        # Minimum inputs for the tfidf+kmeans slice.
        app.cluster_files = [str(csv_path)]
        app.desc_col.set("text")
        app.k_clusters.set(2)
        app.cluster_vec_mode.set("tfidf")
        app.cluster_algo.set("kmeans")
        app.cluster_role_mode.set("all")
        # Silence heavy post-clustering steps that aren't part of the
        # contract we're locking down. use_elbow=False so K=2 is honoured
        # rather than the elbow method picking its own K on 10 rows.
        if hasattr(app, "use_elbow"):
            app.use_elbow.set(False)
        if hasattr(app, "use_t5_summary"):
            app.use_t5_summary.set(False)
        if hasattr(app, "use_sbert_cluster"):
            app.use_sbert_cluster.set(False)
        if hasattr(app, "llm_name_clusters"):
            app.llm_name_clusters.set(False)

        app.run_cluster()
        _wait_for_completion(app, timeout_sec=120.0)

        outputs = list(clust_dir.glob("*clustered*.xlsx"))
        assert outputs, (
            f"run_cluster() finished but no clustered XLSX in {clust_dir}; "
            f"contents: {list(clust_dir.iterdir())}"
        )
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001 — Tk-cleanup best-effort
            pass
