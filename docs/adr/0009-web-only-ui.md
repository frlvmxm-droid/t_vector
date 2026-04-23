# ADR-0009: Web-only UI (desktop Tk/CTk removed)

## Status
Accepted; shipped in release 10.0.0 (2026-04-22). Supersedes the
desktop-related portions of ADR-0001 (layered architecture),
ADR-0002 (pipeline stages), ADR-0003 (frozen run state), ADR-0007
(quality polish, `ui-smoke` job), and ADR-0008 (`bootstrap_run.py`
as the desktop launcher).

## Context
BankReasonTrainer shipped two UIs in parallel through the 3.x–9.x
series:

1. **Desktop** (`app.py` + `TrainTabMixin`/`ApplyTabMixin`/
   `ClusterTabMixin` + `bootstrap_run.py`) — Tkinter, later CTk,
   ~20 k LoC, PyInstaller EXE for Windows.
2. **Web** (`ui_widgets/*` + `notebooks/ui.ipynb`) — Voilà +
   ipywidgets, rendered over the same headless service layer
   (`app_train_service`, `apply_prediction_service`,
   `cluster_workflow_service`).

The service layer had been Tk-free since Wave 3a (ADR-0002), so
both UIs were consumers, not owners, of business logic. But the
duplication imposed recurring costs:

- **Packaging.** Two runtime dependency sets (`customtkinter`,
  `pystray`, `Pillow` for desktop; `ipywidgets`, `voila` for web);
  Dockerfile had to install `python3-tk` for the `ui-smoke` Xvfb
  job.
- **CI.** `ui-smoke` under `xvfb-run` ran tests/test_ui_smoke.py
  and tests/test_ui_cluster_e2e.py — slow, non-deterministic,
  regularly broken by upstream Tk changes.
- **Drift.** Design polish (palettes, theme switching, dialog
  refactoring) had to happen twice. The desktop cluster panel
  still referenced closure helpers that the web panel didn't
  need.
- **Developer confusion.** Contributors had to decide *which* UI to
  wire a new feature into; the old ADR-0001 dated from the
  tkinter-only era.

By mid-2026 the Voilà dashboard had reached feature parity for the
three top-level workflows (train / apply / cluster), Voilà
cold-start landed at ~1.4 s (down from 72 s in Wave 7), and
`./run_web.sh` / `run_web.bat` provided one-command boot. The
desktop UI became a maintenance liability without a user.

## Decision

Remove the desktop UI stack entirely. The Voilà dashboard becomes
the only interactive UI; the headless CLI
(`python -m bank_reason_trainer …`) remains untouched.

Concretely (all shipped in release 10.0.0, plan in
`docs/WEB_MIGRATION_PLAN.md`):

1. Delete all Tk/CTk presentation code (19 files, −18 943 LoC):
   `app.py`, `app_{train,apply,cluster}.py`, `app_deps.py`,
   `app_{train,apply,cluster}_view.py` + `_ctk.py`,
   `app_dialogs_ctk.py`, `experiment_history_dialog.py`,
   `cluster_ui_builder.py`, `bank_reason_trainer_gui.py`,
   `ui_widgets_tk.py`, `ui_theme.py`, `ui_theme_ctk.py`,
   `background.png`.
2. Delete desktop launchers and PyInstaller build:
   `bootstrap_run.py`, `bootstrap_ctk.py`, `run.sh`,
   `run_app.bat`, `bank_reason_trainer.spec`, `build_exe.bat`.
3. Drop `customtkinter`, `pystray`, `Pillow` from `pyproject.toml`;
   `Pillow` survives transitively via matplotlib/ipywidgets. Run
   `uv lock` and commit both files (ADR-0008 enforcement).
4. Replace the CI job `ui-smoke` (Xvfb + customtkinter +
   `python3-tk`) with `web-smoke` (Voilà HTTP probe via
   `tests/test_voila_smoke.py` with `RUN_VOILA_SMOKE=1`).
5. Delete desktop-specific tests and the Tk/CTk stub block from
   `tests/conftest.py` (9 files, −1 202 LoC).
6. Rewrite README, CLAUDE.md, docs/DEPLOY.md, docs/JUPYTERHUB_UI.md,
   Dockerfile around the web-only stack.
7. Copy the three palettes (dark-teal, paper, amber-crt) from
   `ctk_migration_pack/ui_theme_ctk.py` into
   `ui_widgets.theme.PALETTES`, then delete
   `ctk_migration_pack/` and `UI_IMPLEMENTATION_GUIDE.md`.
8. Ship JupyterHub / Docker deploy templates:
   `Dockerfile.jupyterhub`, `jupyterhub_config.py.example`,
   `docker-compose.yml.example`, plus a §"Self-hosted via
   docker-compose" section in `docs/JUPYTERHUB_UI.md`.

The fallback option — "desktop stays as a thin wrapper over
`ClusteringWorkflow.run`" — was rejected: it would preserve the
dep/CI/drift costs above for a UI with ~zero users since
mid-Wave-7.

## Consequences

**Positive**

- **Single UI path.** Features live in `ui_widgets/*.py` +
  `ui_widgets/theme.py`; CSS classes + CSS custom properties give
  runtime theme-switching for free (three palettes ship in 10.0.0;
  adding more is a dict entry).
- **Lighter runtime.** `pip install -e ".[ui]"` pulls only
  ipywidgets + voila + the ML stack; no CTk/Tk/Pillow transitives
  that Windows/macOS users had to build from wheels.
- **Faster CI.** `web-smoke` runs in ~15 s (Voilà subprocess + HTTP
  probe); `ui-smoke` under Xvfb was ~90 s and flaky.
- **Cross-platform parity.** `./run_web.sh` / `run_web.bat` boot
  identical UIs; the previous Windows-only PyInstaller EXE is
  gone.
- **Doc footprint.** −894 LoC in `UI_IMPLEMENTATION_GUIDE.md`, no
  more "which tab?" mental overhead.

**Negative / mitigations**

- **Plotly / Cleanlab visualisations** weren't wired into the web
  cluster panel. Known limitation; can be added via
  `plotly.FigureWidget` in a follow-up.
- **Offline single-user "tray" mode** (icon-in-taskbar) is gone.
  Voilà needs a browser tab; users on locked-down workstations
  without a browser are out-of-scope.
- **Cancel button** is wired for the cluster workflow (Phase 14:
  `ui_widgets/progress.py::attach_cancel_event` →
  `threading.Event` → `ClusteringWorkflow.run` → `WorkflowCancelled`)
  and covered by `tests/test_cluster_cancel.py`. For the train and
  apply workflows the event is not yet threaded through the service
  layer — cancel there requires `cancel_event` hooks in
  `TrainingWorkflow.fit_and_evaluate` and
  `predict_with_thresholds`; tracked as a follow-up.
- **Pinning note.** `Dockerfile.jupyterhub` uses
  `jupyterhub/jupyterhub:4.1.5` — the hub image is the only new
  non-wheel dep outside `uv.lock`. Releases should pin the digest
  per ADR-0008 § "Pinning the base image".

## Consequences for older ADRs

- **ADR-0001** (layered architecture): still describes the correct
  layering, but the "tkinter modules" phrasing is historical —
  `ui_widgets/*.py` is the current presentation layer.
- **ADR-0002** (pipeline stages): the cluster pipeline stages
  (`prepare_inputs`, `build_vectors`, `run_clustering`,
  `postprocess_clusters`, `export_cluster_outputs`) are unchanged;
  the paragraph about "`run_cluster()` in `app_cluster.py` is ~969
  LOC" describes the pre-extraction state of a file that no longer
  exists.
- **ADR-0003** (frozen run state): targets `app_cluster.py` /
  `app_train.py`, both deleted. Run-state dataclasses now live next
  to the service layer (`cluster_run_coordinator.py`).
- **ADR-0007** (quality polish): the `ui-smoke` job and the
  `.coveragerc` omits for `app_deps.py` / `app_*_view*.py` /
  `ui_theme*.py` are superseded by the current `web-smoke` job and
  the trimmed `.coveragerc` in release 10.0.0.
- **ADR-0008** (reproducible builds): `requirements.txt` was kept
  "for the `bootstrap_run.py` desktop launcher"; with that launcher
  gone, the file was deleted and `pyproject.toml` + `uv.lock`
  remain the sole source of truth.
- **ADR-0005** (SBOM and signed releases): the build provenance
  listing no longer includes `pyinstaller` — only `uv`, `python`,
  commit, and timestamp.

## References

- `docs/WEB_MIGRATION_PLAN.md` — sprint-by-sprint plan (Sprints 1–4
  all ✅ DONE).
- `CHANGELOG.md` — release 10.0.0 entry.
- Related pull request: #2 (merged 2026-04-22).
