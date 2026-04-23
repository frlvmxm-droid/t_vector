# Changelog (SemVer)

All notable changes to BankReasonTrainer are documented here. The
project follows [SemVer 2.0](https://semver.org/); the version in
`pyproject.toml` and `bank_reason_trainer.__version__` is the source of
truth.

## [Unreleased]
### Added
-
### Changed
-
### Fixed
-
### Security
-

---

## [10.0.0] — 2026-04-22

**Breaking release — desktop UI removed.** The project ships a
browser-based Voilà dashboard as its only interactive UI; migrate via
`./run_web.sh` (Linux/macOS) or `run_web.bat` (Windows). See
`docs/WEB_MIGRATION_PLAN.md` for the full migration plan; all 4
sprints are ✅ DONE.

### Added
- **Web UI launchers**: `run_web.sh` and `run_web.bat` auto-detect
  Python 3.11+, install the `[ui]` extra, start Voilà on
  `http://127.0.0.1:8866/`, and open the browser.
  `docs/QUICKSTART_WEB_UI.md` covers the full flow.
- **Multi-theme support** in `ui_widgets.theme`:
  - `PALETTES` dict with three palettes (`dark-teal`, `paper`,
    `amber-crt`).
  - `apply_theme(name)` / `rebuild_css()` / `get_active_theme()`.
  - Sidebar theme-switcher in `ui_widgets.notebook_app`; selection is
    persisted in `~/.classification_tool/last_session.json` under key
    `ui.theme`.
  - New design helpers `field_label()`, `separator()`;
    `section_card(..., right=widget)` now accepts a header-right slot;
    `chip()`/`badge()` gained kind `"accent"`.
- **JupyterHub / Docker deploy templates** at the repo root:
  `Dockerfile.jupyterhub`, `jupyterhub_config.py.example`,
  `docker-compose.yml.example`. Documented in
  `docs/JUPYTERHUB_UI.md` § "Self-hosted via docker-compose".
- **Unit tests**: `tests/test_theme_palettes.py` (45 cases covering
  palette schema, `apply_theme`, `chip`/`badge` kinds, CSS selectors
  across all palettes) and `tests/test_theme_session.py` (12 cases
  for `ui.theme` round-trip, transient-key filtering, atomic writes).

### Changed
- **Service layer is now the only consumer of business logic.**
  `app_train_service`, `apply_prediction_service`,
  `cluster_workflow_service`, `app_cluster_pipeline` are unchanged —
  only the presentation layer switched.
- **CI job `ui-smoke` (Xvfb + CustomTkinter)** replaced with
  `web-smoke` (Voilà HTTP probe via `tests/test_voila_smoke.py`,
  `RUN_VOILA_SMOKE=1` on by default).
- **`pyproject.toml`** no longer lists `customtkinter`, `pystray`, or
  `Pillow` as runtime deps (Pillow survives transitively via
  matplotlib/ipywidgets). `uv.lock` regenerated accordingly.
- **Docker image** (`Dockerfile`) no longer installs `python3-tk`;
  base stays at `python:3.11.11-slim-bookworm` without X11.
- **Documentation** (`README.md`, `CLAUDE.md`, `docs/DEPLOY.md`,
  `docs/JUPYTERHUB_UI.md`) rewritten for the web-only stack;
  keyboard-shortcut and EXE-build sections removed.

### Removed
- **Desktop UI layer** — 19 files, ~750 KB, ~18 943 LoC:
  `app.py`, `app_train.py`, `app_apply.py`, `app_cluster.py`,
  `app_deps.py`, `app_train_view.py` / `app_apply_view.py` /
  `app_cluster_view.py` + `_ctk.py` counterparts,
  `app_dialogs_ctk.py`, `experiment_history_dialog.py`,
  `cluster_ui_builder.py`, `bank_reason_trainer_gui.py`,
  `ui_widgets_tk.py`, `ui_theme.py`, `ui_theme_ctk.py`,
  `background.png`.
- **Desktop launchers and PyInstaller build**: `bootstrap_run.py`,
  `bootstrap_ctk.py`, `run.sh`, `run_app.bat`,
  `bank_reason_trainer.spec`, `build_exe.bat`.
- **Deprecated shims**: `core/feature_builder.py`,
  `core/hw_profile.py`, `core/text_utils.py`, `core/__init__.py`
  (re-export warnings only; no external callers).
- **Desktop tests**: `tests/test_ui_smoke.py`,
  `tests/test_ui_cluster_e2e.py`, `tests/test_app_mixins_import_smoke.py`,
  `tests/test_ui_widgets_tk_export_smoke.py`,
  `tests/test_artifact_extension_policy.py`,
  `tests/test_bootstrap_and_utils.py`,
  `tests/test_cluster_llm_provider_defaults.py`,
  `tests/test_cluster_processing_guard.py`,
  `tests/test_cluster_view_sections_smoke.py`. Tk/CTK stubs removed
  from `tests/conftest.py`.
- **Obsolete files**: `requirements.txt` (use `pyproject.toml` +
  `uv.lock`), `ctk_migration_pack/` (palettes copied to
  `ui_widgets.theme:PALETTES`), `UI_IMPLEMENTATION_GUIDE.md`,
  `docs/ui_refactor_plan.md`.
- **CI`ui-smoke`job** and its Xvfb + `python3-tk` apt-install step.

### Migration notes
- CLI is untouched: `python -m bank_reason_trainer {train,apply,cluster}`
  works exactly as before.
- Sessions saved by 9.x store widget values only; 10.x adds
  `"ui.theme"`. Old sessions load without warnings.
- `.joblib` bundles, trust store, LLM API-key encryption — all
  unchanged.
- Users on JupyterHub: install `[ui]` extra and point spawner at
  `/voila/render/notebooks/ui.ipynb` (see
  `docs/JUPYTERHUB_UI.md` § "Operator setup").

---

## [9.x and earlier]

Pre-10.0.0 changelog entries live in
[`CHANGELOG_TEMPLATE.md`](CHANGELOG_TEMPLATE.md) (empty skeleton) and
in `git log --oneline origin/main`. The repo did not previously
maintain a running changelog; this file starts fresh with 10.0.0.
