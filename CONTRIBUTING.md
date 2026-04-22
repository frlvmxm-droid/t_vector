# Contributing to BankReasonTrainer

Thank you for your interest in improving BankReasonTrainer. This document
describes the development workflow, local setup, quality gates, and expected
conventions. For a high-level architectural tour, read `CLAUDE.md` first.

---

## Local Setup

Requires **Python 3.11 or 3.12** (earlier versions are not tested).

```bash
git clone <repo-url>
cd t_vector
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install ruff==0.6.9 mypy==1.11.2 bandit==1.7.9 pytest pytest-cov
```

Run the app: `python bootstrap_run.py`.

## Running the Quality Gates Locally

Mirror the CI pipeline (`.github/workflows/quality-gates.yml`) before opening
a PR:

```bash
# 1. Lint — strict subset (contracts + ML core)
ruff check \
  workflow_contracts.py artifact_contracts.py \
  ml_core.py ml_compat.py ml_diagnostics.py ml_distillation.py \
  ml_mlm_pretrain.py excel_utils.py entity_normalizer.py \
  model_loader.py llm_client.py llm_reranker.py

# 2. Mypy — strict on gated modules
mypy --strict --ignore-missing-imports --follow-imports=silent \
  ml_core.py ml_compat.py workflow_contracts.py artifact_contracts.py

# 3. Bandit — HIGH severity gate
bandit -r . --exclude ./tests,./ci,./fixtures -ll -q

# 4. Tests + coverage floor
PYTHONPATH=. pytest --cov=. --cov-fail-under=45 \
  --ignore=tests/test_heavy_modules.py -q
```

## Branching & Commits

- Branch from `main`: `feature/<short-topic>`, `fix/<short-topic>`,
  `refactor/<short-topic>`.
- Keep PRs focused. If scope grows, split into multiple PRs.
- Commit messages use imperative mood ("Add X", "Fix Y"), 72-char subject
  line, body explains **why** rather than **what**.
- Prefer several small commits to one giant commit — reviewers appreciate it.

## Pull Request Expectations

A PR is ready for review when:

- [ ] All quality gates above pass locally.
- [ ] New code has tests (`tests/test_<module>.py`); bug fixes include a
      regression test.
- [ ] Public API changes update `CLAUDE.md` and any affected docstrings.
- [ ] `experiments.jsonl` schema changes update `artifact_contracts.py`.
- [ ] Security-relevant changes (`model_loader.py`, `llm_*.py`,
      `llm_key_store.py`, `excel_utils.py`, `entity_normalizer.py`) are
      noted in the PR description and — when introducing new mitigations —
      documented in `SECURITY.md`.
- [ ] Coverage of touched files did not drop (read the coverage diff in CI).

## Coding Conventions

- **Type hints** mandatory on new public functions. Modules gated by
  `mypy --strict` (listed in `pyproject.toml` `[[tool.mypy.overrides]]`)
  must stay strict-clean; prefer `TYPE_CHECKING` imports and `cast()` over
  `# type: ignore`.
- **Docstrings** on public classes and functions. Describe **intent and
  contract**, not the implementation.
- **Comments** only when the *why* is non-obvious. Do not annotate code
  with PR numbers, author names, or "added for X" — that belongs in git
  history.
- **Error handling**: prefer specific exceptions (`ValueError`, `OSError`,
  `MemoryError`, `urllib3.exceptions.HTTPError`) to bare `except Exception`.
  Re-raise with `raise ... from exc` to preserve the chain.
- **Tkinter** calls only from the main thread. Worker threads communicate
  via progress callbacks (`task_runner.prepare_long_task_ui`).
- **Imports**: first-party modules sorted by `ruff --select I`. No
  circular imports — prefer deferring a function-local import to breaking
  a module.

## Architecture Boundaries

These boundaries are enforced by
`tests/test_architecture_boundaries.py`:

- `joblib.load` only from `model_loader.py`.
- UI mixins (`app_train.py`, `app_apply.py`, `app_cluster.py`) do not
  `import tkinter` *transitively* into service-layer modules
  (`*_service.py`, `workflow_contracts.py`, `ml_*.py`).
- `run_cluster` exits cleanly after pre-flight failures (no stuck
  `_processing = True`).

If your change requires relaxing one of these rules, discuss in the PR
description first.

## Tests

- Unit tests live in `tests/test_<module>.py`. Keep them headless —
  no Tkinter calls, no network, no disk outside `tmp_path`.
- E2E tests (`tests/test_e2e_*.py`) may exercise the full training /
  prediction pipeline on tiny fixtures (< 100 rows).
- Use `@pytest.mark.parametrize` for tabular test cases.
- Slow or heavy tests go behind `@pytest.mark.slow`; they are skipped
  in the default run.

## Dependency Changes

- Add new runtime deps to `requirements.txt` with a version range
  (`>=X.Y,<X+1`). No pinned exact versions outside `requirements.lock`.
- Prefer dependencies already in the graph — adding a new library is a
  conversation, not a detail.

## Questions

If the conventions above don't cover your case, open a draft PR and ask
in the description. We'd rather discuss early than review-reject late.
