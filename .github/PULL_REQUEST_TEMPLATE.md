<!--
Thanks for contributing! Please fill out the sections below.
Delete sections that aren't applicable.
-->

## Summary

<!-- One or two sentences: what changes, and why. -->

## Motivation

<!--
Link the issue or describe the problem. If this is a refactor without a
user-visible change, explain what maintainability or performance property
it unlocks.
-->

## Changes

<!-- Bulleted list of the main changes. File paths help reviewers. -->

-
-

## Test Plan

<!-- What you actually ran locally. -->

- [ ] `PYTHONPATH=. pytest --cov=. --cov-fail-under=45 --ignore=tests/test_heavy_modules.py -q`
- [ ] `ruff check <strict-module-list>` (see CONTRIBUTING.md)
- [ ] `mypy --strict --ignore-missing-imports --follow-imports=silent ml_core.py ml_compat.py workflow_contracts.py artifact_contracts.py`
- [ ] `bandit -r . --exclude ./tests,./ci,./fixtures -ll -q`
- [ ] Ran the app via `python bootstrap_run.py` and exercised the affected tab.

## Security Considerations

<!--
Only required when touching model_loader.py, llm_*.py, excel_utils.py,
entity_normalizer.py, or workflow_contracts.py. Otherwise delete.

- What trust boundary did you cross?
- What new inputs does the change expose to attacker-controlled data?
- Did you update SECURITY.md?
-->

## Checklist

- [ ] Commits follow the CONTRIBUTING.md style (imperative, < 72-char subject).
- [ ] Touched files have docstrings for new public API.
- [ ] `CLAUDE.md` updated if module map or developer workflow changed.
- [ ] No new `except Exception:` in ML modules (prefer specific exceptions).
- [ ] No new `joblib.load` / `pickle.load` outside `model_loader.py`.
- [ ] Coverage did not drop for touched files.

## Screenshots / Logs

<!-- Optional. Helpful for UI changes or perf-related work. -->
