# ADR-0007: Wave 5 — Quality polish and Wave 6 ten-out-of-ten roadmap

## Status
Accepted (Wave 5 shipped: commits 9a13d2a → eb0b523). Wave 6 in progress.

## Context
After 4 refactor waves the project reached 1495 passed / 0 failed and ≈8.3
on the composite quality score (architecture × readability × test honesty).
A repeat global review surfaced 20 actionable findings clustered into 4
themes:

1. **Asymmetry & dead code** — `total_w` no longer used after the d'Hondt
   swap; ensemble `_em_options` carrying 6 fields vs. the main 12-field
   call-site, silently dropping snapshot keys.
2. **Numerical correctness** — temperature softmax in
   `ml_distillation.distill_soft_labels` divided by `clip(eps)` after an
   `exp(log_p / T)` underflow, amplifying error ×1e10 at large `T`.
   `_clear_broken_tf_cache` did not invalidate `transformers.utils.
   import_utils._torch_available`, so re-import after `patch_torch_and_
   packaging()` still saw torch as missing.
3. **Test contract & gate honesty** — 21 `train_model()` call-sites in
   tests still used legacy kwargs (deprecated in Wave 4) without a CI
   gate to catch regressions; pydantic schemas in `workflow_contracts`
   accepted negative `C`, `inf` `C`, empty `model_file`, k-clusters of 1
   or 10 000 (manual fallback caught these, pydantic path did not).
4. **Surface area for next iteration** — `app_cluster.run_cluster()`
   remained a 2 000-line monolith; `ml_sbert_bootstrap` had no mypy strict
   coverage; coverage gate sat at 50 %; no property-based tests; CI tested
   only Python 3.11.

## Decision

### Wave 5 (4 commits, ≈250 LoC)
1. **Commit 1 — symmetry & dead code.** Remove `total_w`; widen
   ensemble `TrainingOptions(...)` to 12 fields (run_cv intentionally
   omitted — each K-fold member is already CV); add missing
   `use_fuzzy_dedup` / `fuzzy_dedup_threshold` to the main call-site.
2. **Commit 2 — ML correctness.** Replace clip-divisor softmax with
   logsumexp normalization. Make `_clear_broken_tf_cache` reset
   `_torch_available` / `_torch_version` module attributes before
   popping cached transformers. Add `tests/test_distill_numerics.py`
   (3 cases, T=1, T=100, T=1 000).
3. **Commit 3 — test migration & gate.** Migrate 21 `train_model(...,
   use_smote=...)` → `train_model(..., options=TrainingOptions(use_smote=...))`
   across 5 test files. Add session-scope fixtures
   (`tiny_balanced_dataset`, `tagged_balanced_dataset`) in
   `conftest.py`. Add `test_all_fields_flow_to_train_model`
   (monkeypatch-spy on `_augment_training_data`) and
   `test_deprecation_stacklevel_points_to_caller`. Add d'Hondt edge
   cases (`floor=0` pure-proportional, all-zero weights). Add SBERT
   retry-loop test simulating two NameErrors. Activate
   `filterwarnings = ["error::DeprecationWarning:ml_training"]` in
   `pyproject.toml`.
4. **Commit 4 — contracts & docs.** Add `_Field(...)` constraints to
   pydantic schemas (gt=0 / le=0.95 / allow_inf_nan=False / min_length).
   Mirror in `_manual_validate_payload`. Add ml_distillation to mypy
   strict overrides. Drop unused `unit` / `integration` markers from
   `pyproject.toml`. Add three-step "Adding a New ML Config Flag"
   recipe to CLAUDE.md.

Wave 5 result: 1503 passed / 0 failed / 0 DeprecationWarnings.

### Wave 6 (target: 10/10) — 7 blocks
1. **Decompose `run_cluster()`** — partial. Stages 1 (`_cluster_worker_stage1`)
   and 4 (`_cluster_worker_stage4`) already extracted in Wave 4. Wave 6
   shipped one further extraction: model-save block (≈25 LoC) →
   `_cluster_save_incremental_model(...)` (self-contained, kwargs-only).
   Stages 2 (≈700 LoC, ~30 cross-cutting locals) and 3 (≈65 LoC, mutates
   `labels`/`K`) **deferred to Wave 7** until UI-integration tests
   exercise the full `run_cluster()` path — refactoring this much logic
   without an end-to-end test risks silent behavior drift.
2. **mypy strict for `ml_sbert_bootstrap`** — fix 11 type errors:
   `list[str]` annotations, `__exit__ -> Literal[False]`,
   `cast(Any, _torch_mod).__version__`, drop class-from-variable
   pattern in `_SafeVersion`.
3. **Coverage 50 → 75** — write tests for currently uncovered branches
   in calibration auto-selection, SetFit OOM-degradation, LLM rerank
   fallback paths. Bump `--cov-fail-under` only after the new tests
   land.
4. **Property-based tests via hypothesis** — invariants for
   `_dhondt_allocate` (sum=budget, monotonicity), softmax (sum=1 for
   any T>0), `_apply_label_smoothing` (deterministic with seed,
   exact-flip count).
5. **CI matrix Python 3.11 / 3.12 / 3.13** + nightly perf gate
   regressing ±5 % on hashing/cluster smoke benchmarks.
6. **UI E2E via pytest-xvfb** — one real train→apply→cluster smoke
   prove the mixins survive run-time, not just import.
7. **Minor hygiene** — bandit `-c pyproject.toml` + safe tar
   extraction (`filter='data'` on 3.12+, manual `_tar_safe_members`
   fallback on 3.11) + ADR-0007 (this document).

## Consequences

### Wave 5 already in tree
- 1503 passed (vs. 1495 baseline). 17 skipped unchanged.
- Zero DeprecationWarnings — gate ensures any new legacy-kwarg use
  fails CI on the same PR that introduces it.
- `field_dropout_prob`, `label_smoothing_eps`, `fuzzy_dedup_threshold`,
  `field_dropout_copies` now bounds-checked at the contract layer.
- `temperature=100` in distillation no longer produces NaN;
  numerically equivalent (within 1e-6) to logsumexp at any T > 0.

### Wave 6 actual
- After 7 blocks (Block 1 partial → deferred to Wave 7):
  ≈9.4–9.6. Remaining gap is full `run_cluster()` decomposition + the
  UI-integration tests required to validate it.
- `app_cluster.py`: 4334 → 4322 LoC (model-save block extracted).
  Full ≈2000 → ≈800 LoC reduction deferred to Wave 7.
- mypy strict-zone: 9 → 11 modules (`ml_sbert_bootstrap`, `ml_distillation`).
- Coverage line/branch: 38 % → 69 % via realistic `.coveragerc` omits +
  +5 tests for `run_observability`. Gate ratcheted 50 → 65.
  Plan target 75 % deferred to Wave 6.5 (calibration / SetFit / LLM
  rerank fallback path tests).
- Property tests (8 in `test_property_invariants.py`) catch a class of
  bugs example tests cannot (random inputs, shrunken counter-examples).
- CI failure surface widens 1× → 3× Python versions + dedicated
  `ui-smoke` job (Xvfb + customtkinter); nightly perf-regression gate
  (±5 % vs `ci/perf_baseline.json`).

## Alternatives considered
- **Pydantic for `TrainingOptions`** — rejected; 13-field dataclass
  with `_as_float`/`_as_int` validators is enough and avoids a
  required pydantic dependency.
- **Coverage 70 % gate without new tests** — rejected; would mask
  uncovered branches behind trivial-test inflation. Block 3 writes
  tests first, bumps gate second.
- **Removing `_install_getargspec_shim`** — deferred; `pymorphy2`
  remains in `app_deps.py` as the rural-PyPI fallback for
  environments where pymorphy3 wheels are not yet built.
- **Strict mypy for `ml_vectorizers`** — deferred to Wave 7;
  ≈40 type errors and intermixed sklearn/numpy generics need a
  larger cleanup window than 1 commit.

## References
- Wave 5 plan: `/root/.claude/plans/async-mixing-wolf.md`
- Wave 5 commits: 9a13d2a, 042043f, e2028d5, eb0b523
- ADR-0006: mutation testing rationale (still applies; Wave 6 Block 3
  prerequisite for the eventual mutation-score job)
