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

### Wave 6.5 actual (calibration / SetFit / LLM rerank fallback paths)
- +55 unit tests across three new files:
  `tests/test_calibration_paths.py` (14: auto-pick sigmoid/isotonic,
  ECE/MCE/Brier graceful fallbacks, single-class → LogReg shortcut),
  `tests/test_setfit_oom_paths.py` (19: VRAM band selection 0–80 GB,
  env-overrides `BRT_SETFIT_MAX_TRAIN_OVERRIDE` /
  `BRT_SETFIT_MAX_PAIRS_OVERRIDE`, silent `_cuda_cleanup` no-ops,
  `_is_setfit_config_missing_error` substring detection for
  `EntryNotFoundError` / `config_setfit` / `404`),
  `tests/test_llm_rerank_fallback.py` (22: response parsing whitespace /
  multiline / `Класс:` prefix, `_cache_read` missing-file / invalid-JSON
  / non-string-label, `_cache_write` OSError swallow, in-batch dedup +
  disk-cache hit/miss/write-back, generic-Exception path).
- `.coveragerc` omits expanded with `app_deps.py`, `t5_summarizer.py`,
  `ml_mlm_pretrain.py` — heavy HF-dependent modules covered by nightly
  smoke, not unit tests; consistent rationale with existing UI-mixin
  exclusions.
- Coverage line/branch: 69 % → **73.15 %** on the testable core. Gate
  ratcheted 65 → 72 (+1.15 pt safety margin). 1510 passed / 0 failed /
  23 skipped.

### Wave 8.3 actual (real CLI for train/apply)
`bank_reason_trainer train` and `apply` are no longer skeletons.

* `train` reads `--data` (xlsx/csv) via `excel_utils.open_tabular`, picks
  `--text-col`/`--label-col` from headers, builds a TF-IDF FeatureUnion
  (word 1-2 + char 3-5), calls `TrainingWorkflow.fit_and_evaluate`,
  persists a v1 `train_model_bundle` joblib via
  `TrainingWorkflow.persist_artifact`. `--snap` accepts a JSON dict
  whose keys map to `TrainingOptions` fields (calib_method, use_smote,
  field_dropout_*, etc.); unknown keys are dropped silently rather
  than crashing on a typo. Fails fast (exit 1) on `<2` distinct
  labels or `<4` rows.
* `apply` loads the bundle through `model_loader.load_model_artifact`
  (SHA-256 + artifact-identity validated), runs `predict_proba` and
  `apply_prediction_service.predict_with_thresholds` (per-class
  thresholds from the bundle, global floor from `--threshold`),
  writes CSV (default) or XLSX (`.xlsx` extension) with
  `[text, predicted_label, confidence, needs_review]` columns.
* Tests: `tests/test_cli_entrypoint.py` extended with a real round-trip
  (`test_apply_round_trip_train_then_predict`), missing-data /
  too-few-classes / missing-model error paths, and a parser-level
  contract guard. Net +7 tests.
* Cluster CLI keeps the `--allow-skeleton` gate **only for unsupported
  combos** (sbert/setfit + hdbscan/agglo/lda). The default `tfidf` +
  `kmeans` combo is real after the Wave 3a slice port — see next section.

### Wave 3a slice port (tfidf + kmeans end-to-end) — shipped
The four `NotImplementedError` stubs in `app_cluster_pipeline.py`
(`build_vectors`, `run_clustering`, `postprocess_clusters`,
`export_cluster_outputs`) now have a real implementation for the
narrowest useful combo: `cluster_vec_mode="tfidf"` +
`cluster_algo="kmeans"`. Other combos still raise — the message names
the slice and points to ADR-0002.

* `build_vectors` reads `text_col` from each `--files` path via
  `excel_utils.open_tabular`, deduplicates blank cells, then fits
  `TfidfVectorizer(analyzer="word", ngram_range=(1,2), sublinear_tf=True)`
  with an adaptive `min_df` rule (`1` for `<5k`, `2` for `<50k`, `3`
  otherwise — mirrors the Tk-bound `run_cluster()` heuristic).
* `run_clustering` fits `MiniBatchKMeans(n_clusters=k, init="k-means++",
  batch_size=1024, max_iter=300)` with the snap-supplied `random_state`
  / `n_init`. Validates `k >= 2` ahead of fit; ValueError on `k=1`.
* `postprocess_clusters` computes per-cluster sizes and the top-5 TF-IDF
  feature names per cluster (sparse-row mean → argpartition). No LLM
  naming, no silhouette — those stay in the Tk path until they need
  headless coverage.
* `export_cluster_outputs` writes a single CSV with header
  `[text, cluster_id, top_keywords]`. XLSX export and the multi-sheet
  variants from `run_cluster()` are out of scope for the slice.
* CLI semantics changed: `cluster --files X` defaults to the supported
  combo and now requires `--out`; `--text-col` (default `"text"`) and
  `--k-clusters` (default `8`) round out the args. `--allow-skeleton` is
  still required to acknowledge the prepare-only fallback for
  unsupported combos.
* Tests: `tests/test_cli_entrypoint.py::test_cluster_supported_combo_runs_full_pipeline`
  is the new E2E baseline (10-row CSV → k=2 → output CSV with cluster
  ids). `tests/test_cluster_pipeline_smoke.py` retargeted from "all
  stages raise" to "unsupported combos raise" — the slice is now the
  positive case.
* **Slice extension (Wave 3a.1):** `cluster_algo` set expanded from
  `{"kmeans"}` to `{"kmeans", "agglo"}`. Agglo uses sklearn's
  `AgglomerativeClustering(linkage="ward")` with a 5 000-row hard cap
  (Ward is O(n²) in memory); kmeans keeps the sparse matrix, agglo
  densifies via `.toarray()`. Same validation / postprocess / export
  paths — the dispatch is a single `if/elif` inside `run_clustering`.
  No new deps (sklearn ships AgglomerativeClustering). +1 E2E test
  (`test_cluster_agglo_combo_runs_full_pipeline`). 1551 passed / 18
  skipped.
* **Coverage gate ratcheted 72 → 73** after the slice port pushed
  measured coverage from 73.15 % to 74.42 % (margin 1.42 pt, same
  philosophy as the Wave 6.5 bump).

### Wave 7 safety net (UI-driven cluster E2E) — shipped
Before any stage 2/3 extraction from `app_cluster.run_cluster()`, the
refactor needs a behavioral baseline that exercises the UI wiring +
threading hand-off + CLUST_DIR output contract. Pipeline-only smoke
tests (`test_cluster_pipeline_smoke.py`,
`test_cluster_supported_combo_runs_full_pipeline`) cover the math but
not the tk-bound orchestration the refactor is going to rewrite.

* `tests/test_ui_cluster_e2e.py` (new) drives the full method:
  constructs `App()`, sets `cluster_files` + the minimum tk.Var fields
  for the `tfidf` + `kmeans` slice, monkey-patches `CLUST_DIR` to
  `tmp_path`, calls `app.run_cluster()`, pumps the Tk event loop via
  `app.update()` until `self._processing` flips to `False`, then
  asserts a `*clustered*.xlsx` file exists in the patched directory.
* Skip pattern matches `test_ui_smoke.py` (no DISPLAY / no tkinter /
  no customtkinter → skip). Runs in CI under the existing `ui-smoke`
  job (`.github/workflows/quality-gates.yml`) which installs
  `python3-tk` + `customtkinter` and wraps pytest with `xvfb-run -a`.
* This test pins: output file naming pattern, `_processing` flag
  lifecycle, and the "output next to CLUST_DIR" contract. A refactor
  that accidentally breaks any of these fails the gate.

### Wave 7 execution plan (stage 2/3 extraction) — multi-session
The safety-net test above unblocks the refactor; the extraction itself
is deliberately broken into small, testable sub-commits so CI catches
drift at each step:

1. **Port remaining stage 2 combos into `app_cluster_pipeline.py`,
   one per commit.** The slice already covers `tfidf` + `{kmeans,
   agglo}`. Next candidates in order of simplicity: `tfidf` + `lda`
   (sklearn-only), `tfidf` + `hdbscan` (needs `hdbscan` dep in the
   ml-extras), `sbert` + `kmeans` (pulls `sentence_transformers`),
   `combo` / `ensemble` (composite paths, hardest — defer to last).
   Each commit adds one combo + a pipeline smoke test; CI runs the
   E2E against the existing `tfidf+kmeans` path to guard against
   regressions in the shared code.
2. **Extract stage 3 (LLM naming + `kw_dict` + quality metrics) into
   a pure function.** Stage 3 is ~44 LoC and the LLM-naming helper
   (`_cluster_step_llm_naming`) is already a discrete method;
   promoting it to a pipeline function that takes
   `(labels, X_clean, Xv, snap)` and returns a `PostprocessResult`
   addendum is mechanical.
3. **Finally, rewrite `run_cluster()` to delegate to
   `ClusteringWorkflow.run()`.** Only after (1) and (2) land — the
   method shrinks from ~950 LoC to ~100 (UI wiring + worker thread +
   export). The UI-E2E test gate from above validates the shrink
   did not break externally-visible behavior.

Wave 7.1 (safety net) is Complete. 7.2/7.3 are tractable follow-ups
now that the gate exists.

### Wave 7 (full run_cluster decomposition) — still deferred
Investigation in this wave confirmed Wave 7.2/7.3 (extract stages 2–3
from `app_cluster.run_cluster()`) cannot ship safely yet:

* **Hard prereq missing.** `app_cluster_pipeline.build_vectors`,
  `run_clustering`, `postprocess_clusters`, `export_cluster_outputs`
  currently raise `NotImplementedError` — the real math still lives in
  the Tk-bound 700-line stage 2 of `app_cluster.run_cluster()`. Without
  the Wave 3a port (ADR-0002), there is no UI-free orchestration surface
  to run as an E2E behavioral baseline.
* **The "test under Xvfb instead" alternative was rejected.** `run_cluster`
  binds tightly to `self._right_tabs`, `begin_long_task(run_button=…)`,
  `prepare_long_task_ui(owner=self, …)` and a swarm of `tk.Var`-backed
  status fields. Mocking that stack fakes the very thing the safety net
  is supposed to verify (real widget order-of-update, threading, cancel
  flow). The existing `tests/test_ui_smoke.py` proves the mixins boot —
  not that they cluster correctly.
* **Sequenced fix.** A future wave should: (a) port stage 2 sub-blocks
  (TF-IDF vectorize, KMeans/HDBSCAN fit) into pure functions in
  `app_cluster_pipeline.py`, (b) write unit tests for each as the
  baseline, (c) only then re-write `app_cluster.run_cluster()` to call
  those functions. Refactoring 700 LoC of stage 2 ahead of (a)/(b) ships
  silent-drift risk for nominal LoC reduction; this wave declines that
  trade.
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
