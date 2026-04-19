# ADR-0006: Mutation testing as the coverage-honesty gate

## Status
Accepted; nightly harness shipped in Wave 8.1.

## Implementation status (Wave 8.1)
- Per-module wrapper: `ci/mutation_smoke.py` (~170 LoC). Runs
  `mutmut run` against one source file with a focused subset of the
  test suite, then computes kill ratio and gates on `--threshold`.
  Cache dir is per-module so concurrent CI shards don't collide.
- Nightly workflow: `.github/workflows/mutation-score.yml` with a
  3-shard matrix (`ml_distillation.py` ≥ 75 %, `workflow_contracts.py`
  ≥ 70 %, `ml_core.py` ≥ 65 %). Each shard uploads a JSON summary
  artefact for trend tracking. Warn-only per §4 below — failure
  prints `::warning::` but does not break the workflow.
- pyproject `[tool.mutmut]` keeps `paths_to_mutate` explicit so an
  accidental `mutmut run` from the repo root cannot wander into the
  4 000-LoC UI mixins (which would never finish in 90 minutes).
- The original `app_cluster_pipeline.py` / `cluster_run_stages.py`
  candidates from §2 are deferred until the Wave 3a port lands —
  mutating `NotImplementedError` stubs has no signal value.
- **Mutmut version pin**: the wrapper targets the **2.x CLI**
  (`mutmut run --paths-to-mutate <file> --runner …` + `mutmut results
  --json`). Mutmut 3.x rewrote the CLI surface end-to-end:
  `--paths-to-mutate` was removed (the tool now reads exclusively from
  `[tool.mutmut] paths_to_mutate` in pyproject), `--json` is gone in
  favour of `mutmut export-cicd-stats`, and 3.x mutates *test files
  too*, which makes our "score the source module" contract incoherent.
  We therefore pin `mutmut>=2.4,<3` in both `pyproject.toml`'s
  `[project.optional-dependencies] dev` and the install step in
  `mutation-score.yml`. A future migration to 3.x is tracked as a
  follow-up, not a blocker.

## Context
Current coverage gate is `--cov-fail-under=45` (actual ~49 %). Line
coverage is a weak signal: trivial getters and `if log_cb: log_cb(…)`
branches inflate the number without protecting the math. Review 3
identified "coverage gaming" as a concrete risk of the 70 % target.

## Decision
1. Adopt **branch coverage** as the primary gate:
   `pytest --cov-branch --cov-fail-under=70` blocking on PRs.
2. Layer **mutation testing** on top, scoped to high-value modules:
   - `ml_core/*` (whatever lands after Wave 3a refactor)
   - `cluster_run_stages.py`
   - `app_cluster_pipeline.py`
   - `app_train_service.py`
   - `llm_reranker.py` parsing helpers
3. Tool: `mutmut` (pure Python, zero-config, works with pytest).
   Target: ≥ 75 % mutation score per listed module. Nightly job in
   `mutation-score.yml`; PR-level job is warn-only because mutmut
   runs can exceed 6 hours on the full ml_* suite.
4. Mutation score is **not** a PR gate but **is** a release gate —
   the commit that bumps the version tag must show the current
   mutation-score artefact ≥ target.
5. **Hypothesis** property tests in the same modules (target: ≥ 100
   properties), so mutmut has enough diverse inputs to kill
   mutants. Every property test has a deterministic seed
   (`@given(...)` + `settings(derandomize=True)` on slow cases).

## Consequences
+ Covers the "70 % trivial" failure mode: a mutant that only
  exercises the getter path dies immediately.
+ Hypothesis catches invariant violations (monotonicity of ECE under
  temperature scaling, one-hot Brier bounds, label-id equivalence
  after Hungarian matching) that line coverage will never surface.
− Mutation runtime is real — budget a dedicated nightly runner,
  not the main CI pool. The 6h limit is realistic on GitHub's
  standard runners; use `matrix.include` to shard by module.
− Some mutations are semantically equivalent (e.g. `x > 0` vs
  `x >= 0` on a strictly positive set) and produce false negatives;
  the allow-list lives alongside the module with a one-line reason.

## References
- `pyproject.toml` `[tool.coverage]` (add `branch = true`)
- New workflow: `.github/workflows/mutation-score.yml`
- Candidate hypothesis sites: `_compute_ece_mce`, `_compute_brier_score`,
  `_parse_rerank_response`, `cluster_run_stages.Stage[1-4]Snapshot` round-trip
- Roadmap: Wave 5 in `/root/.claude/plans/graceful-cooking-taco.md`
