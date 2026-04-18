# ADR-0002: Pipeline stages with frozen input snapshots

## Status
**Scaffolding only; behavioural migration pending Wave 3a.**

| Piece | State |
|---|---|
| Pure-function *types* (`VectorPack`, `ClusterResult`, `PostprocessResult`, `ExportSummary`) | ✅ Landed (`app_cluster_pipeline.py:25-49`) |
| `prepare_inputs` (Stage 1) | ✅ Real implementation (pure, no ML math) |
| `build_vectors` / `run_clustering` / `postprocess_clusters` / `export_cluster_outputs` | ⏳ **Raise `NotImplementedError`** — real math still lives in `app_cluster.run_cluster()` ~3307-4273 |
| Service-layer frozen snap (`ClusteringWorkflow.run` with `MappingProxyType`) | ✅ Landed (`cluster_workflow_service.py:68-70`) |
| Golden fixture + Hungarian-matching test | ✅ Landed (`tests/test_cluster_golden_fixture.py`) |
| `ClusterRunState` frozen+slots | ⏳ Pending Wave 3b (still `@dataclass` with 38 mutable fields) |
| `run_cluster()` ≤ 250 LOC | ⏳ Pending Wave 3a-3c (still 967 LOC) |

Until the stages above land, any caller driving the pipeline end-to-end
(e.g. `bank_reason_trainer cluster --allow-skeleton`) runs only Stage 1
and emits a `"Wave 3a"` note in its summary. This honesty is enforced by
the stubs raising instead of returning shape-matching `None`s — see the
`_STAGES_NOT_PORTED_MSG` sentinel in `app_cluster_pipeline.py`.

## Context
`run_cluster()` in `app_cluster.py` is ~969 LOC and mutates a wide
`ClusterRunState` (42 fields) across prepare → vectorize → cluster →
postprocess → export. A parallel pure-function module,
`app_cluster_pipeline.py`, holds the planned boundary with
`prepare_inputs`, `build_vectors`, `run_clustering`,
`postprocess_clusters`, `export_cluster_outputs`. Today the last four
are stubs; only `prepare_inputs` contains real logic. Service layer
(`cluster_workflow_service.ClusteringWorkflow.run`) already freezes the
caller-provided snap to `types.MappingProxyType` and forwards it to
each stage. Training mirrors the same shape via `app_train_service`.

## Decision
The canonical shape for any multi-stage ML run is:
1. **Pure-function stages**: `app_{domain}_pipeline.py` holds
   `stage_fn(prev_stage_result, snap) -> next_stage_result` — no
   tkinter, no mutable globals, no I/O beyond what the stage owns.
2. **Immutable stage-boundary snapshots**: each inter-stage result is a
   frozen dataclass (`dataclass(frozen=True, slots=True)`), or a
   narrowly typed NamedTuple; see `cluster_run_stages.Stage1..4Snapshot`.
3. **Frozen snap at the service boundary**: callers may pass a mutable
   dict; `ClusteringWorkflow.run` / `TrainingWorkflow.run` wraps it in
   `MappingProxyType` once and hands only the view to stages. If an
   already-frozen view arrives, it is forwarded as-is (no double wrap).
4. **Orchestrator stays thin**: the UI-tied `run_cluster()` /
   `run_training()` target is ≤ 250 LOC, delegating all math to the
   pipeline module and reading back typed stage snapshots.

## Consequences
+ Math migration is reviewer-safe because golden-label contract tests
  (`tests/test_cluster_golden_fixture.py`, Wave 3a) can pin the
  Hungarian-matched cluster-id equivalence on a ≥200-row fixture.
+ Stage snapshots make regression bisection tractable: any metric
  drift localises to one stage's frozen inputs/outputs.
+ Service layer is trivially forkable for CLI / batch / tests with
  zero Tk dependency.
− A full migration is ~2000 LOC and MUST be split into ≥5 PRs (3a
  vectorize+cluster, 3b postprocess+export+orchestrator, 3c training).
  See ADR-0003 for the state-freeze angle.

## References
- `cluster_workflow_service.ClusteringWorkflow` (service layer)
- `app_cluster_pipeline.py` (pipeline stubs, lines 215–239 to fill)
- `cluster_run_stages.Stage1..4Snapshot` (boundary types)
- Roadmap: Wave 3a/3b/3c in `/root/.claude/plans/graceful-cooking-taco.md`
