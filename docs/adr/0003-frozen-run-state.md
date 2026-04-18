# ADR-0003: Frozen run-state as the default

## Status
Proposed (implementation in Wave 3b/3c).

## Context
`ClusterRunState` is a 42-field mutable `@dataclass` populated
incrementally across four stage banners inside `run_cluster()`.
`run_training()` follows the same pattern implicitly via local
variables. Mutability caused three classes of bugs visible in git
history:
- Stage N reads a field that stage N-1 forgot to set (soft `None`).
- A retry path mutates a field that later stages treat as write-once.
- UI thread peeks at partially-filled state between stages.

## Decision
1. All multi-stage ML state holders are
   `@dataclass(frozen=True, slots=True)` unless there is a documented,
   enumerated reason to allow mutation.
2. Cross-stage state flows forward by returning a new stage snapshot
   (`StageKSnapshot`), never by mutating a shared record. Optional
   fields are `Optional[T]` and set only at construction.
3. The UI thread reads only from the latest snapshot handle; it never
   reaches into a stage-in-progress. UI updates (labels, progress
   bars) use `self.after(0, …)` with the snapshot's values captured
   at post time.
4. When a stage needs an accumulator (counters, partial caches), it
   is local to that stage and exits through the next snapshot, not
   via a shared field.
5. Conversion sites: anything that used to `state.field = …` becomes
   `state = replace(state, field=…)` at the stage boundary, not in a
   deep inner loop.

## Consequences
+ Thread-safety by construction; no "who wrote field X last" race.
+ Stage boundaries become documented contracts; schema drift is a
  compile-time (mypy-strict) signal.
+ Pairs with ADR-0002 so tests can assert pure input → pure output.
− Migration of `ClusterRunState` touches every call-site inside
  `run_cluster()` (≥150 writes). Budget 2–3 days inside Wave 3b.
− `replace(state, …)` allocations are cheap but non-zero; hot loops
  must not use it — move the accumulator into a local.

## References
- `cluster_run_stages.py` (stage snapshots, already frozen)
- `app_cluster.py` `ClusterRunState` (target of migration)
- `app_train.py` `run_training()` (target for Wave 3c)
