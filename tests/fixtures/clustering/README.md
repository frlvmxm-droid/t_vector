# Golden cluster fixtures

Synthetic fixture used to pin clustering-pipeline behaviour across the
Wave 3a math migration. Each fixture file describes a deterministic
input + expected cluster composition, and is verified via
**Hungarian matching** — i.e. we require the migrated `run_clustering`
to produce the same *partition* of points into clusters, up to a
permutation of cluster IDs.

See:
- `docs/adr/0002-pipeline-stages-and-snapshots.md`
- `tests/test_cluster_golden_fixture.py`
- `/root/.claude/plans/graceful-cooking-taco.md` (Wave 3a)

## File format (`*.json`)

```json
{
  "name": "three-blobs-k3",
  "seed": 20260418,
  "n_points": 120,
  "n_features": 16,
  "n_clusters": 3,
  "kmeans_n_init": 10,
  "kmeans_max_iter": 300,
  "expected_cluster_sizes_sorted": [40, 40, 40],
  "expected_purity_min": 0.95
}
```

Vectors are generated on-the-fly from `seed + n_points + n_features`
via `numpy.random.default_rng` — no binary blobs committed.

## How to extend

1. Add a new `.json` here with a unique `name` and deterministic
   `seed`.
2. Regenerate the ground truth locally by running the test with
   `BRT_CLUSTER_GOLDEN_RECORD=1` — it prints the expected sizes and
   purity to stderr, which you paste into the fixture.
3. Commit the JSON alongside the code change.
