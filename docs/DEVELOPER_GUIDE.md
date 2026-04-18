# Developer guide

## Add a new clustering algorithm
1. Implement pure logic in `app_cluster_service.py` or dedicated `cluster_*_service.py`.
2. Add validation/contract checks in workflow or loader layers.
3. Wire UI controls in `app_cluster.py`/`app_cluster_view.py` only.
4. Add unit tests + integration smoke (`tests/test_cluster_*`).

## Add a new vectorizer
1. Add vectorizer implementation in `ml_vectorizers.py` (avoid heavy import side-effects).
2. Add compatibility patches in `ml_compat.py` if needed.
3. Register selection path in train/apply workflows.
4. Add tests for fallback behavior and serialization compatibility.
