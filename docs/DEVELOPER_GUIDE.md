# Developer guide

## Add a new clustering algorithm
1. Implement pure logic in `app_cluster_service.py`, `app_cluster_pipeline.py`,
   or a dedicated `cluster_*_service.py`.
2. Add validation/contract checks in the workflow or loader layers
   (`app_cluster_workflow.py`, `workflow_contracts.py`).
3. Wire UI controls in `ui_widgets/cluster_panel.py` (and, if the algo
   needs a new snap-key, extend `_SUPPORTED_COMBOS` and the CLI
   router in `bank_reason_trainer/cli.py`).
4. Add unit tests + integration smoke (`tests/test_cluster_*`).

## Add a new vectorizer
1. Add vectorizer implementation in `ml_vectorizers.py` (avoid heavy import side-effects).
2. Add compatibility patches in `ml_compat.py` if needed.
3. Register selection path in train/apply workflows.
4. Add tests for fallback behavior and serialization compatibility.
