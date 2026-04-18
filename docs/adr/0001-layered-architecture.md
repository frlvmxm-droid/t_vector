# ADR-0001: Layered architecture for UI/Workflow/Service

## Status
Accepted

## Context
Large tkinter modules mixed UI rendering, orchestration and domain logic, which made testing and safe evolution difficult.

## Decision
Adopt layered modules:
- UI layer (`app_*.py`, `*_view.py`) for binding/state rendering
- Workflow layer (`*_workflow.py`, `workflow_*`) for validation/orchestration contracts
- Service layer (`*_service.py`, `cluster_*_service.py`) for pure/use-case logic
- Loader/contract layer (`model_loader.py`, `artifact_contracts.py`) for artifact safety and schema checks

## Consequences
- Easier unit and integration testing
- Safer model loading migration path
- Lower coupling for future algorithms/vectorizers
