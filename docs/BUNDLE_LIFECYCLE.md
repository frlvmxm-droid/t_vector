# Model bundle lifecycle

1. **Train/export**: model is serialized with `artifact_type` + `schema_version`.
2. **Load gate**: `load_model_artifact(...)` validates extension, hash (optional), artifact type, schema compatibility.
3. **Domain normalization**: service loaders (`cluster_model_loader.py`, apply/train loaders) normalize fields.
4. **Runtime apply**: workflow/service executes prediction or clustering.
5. **Diagnostics**: on failure, error policy classifies failure and optional diagnostic report can be exported.
