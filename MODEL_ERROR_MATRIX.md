# Model loading error/recovery matrix

| Error class | Typical code markers | Reaction strategy | Recoverable |
|---|---|---|---|
| `ModelLoadError` | `UNSUPPORTED_EXTENSION`, `SHA256_MISMATCH`, deserialize errors | Show user-safe message + allow fallback path (choose another model/retry). | Yes |
| `SchemaError` | `ARTIFACT_TYPE_MISMATCH`, `SCHEMA_MISSING`, `SCHEMA_TYPE_INVALID`, `SCHEMA_UNSUPPORTED` | Hard fail + migration hint (re-export/retrain with supported schema). | No |
| Unexpected exception | N/A | Capture traceback + mark incident in logs (`UNEXPECTED_INCIDENT`). | No |

## Structured log fields

For long-running operations log with the following fields:
- `event`
- `stage`
- `file`
- `rows`
- `duration_sec`
- `error_class`
- `correlation_id`

The helper `model_error_policy.log_structured_event(...)` emits these fields in key-value format.
