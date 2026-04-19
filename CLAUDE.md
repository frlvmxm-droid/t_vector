# CLAUDE.md — Developer Guide

BankReasonTrainer is a desktop Tkinter app for classifying Russian bank dialog texts into reason-of-contact categories. It supports training (TF-IDF/SBERT/SetFit), batch prediction, and unsupervised clustering with optional LLM-powered naming.

---

## Quick Start

```bash
# Install deps + launch app
python bootstrap_run.py        # cross-platform (Windows/Linux/macOS)

# Or directly:
python app.py
```

Shell shortcuts: `run.sh` (Linux/macOS), `run_app.bat` (Windows).

### Reproducible install (CI / Docker / contributors)

`uv.lock` is the source of truth for fully-pinned wheels (ADR-0008).
`requirements.txt` is kept for the desktop launcher only.

```bash
# Install pinned set (matches CI + Docker exactly)
uv sync --frozen --extra dev

# Bumping a dep
$EDITOR pyproject.toml          # edit [project.dependencies]
uv lock                          # regenerate uv.lock
git add pyproject.toml uv.lock   # commit both — CI's `uv lock --check` enforces

# Build the dev image
docker build -t bank-reason-trainer:dev .
docker run --rm bank-reason-trainer:dev    # runs pytest
```

### Headless CLI (no tkinter)

```bash
# Train: TF-IDF (word 1-2 + char 3-5) + LinearSVC(Calibrated)
python -m bank_reason_trainer train \
    --data train.xlsx --out model.joblib \
    --text-col text --label-col label \
    [--snap snap.json]   # optional: TrainingOptions overrides

# Apply: predict with per-class thresholds from the bundle
python -m bank_reason_trainer apply \
    --model model.joblib --data in.xlsx --out out.xlsx \
    --text-col text [--threshold 0.5]

# Cluster: TF-IDF + {KMeans | Agglomerative | LDA | HDBSCAN}, or
# SBERT + KMeans, or Combo (TF-IDF→SVD→L2 + SBERT→L2 hstack) + KMeans
# (Wave 3a slice + 7.2/7.3/7.4/7.5a extensions — real end-to-end)
python -m bank_reason_trainer cluster --files a.xlsx b.xlsx \
    --out clusters.csv --text-col text --k-clusters 8
# Pick algo via snap: {"cluster_vec_mode": "tfidf", "cluster_algo": "agglo"}
# or:                 {"cluster_vec_mode": "sbert", "cluster_algo": "kmeans",
#                      "sbert_model": "cointegrated/rubert-tiny2"}
# or (combo):         {"cluster_vec_mode": "combo", "cluster_algo": "kmeans",
#                      "combo_svd_dim": 200, "combo_alpha": 0.5,
#                      "sbert_model": "cointegrated/rubert-tiny2"}

# Cluster: other combos (setfit / BERTopic / fastopic / ensemble) still
# need --allow-skeleton; only prepare_inputs runs, the rest live inside
# the Tk-bound app_cluster.run_cluster() until they are ported.
python -m bank_reason_trainer cluster --files a.xlsx b.xlsx \
    --snap ensemble_snap.json --allow-skeleton
```

`train` and `apply` are real (Wave 8.3): TF-IDF features, joblib bundle
round-trip, CSV/XLSX I/O. `cluster` is real end-to-end for
`tfidf` + {`kmeans`, `agglo`, `lda`, `hdbscan`}, `sbert` + `kmeans`,
and `combo` + `kmeans` after the Wave 3a slice port + Wave 7.2–7.5a
extensions — they stream text from `--files`, fit the chosen vectorizer
+ clusterer, and write `text,cluster_id,top_keywords` to `--out`.
Agglomerative is capped at 5 000 rows (Ward linkage is O(n²) in memory);
HDBSCAN discovers K itself (the `--k-clusters` flag is ignored); SBERT
downloads the model from HuggingFace Hub on first run (cached in
`SBERT_LOCAL_DIR`); combo mode blends TF-IDF-SVD and SBERT vectors via
`combo_alpha` (0..1). Other combos (BERTopic, SetFit, FASTopic,
`ensemble`) still raise `NotImplementedError` and require
`--allow-skeleton` for the prepare-only fallback — see
`docs/adr/0002-pipeline-stages-and-snapshots.md` and
`docs/adr/0007-wave5-quality-polish.md`.

---

## Running Tests

```bash
# All tests via pytest (recommended)
PYTHONPATH=. pytest -q

# Specific suites
PYTHONPATH=. pytest tests/test_e2e_train_predict.py -v    # E2E train → predict
PYTHONPATH=. pytest tests/test_e2e_excel.py -v            # Excel round-trip
PYTHONPATH=. pytest tests/test_workflow_contracts.py -v   # Pydantic contracts
PYTHONPATH=. pytest tests/test_all_compat.py -v           # 143 unit tests from test_all.py

# Legacy custom runner (prints PASS/FAIL summary)
python test_all.py
```

Requirements for E2E tests: `scikit-learn numpy scipy joblib openpyxl` (core deps, always installed).

---

## Module Map

### Entry Points
| File | Role |
|------|------|
| `app.py` | Main `App` class — tkinter root, state, session save/restore |
| `app_train.py` | `TrainTabMixin` — training UI + ML orchestration |
| `app_apply.py` | `ApplyTabMixin` — batch prediction UI |
| `app_cluster.py` | `ClusterTabMixin` — clustering UI (2 035-line `run_cluster()`) |
| `bootstrap_run.py` | Entry point: checks Python ≥3.9, installs deps, launches |

### Service Layer (no tkinter, testable in isolation)
| File | Role |
|------|------|
| `app_train_service.py` | `TrainingWorkflow` — thin wrapper over `train_model()` |
| `apply_prediction_service.py` | `predict_with_thresholds()`, `validate_apply_bundle()` |
| `cluster_workflow_service.py` | `ClusteringWorkflow.run()` — orchestrates 4 pipeline stages |
| `app_cluster_pipeline.py` | Pure pipeline functions: `prepare_inputs`, `build_vectors`, `run_clustering`, `postprocess_clusters`, `export_cluster_outputs` |

### ML Core
| File | Role |
|------|------|
| `ml_training.py` | `train_model()` — LinearSVC + CalibratedClassifierCV, SMOTE, nn_mix, label smoothing |
| `ml_vectorizers.py` | `make_hybrid_vectorizer()` — TF-IDF (char+word), PerFieldVectorizer, Lemmatizer (pymorphy3→pymorphy2), SBERT, DeBERTa, MetaFeatureExtractor |
| `ml_setfit.py` | `SetFitClassifier` — sklearn-compatible few-shot classifier via HuggingFace SetFit |
| `ml_diagnostics.py` | `rank_for_active_learning()`, `find_cluster_representative_texts()`, `merge_similar_clusters()` |
| `ml_distillation.py` | `distill_soft_labels()` — teacher→student knowledge distillation |
| `ml_mlm_pretrain.py` | `pretrain_mlm()` — domain MLM pretraining via HuggingFace Trainer |
| `llm_reranker.py` | `rerank_top_k()` — LLM-based re-ranking of low-confidence predictions |

### Infrastructure
| File | Role |
|------|------|
| `llm_client.py` | LLM provider client (Anthropic, OpenAI, Qwen, GigaChat, Ollama) — retry, circuit-breaker, cache |
| `model_loader.py` | Safe `.joblib` loading with SHA-256 trust-check + `TrustStore` |
| `artifact_contracts.py` | TypedDict schemas + `validate_bundle_schema()` |
| `workflow_contracts.py` | Pydantic pre/post-condition contracts for train/apply/cluster |
| `experiment_log.py` | Append-only JSONL experiment history (`~/.classification_tool/experiments.jsonl`) |
| `entity_normalizer.py` | Regex rules: Russian bank names → `[БАНК]`, products → `[ПРОДУКТ]` |
| `excel_utils.py` | Streaming XLSX/CSV reader (`open_tabular`), header detection, row counting |

---

## Adding a New Vectorizer

Implement the sklearn transformer interface and integrate via `make_hybrid_vectorizer()`:

```python
# ml_vectorizers.py — template based on MetaFeatureExtractor
class MyVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X: List[str], y=None) -> "MyVectorizer":
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        return np.array([self._featurize(text) for text in X])

    def _featurize(self, text: str) -> List[float]:
        ...
```

Then add it to the `FeatureUnion` in `make_hybrid_vectorizer()` (`ml_vectorizers.py:1863`).

### Adding a New ML Config Flag

When adding a new training-time flag (e.g. a new augmentation, calibration, or
deduplication knob), follow this three-step propagation:

1. **`ml_training.py`** — add the field to the `TrainingOptions` dataclass with
   a sensible default and add it to the `TRAINING_OPTION_KEYS` set used by the
   legacy-kwargs migration shim.
2. **`config/ml_constants.py`** — pull the default into a named constant if it
   should be tunable in one place (e.g. `DEFAULT_FUZZY_DEDUP_THRESHOLD = 92`).
3. **`app_train.py`** — propagate via `snap.get(...)` in **both** call-sites:
   the main one (≈line 3754) and the ensemble/K-fold one (≈line 3422). Both
   must stay symmetric — see Wave 5 Commit 1 for the rationale.

If the flag enters the workflow contract layer (snapshot validation), also add
a corresponding `_Field(...)` constraint to `_TrainSchema` in
`workflow_contracts.py` and mirror it in `_manual_validate_payload`.

---

## Model Bundle Format

Saved with `joblib.dump(bundle, path, compress=3)`. Key fields:

```python
bundle = {
    "artifact_type":        "train_model_bundle",   # validated on load
    "schema_version":       1,
    "pipeline":             sklearn.Pipeline,        # features → clf
    "config":               dict,                    # UI snapshot used for training
    "per_class_thresholds": dict[str, float],        # PR-curve thresholds
    "eval_metrics": {
        "macro_f1":    float,
        "accuracy":    float,
        "n_train":     int,
        "n_test":      int,
        "per_class_f1": dict[str, float],
        "trained_at":  str,                          # ISO 8601
    },
    "class_examples":       dict[str, list[str]],   # for LLM re-ranker
}
```

---

## Key Directories

```
.
├── app.py, app_train.py, app_apply.py, app_cluster.py  # UI mixins
├── ml_*.py                  # ML modules (no tkinter)
├── tests/                   # pytest test suite
│   ├── test_e2e_*.py        # E2E integration tests
│   ├── test_all_compat.py   # pytest wrapper for test_all.py
│   └── test_*.py            # unit / contract tests
├── config/                  # user_config.py (defaults, SBERT model names)
├── tools/                   # perf_smoke.py, CI utilities
├── .github/workflows/       # CI: quality-gates, nightly-perf, weekly-scan
├── requirements.txt         # core deps with version ranges
├── pyproject.toml           # ruff / mypy / pytest config
└── bank_reason_trainer.spec # PyInstaller build spec
```

---

## Building a Distributable

```bash
pip install pyinstaller
pyinstaller bank_reason_trainer.spec --clean
# Output: dist/BankReasonTrainer/
```

---

## CI Pipelines

| Workflow | Trigger | What runs |
|----------|---------|-----------|
| `quality-gates.yml` | PR + push to main | ruff, mypy, bandit, pytest (contracts + E2E) |
| `nightly-perf.yml` | 3 AM daily | hashing perf gates, cluster smoke |
| `weekly-quality-scan.yml` | Weekly | extended quality checks |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F5` | Run current tab operation (Train / Apply / Cluster) |
| `Ctrl+Return` | Same as F5 |
| `Ctrl+1` | Switch to Train tab |
| `Ctrl+2` | Switch to Apply tab |
| `Ctrl+3` | Switch to Cluster tab |
| `Ctrl++` / `Ctrl+-` | Increase / decrease font size |

---

## Security Setup

### LLM API Key Encryption

API keys stored in session snapshots are encrypted with Fernet (symmetric AES-128).
The encryption key is read from the `LLM_SNAPSHOT_KEY` environment variable.

**Generating a new Fernet key:**
```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()
print(key.decode())   # copy this value
```

Set the key before launching the app:
```bash
# Linux / macOS
export LLM_SNAPSHOT_KEY="<paste-key-here>"
python app.py

# Windows (PowerShell)
$env:LLM_SNAPSHOT_KEY = "<paste-key-here>"
python app.py
```

**Recommended**: store the key in your OS keychain rather than in a shell profile:
- **macOS**: `security add-generic-password -s BankReasonTrainer -a LLM_SNAPSHOT_KEY -w <key>`
- **Linux**: use `secret-tool` (libsecret) or `pass`
- **Windows**: use `Windows Credential Manager` via the `keyring` Python package

### Model Trust Store

`.joblib` model files are only loaded after SHA-256 hash verification (`model_loader.TrustStore`).
Trust entries are cached in memory for the session; they are re-verified on the next launch.

---

## Architecture Notes

- **Session state**: `app.py._snap_params()` serializes all `tk.Var` → dict. `_restore_session()` / `_save_session()` persist to `~/.classification_tool/last_session.json`.
- **Security**: `.joblib` files are only loaded after SHA-256 trust-check (`model_loader.TrustStore`). API keys are encrypted in runtime snapshots (`llm_key_store.py`).
- **Cluster pipeline stages**: `run_cluster()` in `app_cluster.py` is organized in 4 explicit stages (СТАДИЯ 1–4 banners). `ClusterRunState` dataclass captures all cross-stage variables for future extraction into separate methods.
- **Calibration metrics**: After temperature scaling, ECE/MCE are computed and logged. `training_duration_sec` and `model_size_bytes` are stored in each experiment record (`~/.classification_tool/experiments.jsonl`).
- **Graceful degradation**: App works without SBERT, torch, LLM, pymorphy — each optional dep is guarded with `try/except ImportError`.
