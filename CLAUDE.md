# CLAUDE.md — Developer Guide

BankReasonTrainer is a web-UI (Voilà + ipywidgets) application for
classifying Russian bank dialog texts into reason-of-contact categories.
It supports training (TF-IDF/SBERT/SetFit), batch prediction, and
unsupervised clustering with optional LLM-powered naming.

Desktop Tk/CTk UI was removed in Sprint 3 of the web migration
(see `docs/WEB_MIGRATION_PLAN.md`). The service layer and ML core are
unchanged — only the presentation layer switched to ipywidgets.

---

## Quick Start

```bash
# Launch web UI — cross-platform (Windows / Linux / macOS)
./run_web.sh                    # or: run_web.bat on Windows

# Headless CLI
python -m bank_reason_trainer {train,apply,cluster} --help
```

Running on a remote server or JupyterHub? See
[`docs/DEPLOY.md`](docs/DEPLOY.md) for headless CLI, Docker, and
notebook recipes. For the browser-based Voilà dashboard (three-panel
web UI backed by the service layer) see
[`docs/JUPYTERHUB_UI.md`](docs/JUPYTERHUB_UI.md).

### Reproducible install (CI / Docker / contributors)

`uv.lock` is the source of truth for fully-pinned wheels (ADR-0008).

```bash
# Install pinned set (matches CI + Docker exactly)
uv sync --frozen --extra dev

# Add the web-UI stack (ipywidgets + voila)
uv sync --frozen --extra ui

# Bumping a dep
$EDITOR pyproject.toml          # edit [project.dependencies]
uv lock                          # regenerate uv.lock
git add pyproject.toml uv.lock   # commit both — CI's `uv lock --check` enforces

# Build the dev image
docker build -t bank-reason-trainer:dev .
docker run --rm bank-reason-trainer:dev    # runs pytest
```

### Headless CLI

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
# SBERT + KMeans, or Combo (TF-IDF→SVD→L2 + SBERT→L2 hstack) + KMeans,
# or Ensemble (TF-IDF + 2×SBERT with silhouette-selected winner) + KMeans
python -m bank_reason_trainer cluster --files a.xlsx b.xlsx \
    --out clusters.csv --text-col text --k-clusters 8
# Pick algo via snap: {"cluster_vec_mode": "tfidf", "cluster_algo": "agglo"}
# or:                 {"cluster_vec_mode": "sbert", "cluster_algo": "kmeans",
#                      "sbert_model": "cointegrated/rubert-tiny2"}
# or (combo):         {"cluster_vec_mode": "combo", "cluster_algo": "kmeans",
#                      "combo_svd_dim": 200, "combo_alpha": 0.5,
#                      "sbert_model": "cointegrated/rubert-tiny2"}
# or (ensemble):      {"cluster_vec_mode": "ensemble", "cluster_algo": "kmeans",
#                      "sbert_model": "cointegrated/rubert-tiny2",
#                      "sbert_model2": "sberbank-ai/sbert_large_nlu_ru"}

# Cluster: other combos (setfit / BERTopic / fastopic) still need
# --allow-skeleton; only prepare_inputs runs end-to-end.
python -m bank_reason_trainer cluster --files a.xlsx b.xlsx \
    --snap bertopic_snap.json --allow-skeleton
```

`train` and `apply` are real: TF-IDF features, joblib bundle
round-trip, CSV/XLSX I/O. `cluster` is real end-to-end for
`tfidf` + {`kmeans`, `agglo`, `lda`, `hdbscan`}, `sbert` + `kmeans`,
`combo` + `kmeans`, and `ensemble` + `kmeans`. They stream text from
`--files`, fit the chosen vectorizer + clusterer, and write
`text,cluster_id,top_keywords` to `--out`. Agglomerative is capped at
5 000 rows (Ward linkage is O(n²) in memory); HDBSCAN discovers K itself
(the `--k-clusters` flag is ignored); SBERT downloads the model from
HuggingFace Hub on first run (cached in `SBERT_LOCAL_DIR`); combo mode
blends TF-IDF-SVD and SBERT vectors via `combo_alpha` (0..1); ensemble
mode fits KMeans on TF-IDF + 2×SBERT candidates and keeps the
silhouette-winner (`sbert_model2` defaults to `sbert_model`). Other
combos (BERTopic, SetFit, FASTopic) still raise `NotImplementedError`
and require `--allow-skeleton` for the prepare-only fallback — see
`docs/adr/0002-pipeline-stages-and-snapshots.md` and
`docs/adr/0007-wave5-quality-polish.md`.

---

## Running Tests

```bash
# All tests via pytest
PYTHONPATH=. pytest -q

# Specific suites
PYTHONPATH=. pytest tests/test_e2e_train_predict.py -v    # E2E train → predict
PYTHONPATH=. pytest tests/test_e2e_excel.py -v            # Excel round-trip
PYTHONPATH=. pytest tests/test_workflow_contracts.py -v   # Pydantic contracts

# Voilà HTTP smoke (slow — starts a real server)
RUN_VOILA_SMOKE=1 PYTHONPATH=. pytest tests/test_voila_smoke.py -v
```

Requirements for E2E tests: `scikit-learn numpy scipy joblib openpyxl` (core deps, always installed).

---

## Module Map

### Web UI (ipywidgets, Voilà)
| File | Role |
|------|------|
| `notebooks/ui.ipynb` | Voilà entry-point — renders `ui_widgets.build_app()` |
| `ui_widgets/notebook_app.py` | `build_app()` — three-panel VBox (Train / Apply / Cluster) |
| `ui_widgets/train_panel.py` | Training panel — file upload, config form, progress, download |
| `ui_widgets/apply_panel.py` | Apply panel — model picker, prediction table, XLSX export |
| `ui_widgets/cluster_panel.py` | Cluster panel — multi-file input, algo snap, cluster table |
| `ui_widgets/session.py` | `save_session()` + `DebouncedSaver` — persists snap to `~/.classification_tool/last_session.json` |
| `ui_widgets/theme.py` | CSS injection + helpers (`section_card`, `chip`, `badge`, `metric_card`) |
| `ui_widgets/dialogs/` | Overlay cards: history, artifacts, settings, trust-prompt |

### Service Layer (no Tk, testable in isolation)
| File | Role |
|------|------|
| `app_train_service.py` | `TrainingWorkflow` — thin wrapper over `train_model()` |
| `apply_prediction_service.py` | `predict_with_thresholds()`, `validate_apply_bundle()` |
| `cluster_workflow_service.py` | `ClusteringWorkflow.run()` — orchestrates 4 pipeline stages |
| `app_cluster_pipeline.py` | Pure pipeline functions: `prepare_inputs`, `build_vectors`, `run_clustering`, `postprocess_clusters`, `export_cluster_outputs` |
| `app_train_workflow.py`, `app_apply_workflow.py`, `app_cluster_workflow.py` | Precondition validators + snapshot builders (used by UI + CLI) |

### CLI
| File | Role |
|------|------|
| `bank_reason_trainer/__main__.py` | `python -m bank_reason_trainer` shim |
| `bank_reason_trainer/cli.py` | argparse router for `train`, `apply`, `cluster` subcommands |

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
deduplication knob), follow this two-step propagation:

1. **`ml_training.py`** — add the field to the `TrainingOptions` dataclass with
   a sensible default and add it to the `TRAINING_OPTION_KEYS` set used by the
   legacy-kwargs migration shim.
2. **`config/ml_constants.py`** — pull the default into a named constant if it
   should be tunable in one place (e.g. `DEFAULT_FUZZY_DEDUP_THRESHOLD = 92`).

The CLI (`bank_reason_trainer/cli.py`) forwards unknown `--snap` keys into
`TrainingOptions` automatically, so no extra wiring is required. Same for
the web UI: `ui_widgets/train_panel.py` builds its snapshot via
`app_train_workflow.build_validated_train_snapshot()`, which passes through
any keys recognised by `TrainingOptions`.

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
    "config":               dict,                    # snapshot used for training
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
├── ui_widgets/             # Web UI (ipywidgets + CSS)
│   ├── notebook_app.py     # entry-point: build_app()
│   ├── train_panel.py, apply_panel.py, cluster_panel.py
│   ├── session.py          # save_session + DebouncedSaver
│   ├── theme.py            # CSS + helpers (section_card, chip, badge, …)
│   └── dialogs/            # overlay cards (history, artifacts, settings)
├── notebooks/ui.ipynb      # Voilà entry
├── bank_reason_trainer/    # headless CLI (`python -m bank_reason_trainer …`)
├── ml_*.py                 # ML modules (no tk, no customtkinter)
├── app_*_service.py        # orchestration services (train/apply/cluster)
├── app_cluster_pipeline.py # pure pipeline functions
├── tests/                  # pytest test suite
│   ├── test_e2e_*.py       # E2E integration tests
│   ├── test_voila_smoke.py # Voilà HTTP smoke (opt-in RUN_VOILA_SMOKE=1)
│   └── test_*.py           # unit / contract tests
├── config/                 # user_config.py (defaults, SBERT model names)
├── tools/                  # perf_smoke.py, CI utilities
├── .github/workflows/      # CI: quality-gates, nightly-perf, weekly-scan
├── pyproject.toml          # deps + ruff / mypy / pytest config
├── uv.lock                 # fully-pinned wheels (ADR-0008)
└── run_web.sh, run_web.bat # launcher scripts
```

---

## CI Pipelines

| Workflow | Trigger | What runs |
|----------|---------|-----------|
| `quality-gates.yml` | PR + push to main | ruff, mypy, bandit, pytest (contracts + E2E), Voilà web smoke |
| `nightly-perf.yml` | 3 AM daily | hashing perf gates, cluster smoke |
| `weekly-quality-scan.yml` | Weekly | extended quality checks |

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
./run_web.sh

# Windows (PowerShell)
$env:LLM_SNAPSHOT_KEY = "<paste-key-here>"
run_web.bat
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

- **Session state**: `ui_widgets/session.py:save_session()` persists a
  JSON-safe snap to `~/.classification_tool/last_session.json` via
  atomic rename. `DebouncedSaver` coalesces frequent widget-change
  callbacks into one write.
- **Security**: `.joblib` files are only loaded after SHA-256 trust-check (`model_loader.TrustStore`). API keys are encrypted in runtime snapshots (`llm_key_store.py`).
- **Cluster pipeline stages**: `ClusteringWorkflow.run()` in
  `cluster_workflow_service.py` orchestrates four headless stages
  defined in `app_cluster_pipeline.py`: `prepare_inputs`,
  `build_vectors`, `run_clustering`, `postprocess_clusters`,
  `export_cluster_outputs`. Both the web UI and the CLI call the
  same workflow — there is no desktop-only variant.
- **Calibration metrics**: After temperature scaling, ECE/MCE are computed and logged. `training_duration_sec` and `model_size_bytes` are stored in each experiment record (`~/.classification_tool/experiments.jsonl`).
- **Graceful degradation**: App works without SBERT, torch, LLM, pymorphy — each optional dep is guarded with `try/except ImportError`.
