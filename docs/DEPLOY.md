# Deploying BankReasonTrainer on a Remote Server / JupyterHub

This guide covers three ways to run the project without the desktop
Tkinter UI: **Python CLI** on a Linux server, **Docker**, and
**JupyterHub / Jupyter notebooks** (the primary focus).

The service layer (`app_train_service.py`, `apply_prediction_service.py`,
`cluster_workflow_service.py`) contains zero `tkinter` imports and is
safe to use from batch scripts, notebooks, and Docker.

---

## TL;DR

| Scenario | When to use | Entry point |
|---|---|---|
| Python CLI on Linux server | batch pipelines, cron, SSH | `python -m bank_reason_trainer <cmd>` |
| Docker | reproducible infra, shared CI | `docker build + docker run` |
| JupyterHub / notebook | interactive analysis, experiments | `from cluster_workflow_service import ClusteringWorkflow` |
| Voilà dashboard on JupyterHub | browser web UI (3 tabs, no Tk) | `voila notebooks/ui.ipynb` → see [`JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md) |

---

## Prerequisites (all scenarios)

- Python 3.11–3.13 (`requires-python = ">=3.11,<3.14"`)
- ≥4 GB RAM for TF-IDF; ≥8 GB + GPU for SBERT / SetFit / DeBERTa
- Linux (Debian bookworm confirmed via Dockerfile); macOS works; Windows
  is desktop-GUI only

Optional environment variables:

| Var | Purpose | Needed for |
|---|---|---|
| `PYTHONPATH=.` | add repo root to `sys.path` when launching from repo | CLI, Jupyter |
| `HF_HOME` | HuggingFace cache location (defaults to `~/.cache/huggingface`) | SBERT / transformers |
| `BRT_LLM_PROVIDER=offline` | deterministic stub LLM (CI / air-gapped envs) | optional |
| `LLM_SNAPSHOT_KEY` | Fernet key for encrypted API keys | **UI only**, skip for CLI/Jupyter |

---

## Scenario A — Python CLI on a Linux server

```bash
# 1. Clone and install (pinned wheels from uv.lock)
git clone <repo-url> t_vector && cd t_vector
python -m pip install uv
uv sync --frozen              # core deps (TF-IDF path)
uv sync --frozen --extra ml   # optional: SBERT / SetFit / torch

# 2. Train a model
PYTHONPATH=. python -m bank_reason_trainer train \
    --data train.xlsx --out model.joblib \
    --text-col text --label-col label
    # optional: --snap snap.json (TrainingOptions overrides)

# 3. Apply the model
PYTHONPATH=. python -m bank_reason_trainer apply \
    --model model.joblib --data in.xlsx --out out.xlsx \
    --text-col text --threshold 0.5

# 4. Cluster (supported combos: tfidf+{kmeans,agglo,lda,hdbscan},
#    sbert+kmeans, combo+kmeans, ensemble+kmeans)
PYTHONPATH=. python -m bank_reason_trainer cluster \
    --files a.xlsx b.xlsx --out clusters.csv \
    --text-col text --k-clusters 8
    # other combos (bertopic / setfit / fastopic) need --allow-skeleton
```

Notes:
- The `--snap` flag accepts a JSON file with the same shape the desktop
  UI serialises via `app._snap_params()`.
- Artifacts are written under `~/.classification_tool/`
  (`experiments.jsonl`, `llm_rerank_cache/`). The process needs write
  access to `$HOME`.
- Cluster XLSX files are written to `<repo_root>/clustering/` (see
  `config/paths.py::CLUST_DIR`); on a read-only repo, symlink that
  directory to a writable path.

---

## Scenario B — Docker

```bash
docker build -t bank-reason-trainer:dev .

# Default CMD runs the headless test suite.
docker run --rm bank-reason-trainer:dev

# Run the CLI against mounted data:
docker run --rm \
    -v "$PWD/data:/app/data" \
    -v "$PWD/models:/app/models" \
    bank-reason-trainer:dev \
    python -m bank_reason_trainer train \
        --data /app/data/train.xlsx \
        --out  /app/models/model.joblib \
        --text-col text --label-col label
```

The image pins `python:3.11.11-slim-bookworm` and installs wheels from
`uv.lock` via `uv sync --frozen` (see ADR-0008). The `python3-tk`
package is present for headless smoke tests — it does **not** enable
the GUI (no display).

---

## Scenario C — JupyterHub / Jupyter notebook

### Step 1 — Install the repo into your JupyterHub home

Open a terminal from the JupyterHub launcher:

```bash
cd ~
git clone <repo-url> t_vector
cd t_vector
python -m pip install --user uv
~/.local/bin/uv sync --frozen --extra ml   # adds SBERT / transformers / torch
```

### Step 2 — (Optional) Register an isolated kernel

```bash
~/.local/bin/uv run python -m ipykernel install \
    --user --name t_vector --display-name "t_vector (uv)"
```

A kernel named **"t_vector (uv)"** will appear in the Jupyter launcher.

### Step 3 — Notebook recipe

**Cell 1 — bootstrap**

```python
import sys, os, pathlib
REPO = pathlib.Path.home() / "t_vector"
sys.path.insert(0, str(REPO))            # so `app_train_service` resolves
os.environ.setdefault("HF_HOME", "/shared/hf-cache")  # optional shared cache
```

**Cell 2 — training via the service layer**

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from ml_training import TrainingOptions
from app_train_service import TrainingWorkflow

df = pd.read_excel("/data/train.xlsx")
texts  = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()

# Same default feature stack the CLI uses (word 1-2 + char 3-5 TF-IDF).
features = FeatureUnion([
    ("word", TfidfVectorizer(analyzer="word",    ngram_range=(1, 2),
                             min_df=2, max_df=0.95, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             min_df=2, max_df=0.95, sublinear_tf=True)),
])

workflow = TrainingWorkflow()
pipe, clf_type, report, classes, cm, extras = workflow.fit_and_evaluate(
    X=texts, y=labels, features=features,
    C=1.0, max_iter=2000, balanced=False,
    test_size=0.2, random_state=42,
    options=TrainingOptions(use_smote=True, calib_method="sigmoid"),
    log_cb=print,
)

bundle = {
    "artifact_type": "train_model_bundle",
    "schema_version": 1,
    "pipeline": pipe,
    "config": {"clf_type": clf_type, "text_col": "text", "label_col": "label"},
    "per_class_thresholds": dict(extras.get("per_class_thresholds", {})),
    "eval_metrics": {
        "macro_f1": float(extras.get("macro_f1", 0.0)),
        "accuracy": float(extras.get("accuracy", 0.0)),
        "n_train":  int(extras.get("n_train", 0)),
        "n_test":   int(extras.get("n_test", 0)),
    },
}
workflow.persist_artifact(bundle, "/models/model.joblib")
print("macro-F1:", bundle["eval_metrics"]["macro_f1"])
```

**Cell 3 — applying the model**

```python
import numpy as np
from model_loader import load_model_artifact
from apply_prediction_service import (
    predict_with_thresholds, validate_apply_bundle,
)

b = load_model_artifact("/models/model.joblib")
validate_apply_bundle(b)

texts_to_score = df["text"].astype(str).tolist()
proba = b["pipeline"].predict_proba(texts_to_score)

result = predict_with_thresholds(
    np.asarray(proba),
    classes=b["pipeline"].classes_,
    per_class_thresholds=b.get("per_class_thresholds"),
    default_threshold=0.5,
    threshold_mode="review_only",
)
pd.DataFrame({
    "text":       texts_to_score,
    "label":      result.labels,
    "confidence": result.confidences,
    "review":     result.needs_review,
}).head()
```

**Cell 4 — clustering**

```python
from cluster_workflow_service import ClusteringWorkflow

result = ClusteringWorkflow.run(
    files_snapshot=["/data/a.xlsx", "/data/b.xlsx"],
    snap={
        "cluster_vec_mode": "tfidf",     # or "sbert", "combo", "ensemble"
        "cluster_algo":     "kmeans",    # or "agglo", "lda", "hdbscan"
        "k_clusters":       8,
        "text_col":         "text",
        "output_path":      "/out/clusters.csv",
    },
    log_cb=print,
    progress_cb=lambda frac, msg: print(f"  [{frac:.0%}] {msg}"),
)
print(f"K={result.n_clusters}  noise={result.n_noise}")
```

### JupyterHub-specific caveats

- **Persistent home.** Make sure `~/t_vector` and `~/.classification_tool`
  live on a persistent volume; otherwise trust store / experiment log
  are lost on every spawn.
- **Shared HF cache.** Setting `HF_HOME` to a shared directory prevents
  each user from re-downloading multi-GB SBERT / transformers weights.
- **GPU.** If the spawner provides a GPU node, the `--extra ml` install
  pulls `torch>=2.1` with CUDA wheels automatically. Verify with
  `torch.cuda.is_available()` in a notebook cell.
- **Secrets.** Keep `LLM_SNAPSHOT_KEY` (if ever used) and LLM provider
  keys in per-user `~/.jupyter/env` or the JupyterHub secret provider —
  not in shared settings.
- **Supported combos only.** `ClusteringWorkflow.run` covers
  `tfidf+{kmeans,agglo,lda,hdbscan}`, `sbert+kmeans`, `combo+kmeans`,
  and `ensemble+kmeans`. Other combos (BERTopic / SetFit / FASTopic /
  hierarchical / GMM) still live inside the desktop-bound
  `app_cluster.run_cluster()` and are not callable from a notebook yet.
- **Desktop UI helpers not portable.** LLM cluster naming, T5
  summarisation, Plotly visualisation, UMAP, and auto-K selection live
  inside broken closure helpers (`_cluster_step_*`) in `app_cluster.py`
  and are not exposed by the service layer. Build your own LLM / plot
  helpers in the notebook if you need them.

---

## Scenario D — Voilà dashboard (browser-based web UI)

If your users want a browser UI instead of writing notebook cells,
the repo ships a three-tab Voilà dashboard
(`notebooks/ui.ipynb` + `ui_widgets/`) that wraps the same service
layer used above.

```bash
# On the JupyterHub user image
uv sync --frozen --extra ml --extra ui   # adds ipywidgets + voila

# In jupyterhub_config.py — land each user on the dashboard
c.Spawner.cmd  = ["jupyter-labhub"]
c.Spawner.args = ["--ServerApp.default_url=/voila/render/notebooks/ui.ipynb"]
```

The dashboard exposes **Обучение / Применение / Кластеризация** as
tabs with file upload, progress bar, live log, and one-click
download of the resulting `model.joblib` / `predictions.xlsx` /
`clusters.csv`.

Full operator setup, user guide, and troubleshooting live in
[`docs/JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md).

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `ModuleNotFoundError: app_train_service` | `sys.path` is missing the repo root — add `sys.path.insert(0, "<repo>")` (notebook) or `PYTHONPATH=.` (CLI). |
| `OSError: [Errno 30] Read-only file system: clustering/` | `CLUST_DIR` is inside a read-only repo. Symlink `clustering/` to a writable directory. |
| `RuntimeError: couldn't connect to 'https://huggingface.co'` | No internet for SBERT / T5. Pre-download the model with `huggingface-cli download <name>` into `$HF_HOME`. |
| `ImportError: cannot import name '_tkinter'` when importing `app.py` | **Do not import `app.py`** in a notebook. Use the `*_service.py` modules and `app_cluster_pipeline` directly. |
| `ModelLoadError: [UNTRUSTED_MODEL_PATH]` | `load_model_artifact` was called with `require_trusted=True` and the path is not in the per-user trust store (`~/.classification_tool/trusted_models.json`). Either drop the flag in notebook code or register the hash via `TrustStore`. |

---

## Security notes

- `.joblib` bundles are validated by SHA-256 via
  `model_loader.TrustStore` and `artifact_contracts.validate_bundle_schema`.
  On JupyterHub the trust store is per-user at
  `~/.classification_tool/trusted_models.json`.
- Never commit API keys. Use `os.environ` in notebooks and, if
  available, the JupyterHub secret provider (or the `keyring` package
  backed by the OS keychain).
- The offline LLM provider (`BRT_LLM_PROVIDER=offline`) returns
  deterministic stub responses — handy for reproducible experiments and
  air-gapped environments.

---

See also: [`CLAUDE.md`](../CLAUDE.md) (Quick Start),
[`docs/JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md) (browser dashboard),
[`docs/DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md),
[`docs/BUNDLE_LIFECYCLE.md`](BUNDLE_LIFECYCLE.md),
[`docs/adr/0008-pin-wheels-uv-lock.md`](adr/0008-pin-wheels-uv-lock.md).
