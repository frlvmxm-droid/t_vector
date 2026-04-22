# JupyterHub Web UI (Voilà dashboard)

This guide describes how to expose the three main BankReasonTrainer
workflows — **Обучение / Применение / Кластеризация** — as a
browser-based web UI on a JupyterHub deployment. (The desktop Tk/CTk
app was removed in Sprint 3 of the web migration; the Voilà dashboard
is now the only UI.)

The UI is built from the headless service layer
(`app_train_service`, `apply_prediction_service`,
`cluster_workflow_service`) via `ipywidgets` and rendered by
[Voilà](https://voila.readthedocs.io/). No Tkinter, Node.js, or
separate web-server is required.

See also: [`docs/DEPLOY.md`](DEPLOY.md) for the plain-notebook
workflow and non-UI deployment scenarios.

---

## Architecture at a glance

```
notebooks/ui.ipynb  (2 cells: sys.path bootstrap + display(build_app()))
      │
      ▼
ui_widgets.build_app()           # 3-tab VBox
      │
      ├── build_train_panel()    → TrainingWorkflow.fit_and_evaluate
      ├── build_apply_panel()    → predict_with_thresholds
      └── build_cluster_panel()  → ClusteringWorkflow.run
                                    │
                                    ▼
                           ml_training / ml_vectorizers / …
```

Long-running operations run in a `threading.Thread`. Widget `.value`
writes are pushed to the browser over the Jupyter COMM protocol, so
the progress bar and log pane update live without extra plumbing.

Downloads of trained models / prediction tables / cluster CSVs are
served as base64 `data:` URLs embedded in an `<a download="…">` tag —
no server-side endpoint is required.

---

## Operator setup

### 1. Install the `ui` extra

On each JupyterHub user image (or per-user env):

```bash
uv sync --frozen --extra ml --extra ui
# or, without uv:
pip install "ipywidgets>=8.0" "voila>=0.5"
```

### 2. Point the spawner at the dashboard

In `jupyterhub_config.py`, make every user session land directly on
the Voilà-rendered dashboard:

```python
c.Spawner.cmd = ["jupyter-labhub"]
c.Spawner.args = [
    "--ServerApp.default_url=/voila/render/notebooks/ui.ipynb",
]
```

Users still have the full Jupyter UI available at
`https://hub/…/lab` — they just start on the dashboard.

### 3. (Optional) Pre-seed the HuggingFace cache

SBERT / transformer models are downloaded on first use. On multi-user
hubs, set a shared cache to avoid re-downloading per user:

```python
c.Spawner.environment = {"HF_HOME": "/shared/hf-cache"}
```

### 4. (Optional) Pre-seed data paths

The UI accepts two input modes: browser upload (<50 MB) or a
server-side path on a shared volume. Mount a data volume and tell
users the absolute path to paste into the "Путь:" field.

---

## Smoke test on a local machine

Before rolling out to the hub, verify locally:

```bash
# 1. Deps
uv sync --frozen --extra ml --extra ui

# 2. Syntax-check the widget modules (no ipywidgets needed)
python -c "import ast; [ast.parse(open(f'ui_widgets/{m}.py').read()) \
    for m in ['io','progress','train_panel','apply_panel','cluster_panel','notebook_app']]"

# 3. Import smoke (ipywidgets required)
python -c "from ui_widgets import build_app; print(type(build_app()))"
#   → <class 'ipywidgets.widgets.widget_box.VBox'>

# 4. Voilà
voila notebooks/ui.ipynb --port 8866 --no-browser
# open http://localhost:8866/ in a browser
```

---

## User guide

After the spawner redirects you, the page shows a header and three
tabs. All long operations report live progress into a log pane and
unlock the action button again when they finish.

### 📚 Обучение (Train)

1. Upload an XLSX/CSV via *«Загрузить XLSX/CSV»*, **or** paste an
   absolute server path (for files >50 MB).
2. Set the text / label column names (defaults: `text`, `label`).
3. Tune hyperparameters — `C`, `max_iter`, `test_size`,
   `class_weight=balanced`, SMOTE, calibration, fuzzy-dedup.
4. Click **Обучить**. The worker thread streams progress; on success
   the panel shows `macro-F1`, `accuracy`, and a **📥 Скачать
   model.joblib** link.

The saved `.joblib` follows the standard bundle format (see
`CLAUDE.md` → "Model Bundle Format") and is directly consumable by
the Apply tab and by `python -m bank_reason_trainer apply`.

### 🎯 Применение (Apply)

1. Upload a `.joblib` model (or paste a server path).
2. Upload an XLSX/CSV with the texts to score.
3. Choose the text column and the threshold mode
   (`review_only` / `strict`).
4. Click **Применить**. The panel reports `macro-F1`, classes, and
   the `needs_review` share, and produces a downloadable
   `predictions.xlsx` / `predictions.csv`.

### 🧩 Кластеризация (Cluster)

1. Upload one or more XLSX/CSV files (or paste multi-line server
   paths, one per line).
2. Pick a vectorizer + algorithm:
   - **TF-IDF** + KMeans / Agglomerative / LDA / HDBSCAN
   - **SBERT** + KMeans (edit `SBERT model` as needed)
   - **Combo** (TF-IDF→SVD→L2 ⊕ SBERT→L2) + KMeans — tune
     `combo α` and `SVD dim`
   - **Ensemble** (TF-IDF + 2×SBERT, silhouette winner) + KMeans
3. Set `K` (ignored for HDBSCAN — it discovers K itself).
4. Click **Кластеризовать**. A **clusters.csv** with
   `text,cluster_id,top_keywords` is produced.

Unsupported combos (BERTopic / SetFit / FASTopic / hierarchical /
GMM) are surfaced as a warning banner in the panel. They fall through
to the prepare-only skeleton path if you force-run via the CLI with
`--allow-skeleton`; full end-to-end for these combos will need to be
re-implemented in the service layer.

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| Blank page at `/voila/render/…` | Voilà is not installed in the kernel. Install `[ui]` extra: `uv sync --extra ui`. |
| `ModuleNotFoundError: ui_widgets` | The notebook's first cell must add the repo root to `sys.path`. `notebooks/ui.ipynb` already does this — if you moved it, update the path. |
| Upload hangs at ~100 MB | Browser limit. Use the "Путь:" field with a shared-volume path instead. |
| Progress bar doesn't move | The worker thread may be blocked on a HuggingFace download. Check the log pane or kernel stderr. |
| `ModuleNotFoundError: app` / `app_train` / `bootstrap_run` | Desktop modules were removed in Sprint 3 of the web migration. The UI uses only `*_service.py` / `app_cluster_pipeline`; update any stale notebook cells that still reference them. |
| `ModelLoadError: [UNTRUSTED_MODEL_PATH]` | The .joblib path is not in `~/.classification_tool/trusted_models.json`. The Apply panel calls `load_model_artifact` with defaults — either register the hash or load with `require_trusted=False` in a notebook cell. |
| Cluster panel rejects combo | Only `tfidf+{kmeans,agglo,lda,hdbscan}`, `sbert+kmeans`, `combo+kmeans`, `ensemble+kmeans` are supported. For BERTopic / SetFit / FASTopic use the CLI with `--allow-skeleton` (prepare-only path). |
| Two workers launched in one session | The action button is disabled while a worker is alive. If the kernel was restarted mid-run, reload the page to reset widget state. |

---

## Security notes

- `.joblib` bundles are loaded through `model_loader.load_model_artifact`,
  which validates the schema and (optionally) SHA-256 against the
  per-user trust store at `~/.classification_tool/trusted_models.json`.
  On a multi-user hub, make sure that directory is on a persistent
  per-user volume.
- The dashboard shares the kernel's filesystem permissions. Users can
  only see paths their JupyterHub account can read.
- API keys (LLM providers, `LLM_SNAPSHOT_KEY`) are **not** used by
  the Voilà UI — none of the three tabs call into LLMs or the
  encrypted session store.

---

## Advanced features (Phases 8–12)

| Tab | Feature | How to use |
|---|---|---|
| Train | Auto-profile (Smart / Strict / Manual) | Dropdown swaps the underlying defaults for SMOTE / calibration / oversampling. |
| Train | SBERT device override | Dropdown picks `auto` / `cpu` / `cuda` / `mps`. |
| Train | Advanced Accordion | Optuna, K-fold, Cleanlab, label smoothing, hard-negatives, field-dropout. Default collapsed. |
| Apply | Header chips | Trust-check, macro-F1, train/test rows, classes, file format, encoding — populated after `Inspect` on the bundle. |
| Apply | Per-class thresholds | Editable sliders, seeded from `bundle['per_class_thresholds']`. |
| Apply | Ensemble | Upload a second `.joblib` + weight slider; predict averages `predict_proba`. |
| Apply | LLM-rerank | Switch + top-K slider; opt-in, requires `BRT_LLM_API_KEY` (or `BRT_LLM_PROVIDER=offline`). |
| Cluster | Auto-K | `cluster_auto_k=True` runs `auto_k_service.select_k` (silhouette / calinski / elbow) before KMeans. |
| Cluster | LLM-naming | `use_llm_naming=True` uses `cluster_naming_service.name_clusters_with_llm` after postprocess. |
| Cluster | T5 summary | `use_t5_summary=True` uses `cluster_summarization_service.summarize_clusters_with_t5`; gracefully skipped without `transformers`. |
| Cluster | UMAP / HDBSCAN / LDA knobs | All exposed in the panel; conditional cards show only when the algo / vec_mode is selected. |
| Sidebar | КОНТЕКСТ → 🕘 История | Shows the last 20 records from `~/.classification_tool/experiments.jsonl`. |
| Sidebar | КОНТЕКСТ → 📦 Артефакты | Recursive `.joblib` scan of `~/.classification_tool/`, with file size + SHA-256 prefix. |
| Sidebar | КОНТЕКСТ → ⚙️ Настройки | Read-only dependency check + LLM environment variable status. |

The Auto-K / LLM-naming / T5 services are pure-Python modules
(`auto_k_service.py`, `cluster_naming_service.py`,
`cluster_summarization_service.py`) — they live next to the headless
service layer and have unit tests in `tests/test_*_service.py`. They
are the canonical implementation used by the web UI and the CLI.

Context dialogs (Phase 12) use an `ipywidgets.Stack` overlay: the three
workflow panels live at slots 0..2, the dialogs at slots 3..5. Clicking a
context button swaps the `selected_index`; clicking the ✕ button returns
to the last-active workflow panel. All dialogs are read-only; editing
LLM keys or deleting artifacts is out of scope for the web UI — use
shell tools (`~/.classification_tool/` is a regular directory tree).

---

## Known limitations

- No cancel button — the action button stays disabled until the
  worker finishes. Cancelation requires a `cancel_event` hook in the
  service layer (planned follow-up).
- No session save/restore (snap.json round-trip).
- LLM-naming requires either `BRT_LLM_PROVIDER=offline` (deterministic
  CI stub from ADR-0004) or a real provider API key in the kernel
  environment. The dialog's Settings tab indicates which keys are set.
- T5 summarization needs ~1 GB of model weights on first call. Pre-seed
  via `HF_HOME` or it will download on the first cluster run.
- Plotly / Cleanlab visualisations not yet wired into the web panels
  (can be added to the cluster panel via `plotly.FigureWidget`).
- No GPU contention guard on multi-user hubs — rely on JupyterHub
  resource limits.

---

## Self-hosted via docker-compose

Three reference files ship in the repo root:

- [`Dockerfile.jupyterhub`](../Dockerfile.jupyterhub) — builds the hub
  image on top of `jupyterhub/jupyterhub:4.1.5` with the project's
  `uv.lock`, the `[ml]` + `[ui]` extras, and deploy-time packages
  (`dummyauthenticator`, `psycopg2-binary`, `jupyterlab`, `notebook`).
- [`jupyterhub_config.py.example`](../jupyterhub_config.py.example) —
  minimal single-admin `jupyterhub_config.py` with `DummyAuthenticator`,
  `LocalProcessSpawner`, Voilà default URL, PostgreSQL-aware
  `db_url`, and a CONFIGPROXY_AUTH_TOKEN guard.
- [`docker-compose.yml.example`](../docker-compose.yml.example) —
  two services (`jupyterhub`, `postgres:16-alpine`) with named
  volumes for hub state, HF cache, persistent user homes, and
  pg-data.

### Quick start

```bash
cp docker-compose.yml.example docker-compose.yml
cp jupyterhub_config.py.example jupyterhub_config.py
#   edit HUB_DUMMY_PASSWORD inside jupyterhub_config.py

export CONFIGPROXY_AUTH_TOKEN="$(openssl rand -hex 32)"
export POSTGRES_PASSWORD="$(openssl rand -hex 16)"

docker compose up --build
#   → http://localhost:8000
#   login: admin / <HUB_DUMMY_PASSWORD>
```

First login lands directly on `/voila/render/notebooks/ui.ipynb`;
the classic Lab UI is still reachable at `/user/<name>/lab`.

### What to change before going to production

| Area | Reference default | Production swap |
|---|---|---|
| Authenticator | `DummyAuthenticator` (one shared password) | LDAP / OAuth / PAM / `NativeAuthenticator` |
| Spawner | `LocalProcessSpawner` (pre-created demo users) | `DockerSpawner` or `KubeSpawner` with a separate single-user image |
| TLS | Plain HTTP on port 8000 | Reverse proxy (Caddy / Traefik / nginx) terminating HTTPS |
| Secrets | `POSTGRES_PASSWORD`, `CONFIGPROXY_AUTH_TOKEN` in env | Docker secrets or an external vault; never commit `.env` |
| Postgres image | `postgres:16-alpine` (floating tag) | Pin to a specific digest (`@sha256:...`), configure backups |
| SBERT / T5 cache | Shared `/shared/hf-cache` volume (bytes-mutable by all spawners) | Read-only mount filled by a privileged job; users can't poison it |
| LLM provider | `BRT_LLM_PROVIDER=offline` (deterministic stub, ADR-0004) | Real provider + `LLM_SNAPSHOT_KEY` (Fernet) + API key injected via secret |

### Volumes created by the compose stack

| Volume | Purpose | What to back up |
|---|---|---|
| `pg-data` | JupyterHub DB (users, tokens, server state) | yes — losing it invalidates user sessions and admin data |
| `hub-state` | cookie secret, sqlite fallback if `DATABASE_URL` is unset, admin logs | yes |
| `hf-cache` | shared HuggingFace cache (SBERT, T5, transformers) | no — re-downloadable; regenerating is slow but deterministic |
| `user-home` | per-user `~/.classification_tool/` (experiment log, trust store) | yes — losing it wipes trust entries and user history |

---

See also: [`CLAUDE.md`](../CLAUDE.md),
[`docs/DEPLOY.md`](DEPLOY.md),
[`docs/BUNDLE_LIFECYCLE.md`](BUNDLE_LIFECYCLE.md).
