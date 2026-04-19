# -*- coding: utf-8 -*-
"""Чистые pipeline-утилиты кластеризации (без tkinter-зависимостей)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, List, Optional, Sequence, TypedDict

from text_utils import parse_dialog_roles


@dataclass(frozen=True)
class ClusterRoleContext:
    cluster_snap: "ClusterRoleSnapshot"
    role_label: str
    ignore_chatbot_label: str


@dataclass(frozen=True)
class PreparedInputs:
    files_snapshot: List[str]
    snap: "ClusterSnapshot"
    role_context: ClusterRoleContext


@dataclass(frozen=True)
class VectorPack:
    prepared: PreparedInputs
    vectors: Optional[object] = None
    meta: Optional[dict[str, object]] = None


@dataclass(frozen=True)
class ClusterResult:
    vectors: VectorPack
    labels: Optional[object] = None
    meta: Optional[dict[str, object]] = None


@dataclass(frozen=True)
class PostprocessResult:
    result: ClusterResult
    payload: Optional["PostprocessPayload"] = None


@dataclass(frozen=True)
class ExportSummary:
    postprocessed: PostprocessResult
    outputs: Optional["ExportOutputs"] = None


class ClusterSnapshot(TypedDict, total=False):
    cluster_algo: str
    cluster_vec_mode: str
    call_col: str
    chat_col: str
    use_summary: bool
    ignore_chatbot_cluster: bool


class ClusterRoleSnapshot(ClusterSnapshot, total=False):
    desc_col: str
    summary_col: str
    ans_short_col: str
    ans_full_col: str
    ignore_chatbot: bool
    base_w: dict[str, int]
    auto_profile: str


class StageMeta(TypedDict, total=False):
    snap: ClusterSnapshot
    prepared: PreparedInputs


class PostprocessPayload(TypedDict, total=False):
    prepared: PreparedInputs
    snap: ClusterSnapshot
    # Wave 3a slice (TF-IDF + KMeans only):
    cluster_sizes: dict[int, int]
    cluster_keywords: dict[int, List[str]]
    texts: List[str]
    labels: object  # numpy ndarray; kept as object to avoid hard numpy dep


class ExportOutputs(TypedDict, total=False):
    snap: ClusterSnapshot
    # Wave 3a slice:
    out_path: str
    rows_written: int


def _as_cluster_snapshot(snap: Mapping[str, object]) -> ClusterSnapshot:
    """Narrow dynamic snapshot to the known pipeline contract shape."""
    out: ClusterSnapshot = {}
    for key in (
        "cluster_algo",
        "cluster_vec_mode",
        "call_col",
        "chat_col",
        "use_summary",
        "ignore_chatbot_cluster",
    ):
        val = snap.get(key)
        if val is not None:
            out[key] = val  # type: ignore[typeddict-item]
    return out


def build_cluster_role_context(snap: Mapping[str, object]) -> ClusterRoleContext:
    """Готовит role-specific snapshot для этапа подготовки входных текстов."""
    allowed_algos = {"kmeans", "agglo", "hdbscan", "bertopic", "lda", "fastopic"}
    allowed_vec_modes = {"tfidf", "sbert", "combo", "ensemble"}
    raw_algo = snap.get("cluster_algo", "kmeans")
    raw_vec_mode = snap.get("cluster_vec_mode", "tfidf")
    if not isinstance(raw_algo, str) or raw_algo not in allowed_algos:
        raise ValueError(f"Unsupported cluster_algo: {raw_algo!r}")
    if not isinstance(raw_vec_mode, str) or raw_vec_mode not in allowed_vec_modes:
        raise ValueError(f"Unsupported cluster_vec_mode: {raw_vec_mode!r}")

    role_mode = snap.get("cluster_role_mode", "all")
    cluster_snap: ClusterRoleSnapshot = {
        "cluster_algo": raw_algo,
        "cluster_vec_mode": raw_vec_mode,
        "desc_col": str(snap.get("desc_col", "")),
        "call_col": str(snap.get("call_col", "")),
        "chat_col": str(snap.get("chat_col", "")),
        "summary_col": str(snap.get("summary_col", "")),
        "ans_short_col": str(snap.get("ans_short_col", "")),
        "ans_full_col": str(snap.get("ans_full_col", "")),
        "use_summary": bool(snap.get("use_summary", False)),
        "ignore_chatbot_cluster": bool(snap.get("ignore_chatbot_cluster", True)),
    }
    base_w = snap.get("base_w")
    if isinstance(base_w, dict):
        cluster_snap["base_w"] = {
            str(k): int(v)
            for k, v in base_w.items()
            if isinstance(k, str) and isinstance(v, (int, float))
        }
    auto_profile = snap.get("auto_profile")
    if isinstance(auto_profile, str):
        cluster_snap["auto_profile"] = auto_profile
    cluster_snap["ignore_chatbot"] = bool(snap.get("ignore_chatbot_cluster", True))

    role_weights_client = {
        "w_desc": 2, "w_client": 3, "w_operator": 0,
        "w_summary": 0, "w_answer_short": 0, "w_answer_full": 0,
    }
    role_weights_operator = {
        "w_desc": 0, "w_client": 0, "w_operator": 3,
        "w_summary": 1, "w_answer_short": 2, "w_answer_full": 2,
    }
    if role_mode == "client":
        cluster_snap["base_w"] = role_weights_client
        cluster_snap["auto_profile"] = "off"
    elif role_mode == "operator":
        cluster_snap["base_w"] = role_weights_operator
        cluster_snap["auto_profile"] = "off"

    role_labels = {
        "all": "Весь диалог",
        "client": "Только клиент",
        "operator": "Только оператор",
    }
    role_label = role_labels.get(role_mode, role_mode)
    ignore_chatbot_label = "да" if cluster_snap["ignore_chatbot"] else "нет"
    return ClusterRoleContext(
        cluster_snap=cluster_snap,
        role_label=role_label,
        ignore_chatbot_label=ignore_chatbot_label,
    )


def build_t5_source_text(
    row_vals: Sequence,
    header: Sequence[str],
    snap: Mapping[str, object],
    cluster_snap: Mapping[str, object],
    header_index: Optional[dict[str, int]] = None,
) -> str:
    """Собирает текст строки для T5-суммаризации из call/chat столбцов."""
    index_map = header_index or {str(h): i for i, h in enumerate(header)}
    norm_index_map = {str(k).strip().lower(): v for k, v in index_map.items()}

    resolved_idx: dict[str, Optional[int]] = {}
    for col_key in ("call_col", "chat_col"):
        col_name_obj = snap.get(col_key, "")
        col_name = col_name_obj if isinstance(col_name_obj, str) else ""
        if not col_name:
            resolved_idx[col_key] = None
            continue
        idx = index_map.get(col_name)
        if idx is None:
            idx = norm_index_map.get(col_name.strip().lower())
        resolved_idx[col_key] = idx

    def _get(col_key: str) -> Optional[object]:
        idx = resolved_idx.get(col_key)
        return row_vals[idx] if (idx is not None and idx < len(row_vals)) else None

    parts: List[str] = []
    for col_key in ("call_col", "chat_col"):
        raw = _get(col_key)
        if raw:
            clean, _, _, _ = parse_dialog_roles(
                raw, ignore_chatbot=cluster_snap.get("ignore_chatbot", True),
            )
            if clean:
                parts.append(clean)
    return " ".join(parts)[:3000]


def prepare_inputs(files_snapshot: List[str], snap: Mapping[str, object]) -> PreparedInputs:
    """Stage 1: подготовка входов pipeline (без вычислительных этапов)."""
    role_context = build_cluster_role_context(snap)
    return PreparedInputs(
        files_snapshot=list(files_snapshot),
        snap=_as_cluster_snapshot(snap),
        role_context=role_context,
    )


_STAGES_NOT_PORTED_MSG = (
    "Pipeline combo {combo!r} is not yet ported. Wave 3a slice ships "
    "vec_mode='tfidf' + algo in {{'kmeans', 'agglo'}} end-to-end; other "
    "combinations still live inside app_cluster.run_cluster() (ADR-0002 tracks "
    "the full migration). Either select a supported combo or call the Tk-bound "
    "run_cluster() until this branch is ported."
)

_SUPPORTED_VEC_MODE = "tfidf"
_SUPPORTED_ALGOS = ("kmeans", "agglo")
# Agglomerative needs dense input; cap n_rows to keep O(n²) memory bounded.
_AGGLO_MAX_ROWS = 5_000


def _combo(snap: Mapping[str, object]) -> str:
    return f"{snap.get('cluster_vec_mode', '?')}+{snap.get('cluster_algo', '?')}"


def _require_supported_combo(snap: Mapping[str, object]) -> None:
    vm = snap.get("cluster_vec_mode")
    algo = snap.get("cluster_algo")
    if vm != _SUPPORTED_VEC_MODE or algo not in _SUPPORTED_ALGOS:
        raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(combo=_combo(snap)))


# ---------------------------------------------------------------------------
# Wave 3a slice — TF-IDF + KMeans implementation
# ---------------------------------------------------------------------------
# The Tk-bound run_cluster() in app_cluster.py supports many more vec_modes
# (sbert/combo/ensemble) and algos (hdbscan/bertopic/lda/fastopic). Porting
# all of them in one PR is a multi-day refactor — see ADR-0007. This slice
# implements only the simplest path so the headless CLI can run end-to-end
# for the most common case (TF-IDF + KMeans on a single text column).
#
# Out of scope for the slice:
#   - Role-aware text composition (use_summary, ignore_chatbot_cluster).
#     We read a single text_col from each input file and concatenate.
#   - Dedup, stop-word lists, T5 summarisation, MLM features.
#   - Incremental-model fast path (cluster_incremental_service).
#   - LLM cluster naming, merge-similar-clusters, silhouette diagnostics.
#   - GPU KMeans (cuML); we always use sklearn's MiniBatchKMeans.
# These remain reachable through the Tk path until further slices land.


def _read_texts(files_snapshot: Sequence[str], text_col: str) -> List[str]:
    """Read ``text_col`` from each file in ``files_snapshot``, concatenated."""
    from pathlib import Path as _Path

    from excel_utils import open_tabular

    texts: List[str] = []
    for raw_path in files_snapshot:
        path = _Path(raw_path)
        with open_tabular(path) as rows:
            try:
                header = next(rows)
            except StopIteration:
                continue
            normalised = [
                "" if h is None else str(h).strip() for h in header
            ]
            try:
                idx = normalised.index(text_col)
            except ValueError:
                raise ValueError(
                    f"Column {text_col!r} not found in {path.name}; "
                    f"available: {normalised}"
                )
            for row in rows:
                if idx >= len(row):
                    continue
                cell = row[idx]
                if cell is None:
                    continue
                s = str(cell).strip()
                if s:
                    texts.append(s)
    return texts


def _adaptive_min_df(n_rows: int) -> int:
    """Mirror the inline adaptive-min_df rule in run_cluster (line ~3088)."""
    if n_rows < 5_000:
        return 1
    if n_rows < 50_000:
        return 2
    return 3


def build_vectors(prepared: PreparedInputs, snap: Mapping[str, object]) -> VectorPack:
    """Stage 2: produce a TF-IDF sample × feature matrix.

    Slice-port: only ``snap['cluster_vec_mode'] == 'tfidf'`` is supported;
    other modes raise ``NotImplementedError`` with the offending combo
    named so the caller can route to the Tk path.
    """
    _require_supported_combo(snap)

    from sklearn.feature_extraction.text import TfidfVectorizer

    text_col_obj = snap.get("text_col")
    if not isinstance(text_col_obj, str) or not text_col_obj.strip():
        raise ValueError(
            "build_vectors requires snap['text_col'] (string, non-empty); "
            "the Tk path derives this from role-aware UI state, the slice does not."
        )
    text_col = text_col_obj.strip()

    texts = _read_texts(prepared.files_snapshot, text_col)
    if len(texts) < 2:
        raise ValueError(
            f"Need at least 2 non-empty rows in column {text_col!r}, got {len(texts)}"
        )

    user_min_df_obj = snap.get("cluster_min_df", 0)
    user_min_df = int(user_min_df_obj) if isinstance(user_min_df_obj, (int, float)) else 0
    min_df = user_min_df if user_min_df > 0 else _adaptive_min_df(len(texts))

    max_features_obj = snap.get("cluster_max_features", 50_000)
    max_features = int(max_features_obj) if isinstance(max_features_obj, (int, float)) else 50_000

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(texts)
    return VectorPack(
        prepared=prepared,
        vectors=matrix,
        meta={"vectorizer": vec, "texts": texts, "min_df": min_df},
    )


def run_clustering(vectors: VectorPack, snap: Mapping[str, object]) -> ClusterResult:
    """Stage 3: fit the configured algo on the TF-IDF matrix.

    Slice-port: ``snap['cluster_algo']`` ∈ {'kmeans', 'agglo'}. Other algos
    (hdbscan/bertopic/lda/fastopic, cuml/GPU KMeans) stay in the Tk path.

    Kmeans keeps the sparse matrix; agglo (hierarchical) densifies via
    ``.toarray()`` with a row guard — Ward linkage is O(n²) in memory.
    """
    _require_supported_combo(snap)
    if vectors.vectors is None:
        raise ValueError("run_clustering: VectorPack.vectors is None")

    k_obj = snap.get("k_clusters")
    if not isinstance(k_obj, (int, float)) or int(k_obj) < 2:
        raise ValueError(
            "run_clustering requires snap['k_clusters'] >= 2"
        )
    k = int(k_obj)

    n_rows = vectors.vectors.shape[0]
    if k > n_rows:
        raise ValueError(
            f"k_clusters={k} exceeds row count {n_rows}; pick a smaller K"
        )

    algo = str(snap.get("cluster_algo", ""))
    if algo == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        n_init_obj = snap.get("n_init_cluster", 10)
        n_init = int(n_init_obj) if isinstance(n_init_obj, (int, float)) else 10
        seed_obj = snap.get("random_state", 42)
        seed = int(seed_obj) if isinstance(seed_obj, (int, float)) else 42

        model: object = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=1024,
            n_init=n_init,
            init="k-means++",
            max_iter=300,
        )
        labels = model.fit_predict(vectors.vectors)  # type: ignore[attr-defined]
    elif algo == "agglo":
        from sklearn.cluster import AgglomerativeClustering

        if n_rows > _AGGLO_MAX_ROWS:
            raise ValueError(
                f"agglo clustering capped at {_AGGLO_MAX_ROWS} rows (got {n_rows}); "
                f"Ward linkage is O(n²) in memory — use kmeans for larger datasets"
            )
        dense = vectors.vectors.toarray()  # type: ignore[union-attr]
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(dense)  # type: ignore[attr-defined]
    else:
        # Unreachable: _require_supported_combo gates on the same set.
        raise NotImplementedError(f"run_clustering: algo={algo!r} not in slice")

    return ClusterResult(
        vectors=vectors,
        labels=labels,
        meta={"K": k, "algo": algo, "model": model},
    )


def _cluster_keywords(
    matrix: object, labels: object, vectorizer: object, *, top_n: int = 5
) -> dict[int, List[str]]:
    """Top-N centroid keywords per cluster (mean TF-IDF weight)."""
    import numpy as _np

    feature_names = vectorizer.get_feature_names_out()  # type: ignore[attr-defined]
    label_arr = _np.asarray(labels)
    out: dict[int, List[str]] = {}
    for cid in _np.unique(label_arr):
        if cid < 0:
            continue
        mask = label_arr == cid
        if not mask.any():
            continue
        # Sparse-row mean → 1×F matrix → flatten to ndarray.
        cluster_mean = _np.asarray(matrix[mask].mean(axis=0)).ravel()  # type: ignore[index]
        top_idx = _np.argsort(cluster_mean)[::-1][:top_n]
        out[int(cid)] = [str(feature_names[i]) for i in top_idx]
    return out


def postprocess_clusters(
    result: ClusterResult,
    prepared: PreparedInputs,
    snap: Mapping[str, object],
) -> PostprocessResult:
    """Stage 4: minimal — cluster sizes + centroid keywords.

    Slice-port: no LLM naming, no merge-similar-clusters, no silhouette.
    The Tk path's diagnostics live in ml_diagnostics; porting them is a
    follow-up slice once the matrix of algos is stable.
    """
    _require_supported_combo(snap)
    if result.labels is None or result.vectors.vectors is None:
        raise ValueError("postprocess_clusters: missing labels or vectors")

    import numpy as _np

    labels = _np.asarray(result.labels)
    sizes: dict[int, int] = {
        int(cid): int(count)
        for cid, count in zip(*_np.unique(labels, return_counts=True))
    }
    vec_meta = result.vectors.meta or {}
    keywords = _cluster_keywords(
        result.vectors.vectors, labels, vec_meta["vectorizer"],
    )
    payload: dict[str, object] = {
        "prepared": prepared,
        "snap": _as_cluster_snapshot(snap),
        "cluster_sizes": sizes,
        "cluster_keywords": keywords,
        "texts": vec_meta.get("texts", []),
        "labels": labels,
    }
    return PostprocessResult(result=result, payload=payload)  # type: ignore[arg-type]


def export_cluster_outputs(
    postprocessed: PostprocessResult, snap: Mapping[str, object]
) -> ExportSummary:
    """Stage 5: write a CSV with (text, cluster_id, top_keywords) per row.

    Slice-port: CSV only (no XLSX); ``snap['output_path']`` is required.
    The Tk path's XLSX exporter (with summary sheet) is a follow-up.
    """
    _require_supported_combo(snap)
    if postprocessed.payload is None:
        raise ValueError("export_cluster_outputs: postprocessed payload is None")

    out_obj = snap.get("output_path")
    if not isinstance(out_obj, str) or not out_obj.strip():
        raise ValueError(
            "export_cluster_outputs requires snap['output_path'] (string, .csv)"
        )
    from pathlib import Path as _Path

    out_path = _Path(out_obj.strip())
    if out_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Slice-port writes CSV only; got suffix {out_path.suffix!r}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = postprocessed.payload  # type: ignore[assignment]
    texts: Sequence[str] = payload["texts"]  # type: ignore[index]
    labels = payload["labels"]  # type: ignore[index]
    keywords: dict[int, List[str]] = payload["cluster_keywords"]  # type: ignore[index]

    import csv as _csv

    written = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.writer(fh)
        writer.writerow(["text", "cluster_id", "top_keywords"])
        for text, cid in zip(texts, labels):
            cid_i = int(cid)
            kw = ",".join(keywords.get(cid_i, []))
            writer.writerow([text, cid_i, kw])
            written += 1

    outputs: dict[str, object] = {
        "snap": _as_cluster_snapshot(snap),
        "out_path": str(out_path),
        "rows_written": written,
    }
    return ExportSummary(postprocessed=postprocessed, outputs=outputs)  # type: ignore[arg-type]
    raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(name="export_cluster_outputs"))
