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
    "Pipeline combo {combo!r} is not yet ported. The Wave 3a slice ships "
    "vec_mode='tfidf' + algo in {{'kmeans','agglo','lda','hdbscan'}}, "
    "vec_mode='sbert'|'combo'|'ensemble' + algo='kmeans' end-to-end; other "
    "combinations still live inside app_cluster.run_cluster() (ADR-0002 "
    "tracks the full migration). Either select a supported combo or call "
    "the Tk-bound run_cluster() until this branch is ported."
)

_SUPPORTED_VEC_MODES = ("tfidf", "sbert", "combo", "ensemble")
_SUPPORTED_ALGOS = ("kmeans", "agglo", "lda", "hdbscan")
# sbert, combo and ensemble currently pair only with kmeans. The Tk path
# does run hdbscan/agglo on dense SBERT / combo vectors, but porting those
# would pull in SBERT-specific density heuristics + anchor handling; they
# are deferred to later sub-commits. Ensemble is kmeans-only by design:
# the silhouette selector fits KMeans to each candidate vectorisation and
# picks the winner, so supporting a non-kmeans algo there is meaningless.
_SUPPORTED_SBERT_ALGOS = ("kmeans",)
_SUPPORTED_COMBO_ALGOS = ("kmeans",)
_SUPPORTED_ENSEMBLE_ALGOS = ("kmeans",)
# Agglomerative needs dense input; cap n_rows to keep O(n²) memory bounded.
_AGGLO_MAX_ROWS = 5_000


def _combo(snap: Mapping[str, object]) -> str:
    return f"{snap.get('cluster_vec_mode', '?')}+{snap.get('cluster_algo', '?')}"


def _require_supported_combo(snap: Mapping[str, object]) -> None:
    vm = snap.get("cluster_vec_mode")
    algo = snap.get("cluster_algo")
    if vm not in _SUPPORTED_VEC_MODES or algo not in _SUPPORTED_ALGOS:
        raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(combo=_combo(snap)))
    if vm == "sbert" and algo not in _SUPPORTED_SBERT_ALGOS:
        raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(combo=_combo(snap)))
    if vm == "combo" and algo not in _SUPPORTED_COMBO_ALGOS:
        raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(combo=_combo(snap)))
    if vm == "ensemble" and algo not in _SUPPORTED_ENSEMBLE_ALGOS:
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
    """Stage 2: produce the clustering matrix (+ a TF-IDF keyword matrix).

    Slice-port: only ``snap['cluster_vec_mode'] == 'tfidf'`` is supported.
    For ``algo='lda'`` a ``CountVectorizer`` is built *additionally* for
    the LDA fit (sklearn's LDA expects count features, not TF-IDF); the
    TF-IDF keyword matrix lives alongside so postprocess still extracts
    TF-IDF-weighted top terms — matching the Tk path at line 3714.
    """
    _require_supported_combo(snap)

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

    tfidf_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf_vec.fit_transform(texts)

    vec_mode = str(snap.get("cluster_vec_mode", ""))
    algo = str(snap.get("cluster_algo", ""))
    if vec_mode == "sbert":
        # Mirrors app_cluster.py:3666-3687 — dense SBERT embeddings drive
        # clustering, TF-IDF matrix stays alongside purely for keyword
        # extraction in postprocess.
        from ml_vectorizers import SBERTVectorizer

        model_name_obj = snap.get("sbert_model", "cointegrated/rubert-tiny2")
        model_name = str(model_name_obj) if model_name_obj else "cointegrated/rubert-tiny2"
        batch_obj = snap.get("sbert_batch", 32)
        batch_size = int(batch_obj) if isinstance(batch_obj, (int, float)) else 32
        device_obj = snap.get("sbert_device", "auto")
        device = str(device_obj) if device_obj else "auto"

        sbert = SBERTVectorizer(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )
        clustering_matrix = sbert.fit_transform(texts)
        clustering_vectorizer: object = sbert
    elif vec_mode == "combo":
        # Mirrors app_cluster.py:3622-3662 — TF-IDF → TruncatedSVD → L2 +
        # SBERT → L2, hstack with α weight. The Tk path puts the combo
        # behind a mouse-driven α slider; here we just read it from snap.
        import numpy as _np
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize as _sk_normalize

        from ml_vectorizers import SBERTVectorizer

        svd_dim_obj = snap.get("combo_svd_dim", 200)
        svd_dim_req = int(svd_dim_obj) if isinstance(svd_dim_obj, (int, float)) else 200
        # SVD requires n_components < min(n_samples, n_features); floor at 10
        # so the combo matrix has enough spread to separate clusters.
        svd_dim = max(10, min(svd_dim_req, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1))

        seed_obj = snap.get("random_state", 42)
        seed = int(seed_obj) if isinstance(seed_obj, (int, float)) else 42
        svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        tfidf_norm = _sk_normalize(tfidf_reduced, norm="l2")

        model_name_obj = snap.get("sbert_model", "cointegrated/rubert-tiny2")
        model_name = str(model_name_obj) if model_name_obj else "cointegrated/rubert-tiny2"
        batch_obj = snap.get("sbert_batch", 32)
        batch_size = int(batch_obj) if isinstance(batch_obj, (int, float)) else 32
        device_obj = snap.get("sbert_device", "auto")
        device = str(device_obj) if device_obj else "auto"
        sbert = SBERTVectorizer(
            model_name=model_name, batch_size=batch_size, device=device,
        )
        sbert_matrix = sbert.fit_transform(texts)
        sbert_norm = _sk_normalize(sbert_matrix, norm="l2")

        alpha_obj = snap.get("combo_alpha", 0.5)
        alpha = float(alpha_obj) if isinstance(alpha_obj, (int, float)) else 0.5
        # Clamp α to [0, 1] so misconfigured snapshots can't flip the sign
        # (the Tk UI slider does the same at the widget level).
        alpha = max(0.0, min(1.0, alpha))
        clustering_matrix = _np.hstack([
            tfidf_norm * (1.0 - alpha),
            sbert_norm * alpha,
        ])
        clustering_vectorizer = {
            "svd": svd, "sbert": sbert, "alpha": alpha, "svd_dim": svd_dim,
        }
    elif vec_mode == "ensemble":
        # Mirrors app_cluster.py:3761-3871 — fit TF-IDF + two SBERT models,
        # run KMeans on each, score silhouette, keep the winner. Unlike
        # combo (which blends all vectorisations into one matrix), ensemble
        # selects *one* candidate and discards the others. The selector
        # is folded into build_vectors so run_clustering can stay a thin
        # dispatch — the winner's labels are stashed in meta and
        # short-circuited through in stage 3.
        import numpy as _np
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score as _sil
        from sklearn.preprocessing import normalize as _sk_normalize

        from ml_vectorizers import SBERTVectorizer

        k_obj = snap.get("k_clusters")
        if not isinstance(k_obj, (int, float)) or int(k_obj) < 2:
            raise ValueError(
                "ensemble build_vectors requires snap['k_clusters'] >= 2"
            )
        k_ens = int(k_obj)
        seed_obj = snap.get("random_state", 42)
        seed = int(seed_obj) if isinstance(seed_obj, (int, float)) else 42

        def _dense_and_norm(mat: object) -> object:
            import scipy.sparse as _sp  # type: ignore[import-not-found]
            arr = mat.toarray() if _sp.issparse(mat) else _np.asarray(mat)  # type: ignore[union-attr]
            # Tk path normalises only when use_cosine_cluster is set; we
            # always L2-normalise here so silhouette scores stay comparable
            # across TF-IDF (raw counts) and SBERT (already unit-norm) —
            # otherwise the selector biases toward whichever matrix has
            # larger native scale.
            return _sk_normalize(arr, norm="l2")

        def _fit_and_score(dense: object, tag: str) -> tuple[object, float]:
            km = MiniBatchKMeans(
                n_clusters=k_ens,
                random_state=seed,
                batch_size=1024,
                n_init=10,
                init="k-means++",
                max_iter=300,
            )
            lbl = km.fit_predict(dense)  # type: ignore[arg-type]
            try:
                sample = min(5_000, dense.shape[0])  # type: ignore[attr-defined]
                sc = float(_sil(dense, lbl, sample_size=sample, random_state=seed))  # type: ignore[arg-type]
            except Exception:
                sc = -1.0
            return (lbl, sc)

        # Candidate 1: TF-IDF (already fit above).
        tfidf_dense = _dense_and_norm(tfidf_matrix)
        lbl_tfidf, sc_tfidf = _fit_and_score(tfidf_dense, "tfidf")

        # Candidates 2 & 3: two SBERT models. sbert_model2 defaults to
        # sbert_model — matches the Tk fallback at app_cluster.py:3831.
        model1 = str(snap.get("sbert_model", "cointegrated/rubert-tiny2"))
        model2 = str(snap.get("sbert_model2", model1))
        batch_obj = snap.get("sbert_batch", 32)
        batch_size = int(batch_obj) if isinstance(batch_obj, (int, float)) else 32
        device = str(snap.get("sbert_device", "auto"))

        sbert1 = SBERTVectorizer(
            model_name=model1, batch_size=batch_size, device=device,
        )
        s1_dense = _dense_and_norm(sbert1.fit_transform(texts))
        lbl_s1, sc_s1 = _fit_and_score(s1_dense, f"sbert:{model1}")

        sbert2 = SBERTVectorizer(
            model_name=model2, batch_size=batch_size, device=device,
        )
        s2_dense = _dense_and_norm(sbert2.fit_transform(texts))
        lbl_s2, sc_s2 = _fit_and_score(s2_dense, f"sbert:{model2}")

        candidates = [
            ("tfidf",           tfidf_dense, lbl_tfidf, sc_tfidf, tfidf_vec),
            (f"sbert:{model1}", s1_dense,    lbl_s1,    sc_s1,    sbert1),
            (f"sbert:{model2}", s2_dense,    lbl_s2,    sc_s2,    sbert2),
        ]
        winner = max(candidates, key=lambda c: c[3])
        win_name, win_dense, win_labels, win_score, win_vec = winner

        clustering_matrix = win_dense
        clustering_vectorizer = {
            "ensemble_winner": win_name,
            "ensemble_silhouette": win_score,
            "ensemble_scores": {name: score for name, _, _, score, _ in candidates},
            "winning_vectorizer": win_vec,
        }
        # Stash winner's labels so run_clustering can short-circuit the
        # kmeans refit (we already picked the best one based on silhouette).
        _ensemble_labels = win_labels
    elif algo == "lda":
        count_vec = CountVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
        )
        clustering_matrix = count_vec.fit_transform(texts)
        clustering_vectorizer = count_vec
    else:
        clustering_matrix = tfidf_matrix
        clustering_vectorizer = tfidf_vec

    meta: dict[str, object] = {
        "vectorizer": clustering_vectorizer,
        "keyword_vectorizer": tfidf_vec,
        "keyword_matrix": tfidf_matrix,
        "texts": texts,
        "min_df": min_df,
    }
    if vec_mode == "ensemble":
        # run_clustering reads meta['ensemble_labels'] and skips the
        # kmeans refit (the selector above already fit + scored three).
        meta["ensemble_labels"] = _ensemble_labels

    return VectorPack(
        prepared=prepared,
        vectors=clustering_matrix,
        meta=meta,
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

    algo = str(snap.get("cluster_algo", ""))
    vec_mode = str(snap.get("cluster_vec_mode", ""))
    n_rows = vectors.vectors.shape[0]

    # Ensemble short-circuit: build_vectors already fit KMeans on every
    # candidate vectorisation and picked the winner by silhouette; the
    # winner's labels live in meta and we hand them straight back.
    if vec_mode == "ensemble":
        vmeta = vectors.meta or {}
        ensemble_labels = vmeta.get("ensemble_labels")
        if ensemble_labels is None:
            raise ValueError(
                "ensemble run_clustering: meta['ensemble_labels'] missing"
            )
        return ClusterResult(
            vectors=vectors,
            labels=ensemble_labels,
            meta={
                "K": int(snap.get("k_clusters", 0) or 0),
                "algo": algo,
                "ensemble_winner": vmeta.get("vectorizer", {}).get(
                    "ensemble_winner"
                ) if isinstance(vmeta.get("vectorizer"), dict) else None,
            },
        )

    # HDBSCAN discovers K on its own; all other slice algos need explicit K.
    if algo != "hdbscan":
        k_obj = snap.get("k_clusters")
        if not isinstance(k_obj, (int, float)) or int(k_obj) < 2:
            raise ValueError(
                "run_clustering requires snap['k_clusters'] >= 2"
            )
        k = int(k_obj)
        if k > n_rows:
            raise ValueError(
                f"k_clusters={k} exceeds row count {n_rows}; pick a smaller K"
            )
    else:
        k = 0  # discovered post-fit; kept in meta for downstream logging

    seed_obj = snap.get("random_state", 42)
    seed = int(seed_obj) if isinstance(seed_obj, (int, float)) else 42
    if algo == "kmeans":
        from sklearn.cluster import MiniBatchKMeans

        n_init_obj = snap.get("n_init_cluster", 10)
        n_init = int(n_init_obj) if isinstance(n_init_obj, (int, float)) else 10

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
    elif algo == "lda":
        import numpy as _np
        from sklearn.decomposition import LatentDirichletAllocation

        max_iter_obj = snap.get("lda_max_iter", 10)
        max_iter = int(max_iter_obj) if isinstance(max_iter_obj, (int, float)) else 10

        model = LatentDirichletAllocation(
            n_components=k,
            max_iter=max_iter,
            learning_method="online",
            random_state=seed,
        )
        doc_topics = model.fit_transform(vectors.vectors)  # type: ignore[attr-defined]
        labels = _np.argmax(doc_topics, axis=1)
    elif algo == "hdbscan":
        import numpy as _np

        # Prefer sklearn's in-tree HDBSCAN (≥1.3); fall back to the standalone
        # hdbscan package — mirrors the Tk path (app_cluster.py:4031-4035).
        try:
            from sklearn.cluster import HDBSCAN as _HDBSCAN_cls
        except ImportError:
            import hdbscan as _hdbscan_mod

            _HDBSCAN_cls = _hdbscan_mod.HDBSCAN

        # Densify: HDBSCAN on the standalone package does not support sparse.
        dense = vectors.vectors.toarray()  # type: ignore[union-attr]

        min_cs_obj = snap.get("hdbscan_min_cluster_size", 0)
        try:
            min_cs = int(min_cs_obj) if min_cs_obj is not None else 0
        except (TypeError, ValueError):
            min_cs = 0
        if min_cs <= 0:
            # Sentinel: 0 or negative → sqrt(N) heuristic (matches Tk path).
            min_cs = max(5, int(n_rows ** 0.5))

        hdb_kwargs: dict[str, object] = {"min_cluster_size": min_cs}
        min_samples_obj = snap.get("hdbscan_min_samples", 0)
        try:
            min_samples = int(min_samples_obj) if min_samples_obj is not None else 0
        except (TypeError, ValueError):
            min_samples = 0
        if min_samples > 0:
            hdb_kwargs["min_samples"] = min_samples
        eps_obj = snap.get("hdbscan_eps", 0.0)
        try:
            eps = float(eps_obj) if eps_obj is not None else 0.0
        except (TypeError, ValueError):
            eps = 0.0
        if eps > 0.0:
            hdb_kwargs["cluster_selection_epsilon"] = eps

        model = _HDBSCAN_cls(**hdb_kwargs)
        labels = model.fit_predict(dense)  # type: ignore[attr-defined]
        labels_arr = _np.asarray(labels)
        valid = labels_arr[labels_arr >= 0]
        k = int(valid.max()) + 1 if valid.size else 0
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
    # Keyword extraction prefers the TF-IDF matrix when one exists (LDA
    # fits on counts but TF-IDF-weighted keywords are more informative —
    # matching run_cluster()'s behavior at line 3714 / 2555).
    vec_meta = result.vectors.meta or {}
    kw_matrix = vec_meta.get("keyword_matrix", result.vectors.vectors)
    kw_vectorizer = vec_meta.get("keyword_vectorizer", vec_meta["vectorizer"])
    keywords = _cluster_keywords(kw_matrix, labels, kw_vectorizer)
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
