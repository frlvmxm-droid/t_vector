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


class PostprocessPayload(TypedDict):
    prepared: PreparedInputs
    snap: ClusterSnapshot


class ExportOutputs(TypedDict):
    snap: ClusterSnapshot


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
    allowed_algos = {"kmeans", "hdbscan", "bertopic", "lda", "fastopic"}
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
    "Pipeline stage {name!r} is not yet ported. The real math still lives in "
    "app_cluster.run_cluster() around lines 3307–4273; ADR-0002 tracks the "
    "migration (Wave 3a). Calling the pipeline adapter directly would silently "
    "drop vectors/labels — see the previous no-op implementation. Use "
    "app_cluster.ClusterTabMixin.run_cluster() until the migration lands, or "
    "bypass this module with unit-tested replacements."
)


def build_vectors(prepared: PreparedInputs, snap: Mapping[str, object]) -> VectorPack:
    """Stage 2 (NOT PORTED): produce sample × feature vectors.

    Must return a populated ``VectorPack.vectors`` (e.g. ``scipy.sparse``
    matrix for TF-IDF or ``np.ndarray`` for SBERT). The previous stub
    returned ``vectors=None`` with ``meta={"snap": ...}``, which mimicked
    the shape of a real call while silently dropping the computation —
    that pretence is removed in favour of an explicit failure until the
    Wave 3a port of ``run_cluster()``'s Stage-2 block lands.
    """
    _ = (prepared, snap)  # kept for signature stability against the future port.
    raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(name="build_vectors"))


def run_clustering(vectors: VectorPack, snap: Mapping[str, object]) -> ClusterResult:
    """Stage 3 (NOT PORTED): fit a clustering algorithm and predict labels.

    Must return a populated ``ClusterResult.labels`` matching the algo
    branch selected via ``snap['cluster_algo']`` (kmeans / hdbscan /
    bertopic / lda / fastopic). The previous stub returned ``labels=None``.
    """
    _ = (vectors, snap)
    raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(name="run_clustering"))


def postprocess_clusters(
    result: ClusterResult,
    prepared: PreparedInputs,
    snap: Mapping[str, object],
) -> PostprocessResult:
    """Stage 4 (NOT PORTED): merge + name + diagnose clusters.

    Must return a populated ``PostprocessResult.payload`` with merged
    labels, cluster names (LLM or centroid-keyword), and diagnostics
    (silhouette / representative texts / noise share).
    """
    _ = (result, prepared, snap)
    raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(name="postprocess_clusters"))


def export_cluster_outputs(postprocessed: PostprocessResult, snap: Mapping[str, object]) -> ExportSummary:
    """Stage 5 (NOT PORTED): write XLSX/CSV outputs with per-row cluster assignments."""
    _ = (postprocessed, snap)
    raise NotImplementedError(_STAGES_NOT_PORTED_MSG.format(name="export_cluster_outputs"))
