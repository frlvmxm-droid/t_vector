# -*- coding: utf-8 -*-
"""Сервисные helper-функции безопасной загрузки cluster model bundle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

from artifact_contracts import (
    CLUSTER_MODEL_ARTIFACT_TYPE,
    ClusterModelBundleV1,
    validate_bundle_schema,
)
from exceptions import ModelLoadError
from model_loader import ConfirmFn, TrustStore, ensure_trusted_model_path, load_model_artifact


@dataclass(frozen=True)
class IncrementalBundle:
    vectorizer: Any
    algo: str
    k_clusters: int
    kw: list[str]
    use_fastopic_kw_ready: bool
    centers: Any = None
    model: Any = None


def ensure_cluster_model_trusted(
    store: TrustStore,
    path: str,
    *,
    confirm_fn: ConfirmFn,
    label: str = "Модель кластеризации",
    logger: Any = None,
) -> bool:
    return ensure_trusted_model_path(store, path, label=label, confirm_fn=confirm_fn, logger=logger)


def load_cluster_model_bundle(
    path: str,
    *,
    schema_version: int = 1,
    trusted_paths: Sequence[str] | None = None,
    logger: Any = None,
) -> ClusterModelBundleV1:
    bundle = load_model_artifact(
        path,
        supported_schema_version=schema_version,
        expected_artifact_types=(CLUSTER_MODEL_ARTIFACT_TYPE,),
        required_keys=("vectorizer", "K", "kw", "algo"),
        required_key_types={
            "K": int,
            "kw": (list, tuple, dict),
            "algo": str,
        },
        allow_missing_schema=False,
        allowed_extensions=(".joblib",),
        require_trusted=trusted_paths is not None,
        trusted_paths=trusted_paths,
        logger=logger,
    )
    validate_bundle_schema(
        bundle,
        expected_artifact_type=CLUSTER_MODEL_ARTIFACT_TYPE,
        supported_schema_version=schema_version,
        allow_missing_schema=False,
    )
    algo = str(bundle.get("algo", ""))
    if algo not in {"kmeans", "hdbscan"}:
        raise ModelLoadError(
            f"[UNSUPPORTED_CLUSTER_ALGO] Неподдерживаемый algo в cluster bundle: {algo!r}."
        )
    return cast(ClusterModelBundleV1, bundle)


def extract_kw_list(bundle: Mapping[str, Any], k_clusters: int) -> list[str]:
    """Нормализует bundle['kw'] в список ключевых фраз длиной K."""
    raw_kw = bundle.get("kw", [])
    if isinstance(raw_kw, Sequence) and not isinstance(raw_kw, (str, bytes, dict)):
        out = [str(x) if x is not None else "" for x in raw_kw]
        if len(out) < k_clusters:
            out.extend([""] * (k_clusters - len(out)))
        return out[:k_clusters]
    if isinstance(raw_kw, dict):
        out = [""] * k_clusters
        for i in range(k_clusters):
            out[i] = str(raw_kw.get(i, raw_kw.get(str(i), "")) or "")
        return out
    raise ModelLoadError(
        f"Неверный тип ключа 'kw': {type(raw_kw).__name__} "
        "(ожидалось: list/tuple или dict)."
    )


def normalize_incremental_bundle_payload(
    saved: Mapping[str, Any],
) -> IncrementalBundle:
    """Подготавливает bundle для incremental runtime-ветки app_cluster."""
    if "vectorizer" not in saved:
        raise ModelLoadError("Инкрементальная модель не содержит 'vectorizer'")
    vec = saved["vectorizer"]
    algo = str(saved.get("algo", "kmeans"))
    if algo not in {"kmeans", "hdbscan"}:
        raise ModelLoadError(f"Неподдерживаемый algo для incremental bundle: {algo!r}")
    if "K" not in saved:
        raise ModelLoadError("Инкрементальная модель не содержит 'K' (число кластеров)")
    k_clusters = int(saved["K"])
    kw_list = extract_kw_list(saved, k_clusters)
    use_fastopic_kw_ready = len(kw_list) == k_clusters and any(kw_list)
    return IncrementalBundle(
        vectorizer=vec,
        algo=algo,
        k_clusters=k_clusters,
        kw=kw_list,
        use_fastopic_kw_ready=use_fastopic_kw_ready,
        centers=saved.get("centers"),
        model=saved.get("model"),
    )
