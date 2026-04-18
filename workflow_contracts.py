"""Контракты конфигурации workflow-слоя (валидация снапшотов UI)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from pydantic import BaseModel, ValidationError
except (ImportError, ModuleNotFoundError):  # optional dependency
    BaseModel = None  # type: ignore[assignment]
    ValidationError = Exception


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Отсутствует обязательный параметр: {key}")
    return data[key]


def _as_str(data: dict[str, Any], key: str, *, min_len: int = 1) -> str:
    val = _require(data, key)
    if not isinstance(val, str):
        raise ValueError(f"Параметр {key} должен быть строкой.")
    out = val.strip()
    if len(out) < min_len:
        raise ValueError(f"Параметр {key} не должен быть пустым.")
    return out


def _as_float(data: dict[str, Any], key: str, *, min_value: float | None = None, max_value: float | None = None) -> float:
    val = _require(data, key)
    try:
        out = float(val)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"Параметр {key} должен быть числом.") from ex
    if not math.isfinite(out):
        raise ValueError(f"Параметр {key} должен быть конечным числом.")
    if min_value is not None and out < min_value:
        raise ValueError(f"Параметр {key} должен быть >= {min_value}.")
    if max_value is not None and out > max_value:
        raise ValueError(f"Параметр {key} должен быть <= {max_value}.")
    return out


def _as_int(data: dict[str, Any], key: str, *, min_value: int | None = None, max_value: int | None = None) -> int:
    val = _require(data, key)
    try:
        out = int(val)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"Параметр {key} должен быть целым числом.") from ex
    if min_value is not None and out < min_value:
        raise ValueError(f"Параметр {key} должен быть >= {min_value}.")
    if max_value is not None and out > max_value:
        raise ValueError(f"Параметр {key} должен быть <= {max_value}.")
    return out


def _manual_validate_payload(schema_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Строгая ручная валидация при отсутствии pydantic."""
    out = dict(payload)
    if schema_name == "train":
        out["train_mode"] = _as_str(out, "train_mode")
        out["C"] = _as_float(out, "C", min_value=0.0)
        out["max_iter"] = _as_int(out, "max_iter", min_value=1)
        out["test_size"] = _as_float(out, "test_size", min_value=0.0, max_value=0.95)
        out.setdefault("use_smote", True)
        out.setdefault("oversample_strategy", "augment_light")
        out.setdefault("diagnostic_mode", False)
    elif schema_name == "apply":
        out["model_file"] = _as_str(out, "model_file")
        out["apply_file"] = _as_str(out, "apply_file")
        out["pred_col"] = _as_str(out, "pred_col")
        out.setdefault("use_ensemble", False)
        out.setdefault("diagnostic_mode", False)
    elif schema_name == "cluster":
        out["cluster_algo"] = _as_str(out, "cluster_algo")
        out["cluster_vec_mode"] = _as_str(out, "cluster_vec_mode")
        out["k_clusters"] = _as_int(out, "k_clusters", min_value=2, max_value=500)
        out["n_init_cluster"] = _as_int(out, "n_init_cluster", min_value=1, max_value=1000)
        out["cluster_min_df"] = _as_int(out, "cluster_min_df", min_value=0, max_value=1000)
        out["use_umap"] = bool(_require(out, "use_umap"))
        out.setdefault("use_llm_naming", False)
        out.setdefault("use_t5_summary", False)
        out.setdefault("diagnostic_mode", False)
        out.setdefault("merge_similar_clusters", False)
        out.setdefault("merge_threshold", 0.85)
        out.setdefault("n_repr_examples", 5)
    else:
        raise ValueError(f"Неизвестная схема валидации: {schema_name}")
    return out


def _validate_payload(schema, payload: dict[str, Any], *, schema_name: str) -> dict[str, Any]:
    """Централизованная валидация payload: Pydantic (если доступен) или ручная."""
    if BaseModel is not None and schema is not None:
        try:
            obj = schema(**payload)
            return obj.model_dump()
        except ValidationError as ex:
            raise ValueError(str(ex)) from ex
    return _manual_validate_payload(schema_name, payload)


if BaseModel is not None:
    class _TrainSchema(BaseModel):
        train_mode: str
        C: float
        max_iter: int
        test_size: float
        # Опциональные поля с defaults, совпадающими с from_snapshot()
        use_smote: bool = True
        oversample_strategy: str = "augment_light"
        diagnostic_mode: bool = False

    class _ApplySchema(BaseModel):
        model_file: str
        apply_file: str
        pred_col: str
        use_ensemble: bool = False
        diagnostic_mode: bool = False

    class _ClusterSchema(BaseModel):
        cluster_algo: str
        cluster_vec_mode: str
        k_clusters: int
        n_init_cluster: int
        cluster_min_df: int
        use_umap: bool
        use_llm_naming: bool = False
        use_t5_summary: bool = False
        diagnostic_mode: bool = False
        merge_similar_clusters: bool = False
        merge_threshold: float = 0.85
        n_repr_examples: int = 5
else:
    _TrainSchema = _ApplySchema = _ClusterSchema = None


@dataclass(frozen=True)
class TrainWorkflowConfig:
    train_mode: str
    c_value: float
    max_iter: int
    test_size: float
    use_smote: bool
    oversample_strategy: str
    diagnostic_mode: bool

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> TrainWorkflowConfig:
        validated = _validate_payload(_TrainSchema, snap, schema_name="train")
        return cls(
            train_mode=str(_require(validated, "train_mode")),
            c_value=float(_require(validated, "C")),
            max_iter=int(_require(validated, "max_iter")),
            test_size=float(_require(validated, "test_size")),
            use_smote=bool(validated.get("use_smote", True)),
            oversample_strategy=str(validated.get("oversample_strategy", "augment_light")),
            diagnostic_mode=bool(validated.get("diagnostic_mode", False)),
        )


@dataclass(frozen=True)
class ApplyWorkflowConfig:
    model_file: str
    apply_file: str
    pred_col: str
    use_ensemble: bool
    diagnostic_mode: bool

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> ApplyWorkflowConfig:
        validated = _validate_payload(_ApplySchema, snap, schema_name="apply")
        return cls(
            model_file=str(_require(validated, "model_file")),
            apply_file=str(_require(validated, "apply_file")),
            pred_col=str(_require(validated, "pred_col")),
            use_ensemble=bool(validated.get("use_ensemble", False)),
            diagnostic_mode=bool(validated.get("diagnostic_mode", False)),
        )


@dataclass(frozen=True)
class ClusterWorkflowConfig:
    use_llm_naming: bool
    use_t5_summary: bool
    algo: str
    cluster_vec_mode: str
    k_clusters: int
    n_init_cluster: int
    cluster_min_df: int
    use_umap: bool
    diagnostic_mode: bool

    @classmethod
    def from_snapshot(cls, snap: dict[str, Any]) -> ClusterWorkflowConfig:
        validated = _validate_payload(_ClusterSchema, snap, schema_name="cluster")
        cluster_algo = str(_require(validated, "cluster_algo"))
        cluster_vec_mode = str(_require(validated, "cluster_vec_mode"))
        k_clusters = int(_require(validated, "k_clusters"))
        n_init_cluster = int(_require(validated, "n_init_cluster"))
        cluster_min_df = int(_require(validated, "cluster_min_df"))
        use_umap = bool(_require(validated, "use_umap"))
        allowed_algos = {"kmeans", "hdbscan", "lda", "bertopic", "fastopic"}
        allowed_vec_modes = {"tfidf", "sbert", "combo", "ensemble"}
        if cluster_algo not in allowed_algos:
            raise ValueError(f"Недопустимый cluster_algo: {cluster_algo}")
        if cluster_vec_mode not in allowed_vec_modes:
            raise ValueError(f"Недопустимый cluster_vec_mode: {cluster_vec_mode}")
        return cls(
            use_llm_naming=bool(validated.get("use_llm_naming", False)),
            use_t5_summary=bool(validated.get("use_t5_summary", False)),
            algo=cluster_algo,
            cluster_vec_mode=cluster_vec_mode,
            k_clusters=k_clusters,
            n_init_cluster=n_init_cluster,
            cluster_min_df=cluster_min_df,
            use_umap=use_umap,
            diagnostic_mode=bool(validated.get("diagnostic_mode", False)),
        )
