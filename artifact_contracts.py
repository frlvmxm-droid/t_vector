# -*- coding: utf-8 -*-
"""Typed contracts and schema validators for persisted model artifacts."""
from __future__ import annotations

from typing import Any, Mapping, TypedDict

from exceptions import SchemaError


CLUSTER_MODEL_ARTIFACT_TYPE = "cluster_model_bundle"
TRAIN_MODEL_ARTIFACT_TYPE = "train_model_bundle"
SCHEMA_V1 = 1


# Единственный контракт для cluster-bundle.
# Дублирующие dataclass-версии удалены (нигде не использовались).
class ClusterModelBundleV1(TypedDict, total=False):
    artifact_type: str
    schema_version: int
    vectorizer: Any
    algo: str
    K: int
    kw: list[str]
    centers: Any
    model: Any


def validate_bundle_schema(
    bundle: Mapping[str, Any],
    *,
    expected_artifact_type: str,
    supported_schema_version: int = SCHEMA_V1,
    allow_missing_schema: bool = True,
) -> None:
    """Validates shared artifact identity invariants with stable error codes."""
    artifact_type = bundle.get("artifact_type")
    if artifact_type != expected_artifact_type:
        raise SchemaError(
            f"[ARTIFACT_TYPE_MISMATCH] artifact_type={artifact_type!r} "
            f"ожидалось={expected_artifact_type!r}."
        )

    schema_version = bundle.get("schema_version")
    if schema_version is None:
        if allow_missing_schema:
            return
        raise SchemaError("[SCHEMA_MISSING] В модели отсутствует schema_version.")
    if not isinstance(schema_version, int):
        raise SchemaError(
            f"[SCHEMA_TYPE_INVALID] schema_version={schema_version!r} (ожидается int)."
        )
    if schema_version > supported_schema_version:
        raise SchemaError(
            f"[SCHEMA_UNSUPPORTED] schema_version={schema_version}, "
            f"поддерживается ≤{supported_schema_version}."
        )
