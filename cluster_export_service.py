# -*- coding: utf-8 -*-
"""Export helpers for cluster model artifacts."""
from __future__ import annotations

from typing import Any

from artifact_contracts import CLUSTER_MODEL_ARTIFACT_TYPE, ClusterModelBundleV1


def build_cluster_model_bundle(
    *,
    schema_version: int,
    vectorizer: Any,
    algo: str,
    k_clusters: int,
    kw: list[str],
    centers: Any = None,
    model: Any = None,
) -> ClusterModelBundleV1:
    bundle: ClusterModelBundleV1 = {
        "artifact_type": CLUSTER_MODEL_ARTIFACT_TYPE,
        "schema_version": schema_version,
        "vectorizer": vectorizer,
        "algo": algo,
        "K": k_clusters,
        "kw": list(kw),
        "centers": centers,
    }
    if model is not None:
        bundle["model"] = model
    return bundle
