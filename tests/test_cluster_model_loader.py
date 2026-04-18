from pathlib import Path

import joblib
import pytest

from cluster_model_loader import (
    extract_kw_list,
    load_cluster_model_bundle,
    normalize_incremental_bundle_payload,
)
from exceptions import ModelLoadError


def test_load_cluster_model_bundle_rejects_invalid_algo_type(tmp_path: Path):
    p = tmp_path / "cluster.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "artifact_type": "cluster_model_bundle",
            "vectorizer": object(),
            "K": 3,
            "kw": {},
            "algo": 123,
        },
        p,
    )
    with pytest.raises(ModelLoadError):
        load_cluster_model_bundle(str(p), schema_version=1, trusted_paths=[str(p)])


def test_normalize_incremental_bundle_payload_rejects_unknown_algo():
    with pytest.raises(ModelLoadError):
        normalize_incremental_bundle_payload(
            {"vectorizer": object(), "algo": "other", "K": 1, "kw": ["a"]}
        )


def test_extract_kw_list_supports_dict_payload():
    bundle = {"kw": {"0": "a", 1: "b"}}
    assert extract_kw_list(bundle, 3) == ["a", "b", ""]


def test_extract_kw_list_supports_list_payload():
    bundle = {"kw": ["x", "y"]}
    assert extract_kw_list(bundle, 3) == ["x", "y", ""]


def test_extract_kw_list_supports_tuple_payload():
    bundle = {"kw": ("x", "y")}
    assert extract_kw_list(bundle, 3) == ["x", "y", ""]


def test_load_cluster_model_bundle_happy_path_with_kw_dict(tmp_path: Path):
    p = tmp_path / "cluster_ok.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "artifact_type": "cluster_model_bundle",
            "vectorizer": object(),
            "K": 3,
            "kw": {"0": "payments", 1: "limits", "2": "cards"},
            "algo": "kmeans",
        },
        p,
    )
    bundle = load_cluster_model_bundle(str(p), schema_version=1, trusted_paths=[str(p)])
    assert extract_kw_list(bundle, int(bundle["K"])) == ["payments", "limits", "cards"]


def test_loader_normalizer_postconditions_integration(tmp_path: Path):
    p = tmp_path / "cluster_ok2.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "artifact_type": "cluster_model_bundle",
            "vectorizer": object(),
            "K": 2,
            "kw": {"0": "payments", "1": "limits"},
            "algo": "kmeans",
            "centers": [[0.0], [1.0]],
        },
        p,
    )
    loaded = load_cluster_model_bundle(str(p), schema_version=1, trusted_paths=[str(p)])
    norm = normalize_incremental_bundle_payload(loaded)
    assert norm.k_clusters == 2
    assert norm.kw == ["payments", "limits"]
    assert norm.use_fastopic_kw_ready is True


def test_load_cluster_model_bundle_rejects_unsupported_algo_value(tmp_path: Path):
    p = tmp_path / "cluster_bad_algo.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "artifact_type": "cluster_model_bundle",
            "vectorizer": object(),
            "K": 3,
            "kw": {},
            "algo": "gmm",
        },
        p,
    )
    with pytest.raises(ModelLoadError, match="UNSUPPORTED_CLUSTER_ALGO"):
        load_cluster_model_bundle(str(p), schema_version=1, trusted_paths=[str(p)])


def test_load_cluster_model_bundle_requires_trusted_path(tmp_path: Path):
    p = tmp_path / "cluster_untrusted.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "artifact_type": "cluster_model_bundle",
            "vectorizer": object(),
            "K": 2,
            "kw": {},
            "algo": "kmeans",
        },
        p,
    )
    with pytest.raises(ModelLoadError, match="UNTRUSTED_MODEL_PATH"):
        load_cluster_model_bundle(str(p), schema_version=1, trusted_paths=[])
