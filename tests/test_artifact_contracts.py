import pytest

import artifact_contracts
from artifact_contracts import (
    CLUSTER_MODEL_ARTIFACT_TYPE,
    SCHEMA_V1,
    TRAIN_MODEL_ARTIFACT_TYPE,
    validate_bundle_schema,
)
from exceptions import SchemaError


def test_validate_bundle_schema_accepts_v1_cluster():
    validate_bundle_schema(
        {"artifact_type": CLUSTER_MODEL_ARTIFACT_TYPE, "schema_version": SCHEMA_V1},
        expected_artifact_type=CLUSTER_MODEL_ARTIFACT_TYPE,
        allow_missing_schema=False,
    )


def test_validate_bundle_schema_rejects_artifact_type():
    with pytest.raises(SchemaError, match="ARTIFACT_TYPE_MISMATCH"):
        validate_bundle_schema(
            {"artifact_type": TRAIN_MODEL_ARTIFACT_TYPE, "schema_version": SCHEMA_V1},
            expected_artifact_type=CLUSTER_MODEL_ARTIFACT_TYPE,
            allow_missing_schema=False,
        )


def test_validate_bundle_schema_rejects_missing_schema_when_required():
    with pytest.raises(SchemaError, match="SCHEMA_MISSING"):
        validate_bundle_schema(
            {"artifact_type": CLUSTER_MODEL_ARTIFACT_TYPE},
            expected_artifact_type=CLUSTER_MODEL_ARTIFACT_TYPE,
            allow_missing_schema=False,
        )


def test_validate_bundle_schema_rejects_future_schema():
    with pytest.raises(SchemaError, match="SCHEMA_UNSUPPORTED"):
        validate_bundle_schema(
            {"artifact_type": CLUSTER_MODEL_ARTIFACT_TYPE, "schema_version": 999},
            expected_artifact_type=CLUSTER_MODEL_ARTIFACT_TYPE,
            supported_schema_version=SCHEMA_V1,
            allow_missing_schema=False,
        )


def test_removed_duplicate_types_not_exported():
    """Удалённые dataclass/TypedDict дубли не должны существовать в модуле."""
    for name in (
        "ClusterModelBundleDataclassV1",
        "TrainModelBundleV1",
        "TrainModelBundleDataclassV1",
    ):
        assert not hasattr(artifact_contracts, name), (
            f"{name} не должен присутствовать в artifact_contracts (был удалён как дубль)"
        )


def test_cluster_bundle_v1_is_typeddict():
    """ClusterModelBundleV1 остаётся TypedDict с ожидаемыми полями."""
    from artifact_contracts import ClusterModelBundleV1
    bundle: ClusterModelBundleV1 = {
        "artifact_type": CLUSTER_MODEL_ARTIFACT_TYPE,
        "schema_version": SCHEMA_V1,
        "algo": "kmeans",
        "K": 5,
    }
    assert bundle["algo"] == "kmeans"
