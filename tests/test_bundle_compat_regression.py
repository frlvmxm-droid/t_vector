from pathlib import Path

import joblib
import pytest

from model_loader import load_model_artifact
from exceptions import SchemaError


def _dump(path: Path, bundle: dict) -> str:
    joblib.dump(bundle, path)
    return str(path)


def test_bundle_v0_allowed_if_missing_schema(tmp_path: Path):
    p = _dump(tmp_path / "v0.joblib", {"artifact_type": "train_model_bundle", "pipeline": object()})
    out = load_model_artifact(
        p,
        expected_artifact_types=("train_model_bundle",),
        required_keys=("pipeline",),
        allow_missing_schema=True,
    )
    assert "pipeline" in out


def test_bundle_v1_ok(tmp_path: Path):
    p = _dump(
        tmp_path / "v1.joblib",
        {"artifact_type": "train_model_bundle", "schema_version": 1, "pipeline": object()},
    )
    out = load_model_artifact(
        p,
        expected_artifact_types=("train_model_bundle",),
        required_keys=("pipeline",),
        allow_missing_schema=False,
    )
    assert out["schema_version"] == 1


def test_bundle_future_schema_rejected(tmp_path: Path):
    p = _dump(
        tmp_path / "v99.joblib",
        {"artifact_type": "train_model_bundle", "schema_version": 99, "pipeline": object()},
    )
    with pytest.raises(SchemaError, match="SCHEMA_UNSUPPORTED"):
        load_model_artifact(
            p,
            expected_artifact_types=("train_model_bundle",),
            required_keys=("pipeline",),
            allow_missing_schema=False,
            supported_schema_version=1,
        )
