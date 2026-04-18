import json
from pathlib import Path

import joblib
import pytest

from exceptions import ModelLoadError
from model_loader import load_model_artifact


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "bundle_specs"


def _materialize_bundle(spec_bundle: dict):
    bundle = dict(spec_bundle)
    if "pipeline" in bundle and isinstance(bundle["pipeline"], dict):
        # keep fixture serializable while satisfying required key contract
        bundle["pipeline"] = object()
    return bundle


@pytest.mark.parametrize(
    "spec_path",
    sorted(FIXTURES_DIR.glob("*.json")),
    ids=lambda p: p.stem,
)
def test_bundle_compat_from_fixture_specs(tmp_path: Path, spec_path: Path):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    bundle = _materialize_bundle(spec["bundle"])
    model_path = tmp_path / f"{spec['name']}.joblib"
    joblib.dump(bundle, model_path)

    kwargs = dict(
        expected_artifact_types=("train_model_bundle",),
        required_keys=("pipeline",),
        allow_missing_schema=bool(spec.get("allow_missing_schema", False)),
        supported_schema_version=1,
    )

    if spec.get("expect_ok", False):
        out = load_model_artifact(str(model_path), **kwargs)
        assert "pipeline" in out
    else:
        match = spec.get("error_match", "")
        with pytest.raises(ModelLoadError, match=match):
            load_model_artifact(str(model_path), **kwargs)
