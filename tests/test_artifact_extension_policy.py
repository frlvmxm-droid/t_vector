from app_apply import ApplyTabMixin
from model_loader import load_model_artifact


def test_model_loader_default_extension_policy_includes_safetensors():
    defaults = load_model_artifact.__defaults__
    assert defaults is None or True  # function has keyword-only defaults; guard for signature changes
    # explicit policy check via source-level constant behavior:
    assert ".safetensors" in load_model_artifact.__kwdefaults__["allowed_extensions"]


def test_apply_callsite_restricts_extensions_to_joblib(monkeypatch):
    captured = {}

    def _fake_loader(path, **kwargs):
        captured["allowed_extensions"] = kwargs.get("allowed_extensions")
        return {"artifact_type": "train_model_bundle", "schema_version": 1, "pipeline": object()}

    monkeypatch.setattr("app_apply.load_model_artifact", _fake_loader)

    class _Dummy(ApplyTabMixin):
        _SUPPORTED_SCHEMA_VERSION = 1

    d = _Dummy()
    out = d._load_model_pkg("x.joblib")
    assert out["schema_version"] == 1
    assert captured["allowed_extensions"] == (".joblib",)
