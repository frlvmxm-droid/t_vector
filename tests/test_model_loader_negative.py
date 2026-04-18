from pathlib import Path
import hashlib

import joblib
import pytest

from model_loader import (
    TrustStore,
    load_model_artifact,
    file_sha256,
    get_trusted_model_hash,
    ensure_trusted_model_path,
)
from exceptions import ModelLoadError


def test_load_model_artifact_rejects_non_dict_bundle(tmp_path: Path):
    p = tmp_path / "bad.joblib"
    joblib.dump(["not", "dict"], p)
    with pytest.raises(ModelLoadError):
        load_model_artifact(str(p))


def test_load_model_artifact_rejects_bad_schema_type(tmp_path: Path):
    p = tmp_path / "bad_schema.joblib"
    joblib.dump({"schema_version": "1", "pipeline": object()}, p)
    with pytest.raises(ModelLoadError):
        load_model_artifact(str(p))


def test_load_model_artifact_rejects_missing_required_key(tmp_path: Path):
    p = tmp_path / "missing.joblib"
    joblib.dump({"schema_version": 1, "artifact_type": "cluster_model_bundle"}, p)
    with pytest.raises(ModelLoadError):
        load_model_artifact(
            str(p),
            expected_artifact_types=("cluster_model_bundle",),
            required_keys=("vectorizer", "K"),
            allow_missing_schema=False,
        )


def test_load_model_artifact_rejects_unsupported_future_schema(tmp_path: Path):
    p = tmp_path / "future.joblib"
    joblib.dump({"schema_version": 999, "pipeline": object()}, p)
    with pytest.raises(ModelLoadError):
        load_model_artifact(str(p), supported_schema_version=1)


def test_load_model_artifact_rejects_wrong_artifact_type(tmp_path: Path):
    p = tmp_path / "wrong_type.joblib"
    joblib.dump({"schema_version": 1, "artifact_type": "unknown"}, p)
    with pytest.raises(ModelLoadError):
        load_model_artifact(str(p), expected_artifact_types=("cluster_model_bundle",))


def test_load_model_artifact_rejects_unexpected_extension(tmp_path: Path):
    p = tmp_path / "model.pkl"
    p.write_bytes(b"dummy")
    with pytest.raises(ModelLoadError):
        load_model_artifact(str(p))


def test_load_model_artifact_rejects_corrupted_bundle(tmp_path: Path):
    p = tmp_path / "corrupt.joblib"
    p.write_bytes(b"not-a-joblib-payload")
    with pytest.raises(ModelLoadError):
        load_model_artifact(str(p))


def test_load_model_artifact_rejects_float_schema_type(tmp_path: Path):
    p = tmp_path / "bad_schema_float.joblib"
    joblib.dump({"schema_version": 1.5, "artifact_type": "cluster_model_bundle"}, p)
    with pytest.raises(ModelLoadError):
        load_model_artifact(str(p), expected_artifact_types=("cluster_model_bundle",))


def test_load_model_artifact_rejects_wrong_required_key_type(tmp_path: Path):
    p = tmp_path / "bad_key_type.joblib"
    joblib.dump(
        {"schema_version": 1, "artifact_type": "cluster_model_bundle", "K": "10"},
        p,
    )
    with pytest.raises(ModelLoadError):
        load_model_artifact(
            str(p),
            expected_artifact_types=("cluster_model_bundle",),
            required_keys=("K",),
            required_key_types={"K": int},
        )


def test_load_model_artifact_rejects_sha256_mismatch(tmp_path: Path):
    p = tmp_path / "bundle.joblib"
    joblib.dump({"schema_version": 1, "pipeline": object()}, p)

    with pytest.raises(ModelLoadError, match=r"SHA256_MISMATCH"):
        load_model_artifact(str(p), expected_sha256="0" * 64)


def test_load_model_artifact_accepts_matching_sha256(tmp_path: Path):
    p = tmp_path / "bundle_ok.joblib"
    joblib.dump({"schema_version": 1, "pipeline": object()}, p)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()

    loaded = load_model_artifact(str(p), expected_sha256=expected)

    assert loaded["schema_version"] == 1


def test_file_sha256_streaming_matches_hashlib(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"abc" * (1024 * 128))
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert file_sha256(p) == expected
    assert file_sha256(p) == expected


def test_get_trusted_model_hash_reads_from_app_state():
    store = TrustStore()
    store.add_trusted("x.joblib", "abc")
    assert get_trusted_model_hash(store, "x.joblib") == "abc"


def test_load_model_artifact_uses_precomputed_sha_without_rehash(tmp_path: Path, monkeypatch):
    p = tmp_path / "bundle_prehashed.joblib"
    joblib.dump({"schema_version": 1, "pipeline": object()}, p)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()

    def _forbidden_hash(_path):
        raise AssertionError("file_sha256 should not be called when precomputed_sha256 is provided")

    monkeypatch.setattr("model_loader.file_sha256", _forbidden_hash)
    loaded = load_model_artifact(
        str(p),
        precomputed_sha256=expected,
        expected_sha256=expected,
    )
    assert loaded["schema_version"] == 1


def test_ensure_trusted_model_path_uses_canonical_key(tmp_path: Path, monkeypatch):
    p = tmp_path / "model.joblib"
    p.write_bytes(b"x")

    monkeypatch.setattr("model_loader.file_sha256", lambda _p: "abc")
    store = TrustStore()

    alias_1 = str(p)
    alias_2 = str(p.parent / "." / p.name)
    assert ensure_trusted_model_path(store, alias_1, confirm_fn=lambda _: True) is True
    assert ensure_trusted_model_path(store, alias_2, confirm_fn=lambda _: True) is True
    assert len(store._trusted_paths) == 1
    assert get_trusted_model_hash(store, alias_1) == "abc"
    assert get_trusted_model_hash(store, alias_2) == "abc"


def test_trust_store_revoke(tmp_path: Path, monkeypatch):
    """TrustStore.revoke удаляет путь из доверенных и предотвращает повторное добавление."""
    p = tmp_path / "model.joblib"
    p.write_bytes(b"x")

    monkeypatch.setattr("model_loader.file_sha256", lambda _path: "abc")
    store = TrustStore()
    store.add_trusted(str(p), "abc")
    assert store.is_trusted(str(p))

    store.revoke(str(p))
    assert not store.is_trusted(str(p))
    assert store.is_revoked(str(p))

    # Отозванный путь не может быть подтверждён повторно
    result = ensure_trusted_model_path(store, str(p), confirm_fn=lambda _: True)
    assert result is False


def test_trust_store_hash_changed_revokes(tmp_path: Path, monkeypatch):
    """Если файл изменился на диске — доверие отзывается автоматически."""
    p = tmp_path / "model.joblib"
    p.write_bytes(b"x")

    hashes = iter(["hash_v1", "hash_v2"])
    monkeypatch.setattr("model_loader.file_sha256", lambda _path: next(hashes))
    store = TrustStore()

    # Первый вызов — подтверждаем
    assert ensure_trusted_model_path(store, str(p), confirm_fn=lambda _: True) is True
    assert store.get_hash(str(p)) == "hash_v1"

    # Второй вызов — файл изменился (hash_v2 ≠ hash_v1), доверие отзывается, снова спрашиваем
    assert ensure_trusted_model_path(store, str(p), confirm_fn=lambda _: True) is True
    assert store.get_hash(str(p)) == "hash_v2"


def test_trust_store_confirm_false_denies(tmp_path: Path, monkeypatch):
    """Если confirm_fn вернул False — путь не добавляется в доверенные."""
    p = tmp_path / "model.joblib"
    p.write_bytes(b"x")

    monkeypatch.setattr("model_loader.file_sha256", lambda _path: "abc")
    store = TrustStore()
    result = ensure_trusted_model_path(store, str(p), confirm_fn=lambda _: False)
    assert result is False
    assert not store.is_trusted(str(p))


def test_load_model_artifact_require_trusted_rejects_untrusted(tmp_path: Path):
    p = tmp_path / "bundle_untrusted.joblib"
    joblib.dump({"schema_version": 1, "pipeline": object()}, p)
    with pytest.raises(ModelLoadError, match="UNTRUSTED_MODEL_PATH"):
        load_model_artifact(
            str(p),
            require_trusted=True,
            trusted_paths=[],
        )


def test_load_model_artifact_require_trusted_accepts_trusted_alias(tmp_path: Path):
    p = tmp_path / "bundle_trusted.joblib"
    joblib.dump({"schema_version": 1, "pipeline": object()}, p)
    alias = str(p.parent / "." / p.name)
    loaded = load_model_artifact(
        alias,
        require_trusted=True,
        trusted_paths=[str(p)],
    )
    assert loaded["schema_version"] == 1
