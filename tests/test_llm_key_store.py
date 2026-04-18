import pytest

from llm_key_store import (
    decrypt_api_key_from_snapshot,
    LLMSnapshotDecryptError,
    decrypt_policy_matrix,
)


def test_decrypt_fail_closed_by_default(monkeypatch):
    # Default mode is now fail_closed: corrupt blob → empty string (not fail-open)
    class _Cipher:
        def decrypt(self, _payload):
            raise ValueError("bad token")

    monkeypatch.setattr("llm_key_store._snapshot_cipher", lambda: _Cipher())
    monkeypatch.delenv("LLM_SNAPSHOT_DECRYPT_STRICT", raising=False)
    monkeypatch.delenv("LLM_SNAPSHOT_DECRYPT_FAIL_CLOSED", raising=False)
    monkeypatch.delenv("RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert decrypt_api_key_from_snapshot("encrypted-blob") == ""


def test_decrypt_fail_closed_when_enabled(monkeypatch):
    class _Cipher:
        def decrypt(self, _payload):
            raise ValueError("bad token")

    monkeypatch.setattr("llm_key_store._snapshot_cipher", lambda: _Cipher())
    monkeypatch.setenv("LLM_SNAPSHOT_DECRYPT_FAIL_CLOSED", "1")
    assert decrypt_api_key_from_snapshot("encrypted-blob") == ""


def test_decrypt_strict_raises(monkeypatch):
    class _Cipher:
        def decrypt(self, _payload):
            raise ValueError("bad token")

    monkeypatch.setattr("llm_key_store._snapshot_cipher", lambda: _Cipher())
    monkeypatch.setenv("STRICT_SECRET_DECRYPT", "1")
    with pytest.raises(LLMSnapshotDecryptError):
        decrypt_api_key_from_snapshot("encrypted-blob")


def test_decrypt_explicit_mode_strict_with_error_code(monkeypatch):
    class _Cipher:
        def decrypt(self, _payload):
            raise ValueError("bad token")

    monkeypatch.setattr("llm_key_store._snapshot_cipher", lambda: _Cipher())
    with pytest.raises(LLMSnapshotDecryptError, match="MY_ERR_CODE"):
        decrypt_api_key_from_snapshot(
            "encrypted-blob",
            decrypt_mode="strict",
            strict_error_code="MY_ERR_CODE",
        )


def test_decrypt_explicit_mode_fail_closed(monkeypatch):
    class _Cipher:
        def decrypt(self, _payload):
            raise ValueError("bad token")

    monkeypatch.setattr("llm_key_store._snapshot_cipher", lambda: _Cipher())
    assert (
        decrypt_api_key_from_snapshot("encrypted-blob", decrypt_mode="fail_closed")
        == ""
    )


def test_decrypt_policy_matrix_has_expected_modes():
    matrix = decrypt_policy_matrix()
    assert set(matrix.keys()) == {"strict", "fail_closed", "legacy"}
    assert matrix["strict"]["fail_closed"] == "yes"


def test_decrypt_mode_explicit_has_priority_over_env(monkeypatch):
    class _Cipher:
        def decrypt(self, _payload):
            raise ValueError("bad token")

    monkeypatch.setattr("llm_key_store._snapshot_cipher", lambda: _Cipher())
    monkeypatch.setenv("LLM_SNAPSHOT_DECRYPT_STRICT", "1")
    # explicit legacy must override strict env
    assert decrypt_api_key_from_snapshot("encrypted-blob", decrypt_mode="legacy") == "encrypted-blob"


def test_decrypt_defaults_to_strict_in_production_profile(monkeypatch):
    class _Cipher:
        def decrypt(self, _payload):
            raise ValueError("bad token")

    monkeypatch.setattr("llm_key_store._snapshot_cipher", lambda: _Cipher())
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("LLM_SNAPSHOT_DECRYPT_STRICT", raising=False)
    monkeypatch.delenv("LLM_SNAPSHOT_DECRYPT_FAIL_CLOSED", raising=False)
    with pytest.raises(LLMSnapshotDecryptError):
        decrypt_api_key_from_snapshot("encrypted-blob")


def test_decrypt_legacy_mode_emits_warning(monkeypatch, caplog):
    class _Cipher:
        def decrypt(self, _payload):
            raise ValueError("bad token")

    monkeypatch.setattr("llm_key_store._snapshot_cipher", lambda: _Cipher())
    out = decrypt_api_key_from_snapshot("encrypted-blob", decrypt_mode="legacy")
    assert out == "encrypted-blob"
    assert "legacy decrypt fallback" in caplog.text
