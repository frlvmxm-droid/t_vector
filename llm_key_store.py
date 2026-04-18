"""Утилиты безопасного хранения/чтения API-ключей LLM."""
from __future__ import annotations

import os
from typing import Literal

from app_logger import get_logger

_log = get_logger(__name__)


class LLMSnapshotDecryptError(RuntimeError):
    """Ошибка дешифрования API-ключа из snapshot."""


def _snapshot_cipher():
    """Возвращает шифратор для снапшотов API-ключей (если настроен)."""
    raw = os.getenv("LLM_SNAPSHOT_KEY", "").strip()
    if not raw:
        return None
    try:
        fernet = __import__("cryptography.fernet", fromlist=["Fernet"]).Fernet
        return fernet(raw.encode("utf-8"))
    except Exception:
        _log.warning("invalid LLM_SNAPSHOT_KEY; snapshot encryption disabled")
        return None


def encrypt_api_key_for_snapshot(api_key: str) -> str:
    value = (api_key or "").strip()
    if not value:
        return ""
    cipher = _snapshot_cipher()
    if cipher is None:
        return value
    return cipher.encrypt(value.encode("utf-8")).decode("utf-8")


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _is_production_profile() -> bool:
    profile = (
        os.getenv("RUNTIME_PROFILE", "")
        or os.getenv("APP_ENV", "")
        or os.getenv("ENVIRONMENT", "")
    ).strip().lower()
    return profile in {"prod", "production", "live"}


def _resolve_decrypt_mode(explicit_mode: str | None = None) -> Literal["strict", "fail_closed", "legacy"]:
    mode = (explicit_mode or "").strip().lower()
    if mode in {"strict", "fail_closed", "legacy"}:
        return mode  # type: ignore[return-value]
    if _is_production_profile():
        return "strict"
    if (
        _is_truthy_env("STRICT_SECRET_DECRYPT")
        or _is_truthy_env("strict_secret_decrypt")
        or _is_truthy_env("LLM_SNAPSHOT_DECRYPT_STRICT")
    ):
        return "strict"
    if _is_truthy_env("LLM_SNAPSHOT_DECRYPT_FAIL_CLOSED"):
        return "fail_closed"
    return "fail_closed"


def decrypt_policy_matrix() -> dict[str, dict[str, str]]:
    """Явная runtime-матрица поведения дешифрования ключа snapshot."""
    return {
        "strict": {
            "on_decrypt_error": "raise",
            "fail_closed": "yes",
            "error_code": "configurable via strict_error_code",
            "compatibility": "low",
        },
        "fail_closed": {
            "on_decrypt_error": "empty_string",
            "fail_closed": "yes",
            "error_code": "none",
            "compatibility": "medium",
        },
        "legacy": {
            "on_decrypt_error": "return_input_as_is",
            "fail_closed": "no",
            "error_code": "none",
            "compatibility": "high",
        },
    }


def decrypt_api_key_from_snapshot(
    value: str,
    *,
    decrypt_mode: Literal["strict", "fail_closed", "legacy"] | None = None,
    strict_error_code: str = "LLM_SNAPSHOT_DECRYPT_FAILED",
) -> str:
    """Дешифрует ключ из snapshot.

    decrypt_mode:
    - "strict": fail-closed + LLMSnapshotDecryptError(error_code)
    - "fail_closed": fail-closed с возвратом пустой строки
    - "legacy": fail-open (обратная совместимость, вернуть исходное значение)
    Если decrypt_mode не задан — используется env-конфиг.
    В production-профиле secure-default: strict.
    Default (no profile set): fail_closed — повреждённый ключ → пустая строка вместо plaintext.
    """
    text = (value or "").strip()
    if not text:
        return ""
    cipher = _snapshot_cipher()
    if cipher is None:
        return text
    try:
        return cipher.decrypt(text.encode("utf-8")).decode("utf-8")
    except Exception as ex:
        mode = _resolve_decrypt_mode(decrypt_mode)
        _log.warning("unable to decrypt llm api key from snapshot")
        if mode == "strict":
            raise LLMSnapshotDecryptError(strict_error_code) from ex
        if mode == "fail_closed":
            return ""
        _log.warning("legacy decrypt fallback (fail-open) is active; use strict in production")
        return text


def resolve_api_key(provider: str, user_provided_key: str) -> str:
    p = (provider or "").strip().upper()
    if p:
        env_value = os.getenv(f"LLM_API_KEY_{p}", "").strip()
        if env_value:
            return env_value
    return (user_provided_key or "").strip()
