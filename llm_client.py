# -*- coding: utf-8 -*-
"""Сетевой клиент LLM-провайдеров с retry/circuit-breaker/cache."""
from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import threading
import time
import urllib.error as _urlerr
import urllib.request as _urlreq
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from json import JSONDecodeError
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from app_logger import get_logger
from exceptions import FeatureBuildError
from llm_key_store import (
    LLMSnapshotDecryptError,
    decrypt_api_key_from_snapshot,
    resolve_api_key,
)

_log = get_logger(__name__)

_DEFAULT_EXTERNAL_TIMEOUT_SEC = 30.0
_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_RESET_SEC = 90
_CACHE_TTL_SEC = 600
_CACHE_MAX_ENTRIES = max(32, int(os.getenv("LLM_CACHE_MAX_ENTRIES", "256")))
_CACHE_CLEANUP_EVERY = max(1, int(os.getenv("LLM_CACHE_CLEANUP_EVERY", "32")))

_PROVIDER_HOST_ALLOWLIST: Dict[str, set[str]] = {
    "openai": {"api.openai.com"},
    "anthropic": {"api.anthropic.com"},
    "qwen": {"dashscope.aliyuncs.com"},
    "gigachat": {"gigachat.devices.sberbank.ru"},
    "ollama": {"127.0.0.1", "localhost"},
}


def _resolve_host_ips(hostname: str) -> list[str]:
    ips: list[str] = []
    for addr_info in socket.getaddrinfo(hostname, None):
        ip = addr_info[4][0]
        if ip not in ips:
            ips.append(ip)
    return ips


def _all_ips_safe(ips: list[str], *, allow_private_hosts: bool) -> bool:
    for ip in ips:
        ip_obj = ip_address(ip)
        if not allow_private_hosts and (
            ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local
        ):
            return False
    return True


def is_safe_url(
    url: str,
    *,
    allow_private_hosts: bool = False,
    allowed_hosts: set[str] | None = None,
    logger: Any = None,
) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = (parsed.hostname or "").strip()
    if not hostname:
        return False
    if allowed_hosts is not None and hostname not in allowed_hosts:
        return False
    try:
        ips = _resolve_host_ips(hostname)
        if logger is not None:
            logger.info("ssrf-check host=%s ips=%s", hostname, ",".join(ips))
        if not _all_ips_safe(ips, allow_private_hosts=allow_private_hosts):
            return False
    except Exception:
        return False
    return True


class LLMClient:
    """Сетевой клиент для LLM-провайдеров, используемых в кластеризации."""

    _circuit_state: Dict[str, Dict[str, Any]] = {}
    _response_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    _cache_stats: Dict[str, int] = {"hit": 0, "miss": 0, "eviction": 0}
    _cache_cleanup_ctr: int = 0
    _state_lock = threading.RLock()
    _stats_lock = threading.RLock()

    @classmethod
    def _stat_inc(cls, key: str, delta: int = 1) -> None:
        with cls._stats_lock:
            cls._cache_stats[key] = int(cls._cache_stats.get(key, 0)) + int(delta)

    @classmethod
    def _cache_key(
        cls,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        raw = "|".join(
            [
                (provider or "").strip().lower(),
                (model or "").strip(),
                str(max_tokens),
                (system_prompt or "").strip(),
                (user_prompt or "").strip(),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _cache_get(cls, key: str) -> str | None:
        with cls._state_lock:
            item = cls._response_cache.get(key)
            if not item:
                cls._stat_inc("miss")
                return None
            if datetime.now(timezone.utc) > item["expires_at"]:
                cls._response_cache.pop(key, None)
                cls._stat_inc("miss")
                return None
            cls._response_cache.move_to_end(key)
            cls._stat_inc("hit")
            return str(item.get("value", ""))

    @classmethod
    def _cache_put(cls, key: str, value: str, ttl_sec: int = _CACHE_TTL_SEC) -> None:
        with cls._state_lock:
            now = datetime.now(timezone.utc)
            cls._cache_cleanup_ctr += 1
            if cls._cache_cleanup_ctr % _CACHE_CLEANUP_EVERY == 0:
                expired_keys = [k for k, v in cls._response_cache.items() if now > v.get("expires_at", now)]
                for k in expired_keys:
                    cls._response_cache.pop(k, None)
            cls._response_cache[key] = {
                "value": value,
                "expires_at": now + timedelta(seconds=max(1, int(ttl_sec))),
            }
            cls._response_cache.move_to_end(key)
            while len(cls._response_cache) > _CACHE_MAX_ENTRIES:
                cls._response_cache.popitem(last=False)
                cls._stat_inc("eviction")

    @classmethod
    def cache_stats(cls) -> Dict[str, int]:
        with cls._state_lock:
            cache_size = len(cls._response_cache)
        with cls._stats_lock:
            hit = int(cls._cache_stats.get("hit", 0))
            miss = int(cls._cache_stats.get("miss", 0))
            eviction = int(cls._cache_stats.get("eviction", 0))
        return {
            "size": cache_size,
            "cache_size": cache_size,
            "hit": hit,
            "miss": miss,
            "eviction": eviction,
        }

    @classmethod
    def reset_cache(cls) -> None:
        with cls._state_lock:
            cls._response_cache.clear()
            cls._cache_cleanup_ctr = 0
        with cls._stats_lock:
            cls._cache_stats = {"hit": 0, "miss": 0, "eviction": 0}

    @classmethod
    def _is_circuit_open(cls, provider: str) -> bool:
        p = (provider or "").strip().lower()
        with cls._state_lock:
            state = cls._circuit_state.get(p) or {}
            opened_until = state.get("opened_until")
            return bool(opened_until and datetime.now(timezone.utc) < opened_until)

    @classmethod
    def _mark_success(cls, provider: str) -> None:
        p = (provider or "").strip().lower()
        with cls._state_lock:
            cls._circuit_state[p] = {"failures": 0, "opened_until": None}

    @classmethod
    def _mark_failure(cls, provider: str) -> None:
        p = (provider or "").strip().lower()
        with cls._state_lock:
            state = cls._circuit_state.get(p) or {"failures": 0, "opened_until": None}
            state["failures"] = int(state.get("failures", 0)) + 1
            if state["failures"] >= _CIRCUIT_BREAKER_THRESHOLD:
                state["opened_until"] = datetime.now(timezone.utc) + timedelta(seconds=_CIRCUIT_BREAKER_RESET_SEC)
            cls._circuit_state[p] = state

    @staticmethod
    def _parse_retry_after(headers: Any) -> Optional[float]:
        """Parse RFC 7231 Retry-After header: delta-seconds or HTTP-date.

        Returns the wait in seconds, or None if absent/unparseable.
        Clamped to non-negative values; capped upstream to avoid
        adversarial huge values.
        """
        if headers is None:
            return None
        raw = None
        try:
            raw = headers.get("Retry-After") or headers.get("retry-after")
        except AttributeError:
            return None
        if raw is None:
            return None
        raw_s = str(raw).strip()
        if not raw_s:
            return None
        # Case 1: integer delta-seconds
        try:
            secs = float(raw_s)
            return max(0.0, secs)
        except ValueError:
            pass
        # Case 2: HTTP-date (RFC 7231 IMF-fixdate)
        try:
            from email.utils import parsedate_to_datetime as _pdt
            dt = _pdt(raw_s)
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (dt - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, delta)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Приватные методы-помощники для complete_text
    # ------------------------------------------------------------------

    @staticmethod
    def _decrypt_and_resolve_key(provider: str, api_key: str) -> str:
        """Дешифрует snapshot-ключ и резолвит его из env (если настроен)."""
        try:
            decrypt_mode = os.getenv("LLM_SNAPSHOT_DECRYPT_MODE", "").strip().lower() or None
            decrypted = decrypt_api_key_from_snapshot(
                api_key,
                decrypt_mode=decrypt_mode,
                strict_error_code="LLM_SNAPSHOT_DECRYPT_FAILED",
            )
        except LLMSnapshotDecryptError as ex:
            raise FeatureBuildError(
                "[error_code=LLM_SNAPSHOT_DECRYPT_FAILED] stage=llm.request hint=Не удалось расшифровать API-ключ из snapshot."
            ) from ex
        return resolve_api_key(provider, decrypted)

    @staticmethod
    def _build_provider_request(
        p: str,
        model: str,
        resolved_api_key: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        """Строит URL, заголовки и payload для конкретного провайдера.

        Returns:
            (url, headers, payload)
        """
        if p == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers: Dict[str, str] = {
                "x-api-key": resolved_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload: Dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        elif p == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "authorization": f"Bearer {resolved_api_key}",
                "content-type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        elif p == "qwen":
            url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            headers = {
                "authorization": f"Bearer {resolved_api_key}",
                "content-type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        elif p == "gigachat":
            url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
            headers = {
                "authorization": f"Bearer {resolved_api_key}",
                "content-type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        elif p == "ollama":
            url = "http://127.0.0.1:11434/api/chat"
            headers = {"content-type": "application/json"}
            payload = {
                "model": model,
                "stream": False,
                "options": {"num_predict": max_tokens},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        else:
            raise FeatureBuildError(f"Неподдерживаемый LLM-провайдер: {p}")

        return url, headers, payload

    @staticmethod
    def _validate_url_and_dns(url: str, p: str, resolved_api_key: str) -> None:
        """Проверяет URL по SSRF-allowlist и выполняет TOCTOU DNS re-resolve."""
        if p != "ollama" and not resolved_api_key:
            raise FeatureBuildError(f"Для провайдера {p} требуется API-ключ.")
        allowed_hosts = _PROVIDER_HOST_ALLOWLIST.get(p)
        if not is_safe_url(
            url,
            allow_private_hosts=(p == "ollama"),
            allowed_hosts=allowed_hosts,
            logger=_log,
        ):
            raise FeatureBuildError(
                f"[error_code=LLM_UNSAFE_URL] stage=llm.request hint=URL провайдера не прошёл проверку безопасности: {url}"
            )
        # TOCTOU hardening: re-resolve immediately before connection attempt.
        host = (urlparse(url).hostname or "").strip()
        try:
            ips_now = _resolve_host_ips(host)
        except Exception as ex:
            raise FeatureBuildError(
                f"[error_code=LLM_DNS_RESOLVE_FAILED] stage=llm.request hint=Не удалось разрешить host {host}: {ex}"
            ) from ex
        if not _all_ips_safe(ips_now, allow_private_hosts=(p == "ollama")):
            raise FeatureBuildError(
                f"[error_code=LLM_UNSAFE_DNS] stage=llm.request hint=Повторная DNS-проверка отклонила host {host}."
            )

    @staticmethod
    def _execute_with_retries(
        req: _urlreq.Request,
        provider: str,
        *,
        max_retries: int,
        backoff_base_sec: float,
        backoff_jitter_sec: float,
        timeout_sec: float,
    ) -> str:
        """Выполняет HTTP-запрос с retry/backoff. Возвращает тело ответа."""
        retryable_http = {429, 500, 502, 503, 504}
        attempts = max(1, int(max_retries) + 1)
        body = ""
        for attempt_idx in range(attempts):
            try:
                with _urlreq.urlopen(req, timeout=float(timeout_sec)) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                return body
            except _urlerr.HTTPError as e:
                _body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
                _is_retryable = e.code in retryable_http and attempt_idx < attempts - 1
                if _is_retryable:
                    # Prefer server-advised backoff (RFC 7231 §7.1.3) when present:
                    # Retry-After may be either delta-seconds or HTTP-date.
                    _retry_after = LLMClient._parse_retry_after(
                        getattr(e, "headers", None)
                    )
                    if _retry_after is not None:
                        _sleep = min(float(_retry_after), 60.0) + random.uniform(
                            0.0, backoff_jitter_sec
                        )
                        _log.warning(
                            "llm retry-after honored: provider=%s code=%s attempt=%s/%s retry_after=%.2fs",
                            provider, e.code, attempt_idx + 1, attempts, _sleep,
                        )
                    else:
                        _sleep = backoff_base_sec * (2 ** attempt_idx) + random.uniform(0.0, backoff_jitter_sec)
                        _log.warning(
                            "llm retryable http error: provider=%s code=%s attempt=%s/%s sleep=%.2fs",
                            provider, e.code, attempt_idx + 1, attempts, _sleep,
                        )
                    time.sleep(_sleep)
                    continue
                if e.code in retryable_http:
                    LLMClient._mark_failure(provider)
                if e.code == 429:
                    raise FeatureBuildError(
                        f"[error_code=LLM_RATE_LIMIT] stage=llm.request hint=Превышен лимит запросов; повторите позже или снизьте частоту. | HTTP {e.code}: {_body[:500]}"
                    )
                raise FeatureBuildError(
                    f"[error_code=LLM_HTTP_ERROR] stage=llm.request hint=Проверьте API-ключ/модель и права доступа. | HTTP {e.code}: {_body[:500]}"
                )
            except _urlerr.URLError as e:
                _reason = getattr(e, "reason", e)
                if attempt_idx < attempts - 1:
                    _sleep = backoff_base_sec * (2 ** attempt_idx) + random.uniform(0.0, backoff_jitter_sec)
                    _log.warning(
                        "llm retryable network error: provider=%s reason=%s attempt=%s/%s sleep=%.2fs",
                        provider, type(_reason).__name__, attempt_idx + 1, attempts, _sleep,
                    )
                    time.sleep(_sleep)
                    continue
                LLMClient._mark_failure(provider)
                raise FeatureBuildError(
                    f"[error_code=LLM_NETWORK_ERROR] stage=llm.request hint=Проверьте сеть, DNS/прокси и URL провайдера. | {type(_reason).__name__}: {_reason}"
                )
            except (TimeoutError, socket.timeout) as e:
                if attempt_idx < attempts - 1:
                    _sleep = backoff_base_sec * (2 ** attempt_idx) + random.uniform(0.0, backoff_jitter_sec)
                    _log.warning(
                        "llm retryable timeout: provider=%s attempt=%s/%s sleep=%.2fs",
                        provider, attempt_idx + 1, attempts, _sleep,
                    )
                    time.sleep(_sleep)
                    continue
                LLMClient._mark_failure(provider)
                raise FeatureBuildError(
                    f"[error_code=LLM_TIMEOUT] stage=llm.request hint=Увеличьте timeout или уменьшите payload запроса. | {e}"
                )
        return body  # достижимо только при attempts=0, на практике не происходит

    @staticmethod
    def _parse_and_cache_response(p: str, provider: str, body: str, cache_key: str) -> str:
        """Парсит JSON-ответ провайдера, кеширует результат и возвращает текст."""
        try:
            data = json.loads(body)
        except JSONDecodeError as e:
            raise FeatureBuildError(
                f"[error_code=LLM_INVALID_JSON] stage=llm.parse hint=Проверьте совместимость провайдера и формат ответа API. | Некорректный JSON-ответ от {provider}: {e}"
            )

        if p == "anthropic":
            content = data.get("content") or []
            result = ""
            if content and isinstance(content[0], dict):
                result = str(content[0].get("text", "")).strip()
            LLMClient._mark_success(provider)
            LLMClient._cache_put(cache_key, result)
            return result
        if p == "ollama":
            message = data.get("message") or {}
            choices = data.get("choices") or []
            result = str(message.get("content", "")).strip()
            if not result:
                # Совместимость со старыми/альтернативными форматами Ollama.
                result = str(data.get("response", "")).strip()
            if not result and choices and isinstance(choices[0], dict):
                ch0 = choices[0]
                result = str((ch0.get("message") or {}).get("content", "")).strip()
                if not result:
                    result = str(ch0.get("text", "")).strip()
            LLMClient._mark_success(provider)
            LLMClient._cache_put(cache_key, result)
            return result
        choices = data.get("choices") or []
        result = ""
        if choices and isinstance(choices[0], dict):
            result = str((choices[0].get("message") or {}).get("content", "")).strip()
        LLMClient._mark_success(provider)
        LLMClient._cache_put(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    @staticmethod
    def complete_text(
        provider: str,
        model: str,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 128,
        timeout_sec: float = _DEFAULT_EXTERNAL_TIMEOUT_SEC,
        max_retries: int = 2,
        backoff_base_sec: float = 0.6,
        backoff_jitter_sec: float = 0.25,
    ) -> str:
        """Отправляет запрос к LLM-провайдеру и возвращает текст ответа."""
        p = (provider or "").strip().lower()

        resolved_api_key = LLMClient._decrypt_and_resolve_key(provider, api_key)

        cache_key = LLMClient._cache_key(provider, model, system_prompt, user_prompt, max_tokens)
        cached = LLMClient._cache_get(cache_key)
        if cached is not None:
            if LLMClient._is_circuit_open(provider):
                _log.warning("llm circuit open for provider=%s, returning cached response", provider)
            return cached
        if LLMClient._is_circuit_open(provider):
            raise FeatureBuildError(
                "[error_code=LLM_CIRCUIT_OPEN] stage=llm.request hint=Провайдер временно недоступен; повторите запрос позже."
            )

        url, headers, payload = LLMClient._build_provider_request(
            p, model, resolved_api_key, system_prompt, user_prompt, max_tokens
        )
        LLMClient._validate_url_and_dns(url, p, resolved_api_key)

        req = _urlreq.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        body = LLMClient._execute_with_retries(
            req,
            provider,
            max_retries=max_retries,
            backoff_base_sec=backoff_base_sec,
            backoff_jitter_sec=backoff_jitter_sec,
            timeout_sec=timeout_sec,
        )
        return LLMClient._parse_and_cache_response(p, provider, body, cache_key)
