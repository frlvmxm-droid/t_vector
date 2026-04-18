import io
import socket
import threading
import urllib.error as _urlerr
from datetime import datetime, timedelta, timezone
import tracemalloc

import pytest

from app_cluster_service import LLMClient
import app_cluster_service
from llm_client import is_safe_url


def _kwargs():
    return dict(
        provider="openai",
        model="gpt-4o-mini",
        api_key="token",
        system_prompt="s",
        user_prompt="u",
        max_tokens=8,
    )


def test_llm_client_maps_network_error(monkeypatch):
    def _raise_urlerror(*args, **kwargs):
        raise _urlerr.URLError("network down")

    monkeypatch.setattr("llm_client._resolve_host_ips", lambda _h: ["1.1.1.1"])
    monkeypatch.setattr(app_cluster_service._urlreq, "urlopen", _raise_urlerror)

    with pytest.raises(Exception) as ei:
        LLMClient.complete_text(**_kwargs())
    assert "LLM_NETWORK_ERROR" in str(ei.value)


def test_llm_client_maps_timeout_error(monkeypatch):
    def _raise_timeout(*args, **kwargs):
        raise socket.timeout("timed out")

    monkeypatch.setattr("llm_client._resolve_host_ips", lambda _h: ["1.1.1.1"])
    monkeypatch.setattr(app_cluster_service._urlreq, "urlopen", _raise_timeout)

    with pytest.raises(Exception) as ei:
        LLMClient.complete_text(**_kwargs())
    assert "LLM_TIMEOUT" in str(ei.value)


def test_llm_client_maps_invalid_json(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr("llm_client._resolve_host_ips", lambda _h: ["1.1.1.1"])
    monkeypatch.setattr(app_cluster_service._urlreq, "urlopen", lambda *a, **k: _Resp())

    with pytest.raises(Exception) as ei:
        LLMClient.complete_text(**_kwargs())
    assert "LLM_INVALID_JSON" in str(ei.value)


def test_ollama_uses_response_field_when_message_content_empty(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"message":{"role":"assistant","content":""},"response":"cluster title"}'

    monkeypatch.setattr(app_cluster_service._urlreq, "urlopen", lambda *a, **k: _Resp())

    out = LLMClient.complete_text(
        provider="ollama",
        model="qwen3:30b",
        api_key="",
        system_prompt="s",
        user_prompt="u",
        max_tokens=8,
    )
    assert out == "cluster title"


def test_ollama_uses_choices_text_fallback(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"message":{"content":""},"choices":[{"text":"reason summary"}]}'

    monkeypatch.setattr(app_cluster_service._urlreq, "urlopen", lambda *a, **k: _Resp())

    out = LLMClient.complete_text(
        provider="ollama",
        model="qwen3:30b",
        api_key="",
        system_prompt="s",
        user_prompt="u",
        max_tokens=16,
    )
    assert out == "reason summary"


def test_llm_client_maps_http_error(monkeypatch):
    def _raise_http(*args, **kwargs):
        raise _urlerr.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"bad key"}'),
        )

    monkeypatch.setattr("llm_client._resolve_host_ips", lambda _h: ["1.1.1.1"])
    monkeypatch.setattr(app_cluster_service._urlreq, "urlopen", _raise_http)

    with pytest.raises(Exception) as ei:
        LLMClient.complete_text(**_kwargs())
    assert "LLM_HTTP_ERROR" in str(ei.value)


def test_llm_client_retries_rate_limit_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def _urlopen(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _urlerr.HTTPError(
                url="https://api.openai.com/v1/chat/completions",
                code=429,
                msg="rate limit",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"rate"}'),
            )
        return _Resp()

    monkeypatch.setattr("llm_client._resolve_host_ips", lambda _h: ["1.1.1.1"])
    monkeypatch.setattr(app_cluster_service._urlreq, "urlopen", _urlopen)
    monkeypatch.setattr(app_cluster_service.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(app_cluster_service.random, "uniform", lambda *_a, **_k: 0.0)

    out = LLMClient.complete_text(**_kwargs(), max_retries=1, backoff_base_sec=0.0, backoff_jitter_sec=0.0)
    assert out == "ok"
    assert calls["n"] == 2


def test_llm_client_cache_key_is_hashed():
    key = LLMClient._cache_key("openai", "gpt-4o-mini", "sys", "user", 16)
    assert len(key) == 64
    assert all(ch in "0123456789abcdef" for ch in key)


def test_llm_snapshot_decrypt_strict(monkeypatch):
    from llm_key_store import LLMSnapshotDecryptError

    def _raise(_value: str, **_kwargs):
        raise LLMSnapshotDecryptError("LLM_SNAPSHOT_DECRYPT_FAILED")

    monkeypatch.setattr("llm_client.decrypt_api_key_from_snapshot", _raise)
    with pytest.raises(Exception) as ei:
        LLMClient.complete_text(
            provider="openai",
            model="gpt-4o-mini",
            api_key="not-a-valid-token",
            system_prompt="s",
            user_prompt="u",
            max_tokens=8,
        )
    assert "LLM_SNAPSHOT_DECRYPT_FAILED" in str(ei.value)


def test_llm_cache_lru_is_bounded(monkeypatch):
    monkeypatch.setattr("llm_client._CACHE_MAX_ENTRIES", 2)
    LLMClient.reset_cache()
    LLMClient._cache_put("k1", "v1")
    LLMClient._cache_put("k2", "v2")
    LLMClient._cache_put("k3", "v3")
    assert len(LLMClient._response_cache) == 2
    assert "k1" not in LLMClient._response_cache
    assert LLMClient._cache_get("k2") == "v2"
    assert LLMClient._cache_get("missing") is None
    stats = LLMClient.cache_stats()
    assert stats["eviction"] >= 1
    assert stats["hit"] >= 1
    assert stats["miss"] >= 1


def test_llm_cache_stress_does_not_grow_unbounded(monkeypatch):
    monkeypatch.setattr("llm_client._CACHE_MAX_ENTRIES", 32)
    LLMClient.reset_cache()
    for i in range(500):
        LLMClient._cache_put(f"k{i}", f"v{i}")
    stats = LLMClient.cache_stats()
    assert stats["size"] <= 32
    assert stats["cache_size"] <= 32
    assert stats["eviction"] >= 468


def test_llm_cache_ttl_expiry_counts_as_miss():
    LLMClient.reset_cache()
    LLMClient._response_cache["k"] = {
        "value": "v",
        "expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
    }
    assert LLMClient._cache_get("k") is None
    stats = LLMClient.cache_stats()
    assert stats["miss"] >= 1


def test_llm_cache_memory_soak_bounded(monkeypatch):
    monkeypatch.setattr("llm_client._CACHE_MAX_ENTRIES", 64)
    LLMClient.reset_cache()
    tracemalloc.start()
    for i in range(5000):
        LLMClient._cache_put(f"k{i}", f"value-{i}" * 4)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stats = LLMClient.cache_stats()
    assert stats["size"] <= 64
    # мягкий guardrail для unit-среды; проверяет отсутствие "взрыва" памяти
    assert peak < 5 * 1024 * 1024


def test_llm_cache_thread_safety_deterministic(monkeypatch):
    monkeypatch.setattr("llm_client._CACHE_MAX_ENTRIES", 16)
    LLMClient.reset_cache()
    rounds = 100
    workers = 8

    def _worker(idx: int):
        for i in range(rounds):
            key = f"k-{idx}-{i % 16}"
            LLMClient._cache_put(key, f"v-{i}")
            assert LLMClient._cache_get(key) is not None
            assert LLMClient._cache_get(f"miss-{idx}-{i}") is None

    ts = [threading.Thread(target=_worker, args=(w,)) for w in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    stats = LLMClient.cache_stats()
    assert stats["cache_size"] <= 16
    assert stats["hit"] == workers * rounds
    assert stats["miss"] == workers * rounds


def test_is_safe_url_respects_allowlist(monkeypatch):
    monkeypatch.setattr("llm_client._resolve_host_ips", lambda _h: ["8.8.8.8"])
    assert is_safe_url(
        "https://api.openai.com/v1/chat/completions",
        allowed_hosts={"api.openai.com"},
    )
    assert not is_safe_url(
        "https://example.com",
        allowed_hosts={"api.openai.com"},
    )


def test_llm_cache_periodic_cleanup(monkeypatch):
    monkeypatch.setattr("llm_client._CACHE_MAX_ENTRIES", 64)
    monkeypatch.setattr("llm_client._CACHE_CLEANUP_EVERY", 3)
    LLMClient.reset_cache()
    old = datetime.now(timezone.utc) - timedelta(seconds=5)
    LLMClient._response_cache["stale-1"] = {"value": "x", "expires_at": old}
    LLMClient._response_cache["stale-2"] = {"value": "x", "expires_at": old}
    LLMClient._cache_put("k1", "v1")
    LLMClient._cache_put("k2", "v2")
    # cleanup not yet triggered
    assert "stale-1" in LLMClient._response_cache
    LLMClient._cache_put("k3", "v3")
    # third insertion triggers periodic cleanup
    assert "stale-1" not in LLMClient._response_cache
