"""Тесты Wave 2: temperature в LLMClient и use_lemma в c-TF-IDF."""
from __future__ import annotations

import pytest

from llm_client import LLMClient


class TestTemperaturePayload:
    """Temperature должен корректно попадать в payload каждого провайдера."""

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "qwen", "gigachat"])
    def test_temperature_added_when_set(self, provider):
        _, _, payload = LLMClient._build_provider_request(
            provider, "model-x", "token", "sys", "user", 32, temperature=0.2,
        )
        assert payload.get("temperature") == pytest.approx(0.2)

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "qwen", "gigachat", "ollama"])
    def test_temperature_omitted_when_none(self, provider):
        _, _, payload = LLMClient._build_provider_request(
            provider, "model-x", "token", "sys", "user", 32, temperature=None,
        )
        if provider == "ollama":
            assert "temperature" not in payload.get("options", {})
        else:
            assert "temperature" not in payload

    def test_temperature_ollama_in_options(self):
        _, _, payload = LLMClient._build_provider_request(
            "ollama", "qwen3:8b", "", "sys", "user", 32, temperature=0.2,
        )
        assert payload["options"]["temperature"] == pytest.approx(0.2)


class TestTemperatureCacheKey:
    """Разная temperature должна давать разные кеш-ключи."""

    def test_none_and_value_differ(self):
        k_none = LLMClient._cache_key("openai", "m", "s", "u", 32, temperature=None)
        k_02 = LLMClient._cache_key("openai", "m", "s", "u", 32, temperature=0.2)
        assert k_none != k_02

    def test_two_values_differ(self):
        k1 = LLMClient._cache_key("openai", "m", "s", "u", 32, temperature=0.2)
        k2 = LLMClient._cache_key("openai", "m", "s", "u", 32, temperature=0.8)
        assert k1 != k2

    def test_same_temperature_same_key(self):
        k1 = LLMClient._cache_key("openai", "m", "s", "u", 32, temperature=0.2)
        k2 = LLMClient._cache_key("openai", "m", "s", "u", 32, temperature=0.2)
        assert k1 == k2


class TestCTfidfLemmaKwarg:
    """extract_cluster_keywords_ctfidf должен принимать use_lemma
    и не падать при любом значении (даже если pymorphy недоступен)."""

    def test_signature_accepts_use_lemma(self):
        numpy = pytest.importorskip("numpy")
        pytest.importorskip("sklearn")

        from ml_diagnostics import extract_cluster_keywords_ctfidf

        docs = [
            "оплата картой прошла успешно",
            "оплатил картой в магазине",
            "перевод на счёт не дошёл",
            "перевожу деньги по реквизитам",
        ]
        labels = numpy.array([0, 0, 1, 1])
        # Обе ветки должны работать без исключений — pymorphy может быть
        # как установлен, так и нет; смысл теста — проверить, что kwarg
        # не роняет функцию и возвращает список строк по числу кластеров.
        for use_lemma in (True, False):
            kw = extract_cluster_keywords_ctfidf(
                docs, labels, 2, top_n=5, use_lemma=use_lemma,
            )
            assert len(kw) == 2
            assert all(isinstance(s, str) for s in kw)
