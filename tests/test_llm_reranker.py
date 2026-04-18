# -*- coding: utf-8 -*-
"""Unit tests for llm_reranker.py — no real LLM calls, all mocked."""
from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from llm_reranker import (
    _build_rerank_user,
    _parse_rerank_response,
    build_class_examples_from_training,
    rerank_top_k,
)


# ---------------------------------------------------------------------------
# _build_rerank_user
# ---------------------------------------------------------------------------

def test_build_rerank_user_contains_text():
    prompt = _build_rerank_user("Почему не прошёл платёж?", ["cls_a", "cls_b"], {})
    assert "Почему не прошёл платёж?" in prompt
    assert "cls_a" in prompt
    assert "cls_b" in prompt


def test_build_rerank_user_truncates_long_text():
    long_text = "а" * 2000
    prompt = _build_rerank_user(long_text, ["cls"], {}, max_text_chars=100)
    assert "а" * 101 not in prompt


def test_build_rerank_user_includes_examples():
    examples = {"cls_a": ["Пример 1", "Пример 2"]}
    prompt = _build_rerank_user("текст", ["cls_a"], examples)
    assert "Пример 1" in prompt


def test_build_rerank_user_no_examples_for_class():
    prompt = _build_rerank_user("текст", ["cls_unknown"], {"other_cls": ["ex"]})
    assert "cls_unknown" in prompt  # class still listed
    assert "ex" not in prompt       # but example from other class not included


# ---------------------------------------------------------------------------
# _parse_rerank_response
# ---------------------------------------------------------------------------

def test_parse_exact_match():
    assert _parse_rerank_response("задержка_платежа", ["задержка_платежа", "блокировка"], "задержка_платежа") == "задержка_платежа"


def test_parse_case_insensitive():
    assert _parse_rerank_response("ЗАДЕРЖКА_ПЛАТЕЖА", ["задержка_платежа", "блокировка"], "блокировка") == "задержка_платежа"


def test_parse_substring_match():
    # LLM responds with extra words
    assert _parse_rerank_response("это задержка_платежа очевидно", ["задержка_платежа", "другой"], "другой") == "задержка_платежа"


def test_parse_strips_punctuation():
    assert _parse_rerank_response("«задержка_платежа».", ["задержка_платежа", "другой"], "другой") == "задержка_платежа"


def test_parse_empty_response_returns_fallback():
    assert _parse_rerank_response("", ["a", "b"], "a") == "a"


def test_parse_unknown_response_returns_fallback():
    assert _parse_rerank_response("что-то совершенно другое xyz", ["a", "b"], "b") == "b"


def test_parse_prefers_longer_candidate_in_substring():
    # "блокировка_карты" contains "блокировка" — should prefer longer match
    candidates = ["блокировка", "блокировка_карты"]
    result = _parse_rerank_response("это блокировка_карты", candidates, "блокировка")
    assert result == "блокировка_карты"


# ---------------------------------------------------------------------------
# build_class_examples_from_training
# ---------------------------------------------------------------------------

def test_build_class_examples_basic():
    X = ["текст1", "текст2", "текст3", "текст4"]
    y = ["cls_a", "cls_a", "cls_b", "cls_b"]
    result = build_class_examples_from_training(X, y, n_per_class=2)
    assert set(result.keys()) == {"cls_a", "cls_b"}
    assert len(result["cls_a"]) == 2
    assert len(result["cls_b"]) == 2


def test_build_class_examples_respects_n_per_class():
    X = ["t1", "t2", "t3", "t4", "t5"]
    y = ["a", "a", "a", "a", "a"]
    result = build_class_examples_from_training(X, y, n_per_class=2)
    assert len(result["a"]) == 2


def test_build_class_examples_truncates_text():
    X = ["а" * 500]
    y = ["cls"]
    result = build_class_examples_from_training(X, y, max_chars=50)
    assert len(result["cls"][0]) <= 50


def test_build_class_examples_skips_empty():
    X = ["", "нормальный текст"]
    y = ["cls", "cls"]
    result = build_class_examples_from_training(X, y)
    assert "нормальный текст" in result["cls"]
    assert "" not in result["cls"]


# ---------------------------------------------------------------------------
# rerank_top_k — mocked LLM
# ---------------------------------------------------------------------------

def test_rerank_top_k_uses_llm_response():
    """LLM returns a valid candidate — should use it."""
    with patch("llm_reranker.LLMClient.complete_text", return_value="блокировка_карты"):
        result = rerank_top_k(
            texts=["Карту заблокировали"],
            top_candidates=[["блокировка_карты", "задержка"]],
            argmax_labels=["задержка"],
            provider="openai", model="gpt-4o-mini", api_key="test",
        )
    assert result == ["блокировка_карты"]


def test_rerank_top_k_fallback_on_llm_error():
    """LLM raises — should fall back to argmax_labels."""
    from exceptions import FeatureBuildError
    with patch("llm_reranker.LLMClient.complete_text", side_effect=FeatureBuildError("err")):
        result = rerank_top_k(
            texts=["Карту заблокировали"],
            top_candidates=[["блокировка_карты", "задержка"]],
            argmax_labels=["задержка"],
            provider="openai", model="gpt-4o-mini", api_key="test",
        )
    assert result == ["задержка"]


def test_rerank_top_k_skips_single_candidate():
    """Only one candidate — skip LLM, use argmax directly."""
    mock = MagicMock()
    with patch("llm_reranker.LLMClient.complete_text", mock):
        result = rerank_top_k(
            texts=["текст"],
            top_candidates=[["единственный_класс"]],
            argmax_labels=["единственный_класс"],
            provider="openai", model="gpt-4o-mini", api_key="test",
        )
    mock.assert_not_called()
    assert result == ["единственный_класс"]


def test_rerank_top_k_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        rerank_top_k(
            texts=["t1", "t2"],
            top_candidates=[["a"]],   # length mismatch
            argmax_labels=["a", "b"],
            provider="openai", model="gpt-4o-mini", api_key="test",
        )


def test_rerank_top_k_multiple_rows():
    """Process multiple rows correctly."""
    responses = iter(["задержка_платежа", "блокировка_карты"])
    with patch("llm_reranker.LLMClient.complete_text", side_effect=responses):
        result = rerank_top_k(
            texts=["платёж завис", "карту заблокировали"],
            top_candidates=[
                ["задержка_платежа", "другой"],
                ["блокировка_карты", "другой"],
            ],
            argmax_labels=["другой", "другой"],
            provider="openai", model="gpt-4o-mini", api_key="test",
        )
    assert result == ["задержка_платежа", "блокировка_карты"]


def test_rerank_top_k_log_callback_called():
    """log_fn should be called with a summary string."""
    log_messages = []
    with patch("llm_reranker.LLMClient.complete_text", return_value="cls_a"):
        rerank_top_k(
            texts=["текст"],
            top_candidates=[["cls_a", "cls_b"]],
            argmax_labels=["cls_b"],
            provider="openai", model="gpt-4o-mini", api_key="test",
            log_fn=log_messages.append,
        )
    assert len(log_messages) == 1
    assert "rerank" in log_messages[0].lower() or "ре-ранк" in log_messages[0]


# ---------------------------------------------------------------------------
# Wave 4.4 — temperature default pinning (regression guard)
# ---------------------------------------------------------------------------
# Rationale: provider defaults vary (OpenAI 1.0, Qwen 0.7, Anthropic 1.0).
# Plan §3 W4 requires a deterministic-but-flexible 0.2 default for reranking —
# higher temperature produced flakier label picks in manual eval. These tests
# pin both the signature default and that it reaches the provider request.

def test_rerank_top_k_default_temperature_is_pinned():
    """The signature default for `temperature` must stay 0.2 (Wave 4.4 contract).

    If a future refactor changes this default, rerank quality drifts silently.
    Bump this test *and* docs/adr/000X-*.md together when changing the default.
    """
    import inspect

    sig = inspect.signature(rerank_top_k)
    assert sig.parameters["temperature"].default == pytest.approx(0.2), (
        "rerank_top_k temperature default drifted from 0.2 — see plan §3 W4.4."
    )


def test_rerank_top_k_forwards_temperature_to_llmclient():
    """Default call must forward temperature=0.2 to LLMClient.complete_text."""
    captured: dict = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return "cls_a"

    with patch("llm_reranker.LLMClient.complete_text", side_effect=_spy):
        rerank_top_k(
            texts=["текст"],
            top_candidates=[["cls_a", "cls_b"]],
            argmax_labels=["cls_b"],
            provider="openai", model="gpt-4o-mini", api_key="test",
        )
    assert captured.get("temperature") == pytest.approx(0.2)


def test_rerank_top_k_explicit_temperature_override():
    """Explicit temperature must override the 0.2 default (e.g. temperature=0.0)."""
    captured: dict = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return "cls_a"

    with patch("llm_reranker.LLMClient.complete_text", side_effect=_spy):
        rerank_top_k(
            texts=["текст"],
            top_candidates=[["cls_a", "cls_b"]],
            argmax_labels=["cls_b"],
            provider="openai", model="gpt-4o-mini", api_key="test",
            temperature=0.0,
        )
    assert captured.get("temperature") == pytest.approx(0.0)
