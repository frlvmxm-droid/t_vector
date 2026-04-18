# -*- coding: utf-8 -*-
"""Tests for the offline LLM provider (ADR-0004).

Purpose: assert that with BRT_LLM_PROVIDER=offline the LLM client never
performs network I/O, returns deterministic responses, and integrates
cleanly with llm_reranker.rerank_top_k so CI can rely on it for
coverage and mutation runs without API keys.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from llm_client import LLMClient


# ---------------------------------------------------------------------------
# Env switch
# ---------------------------------------------------------------------------

def test_offline_enabled_false_by_default(monkeypatch):
    monkeypatch.delenv("BRT_LLM_PROVIDER", raising=False)
    assert LLMClient._offline_enabled() is False


def test_offline_enabled_true_on_env(monkeypatch):
    monkeypatch.setenv("BRT_LLM_PROVIDER", "offline")
    assert LLMClient._offline_enabled() is True


def test_offline_enabled_case_insensitive(monkeypatch):
    monkeypatch.setenv("BRT_LLM_PROVIDER", "OFFLINE")
    assert LLMClient._offline_enabled() is True


def test_offline_enabled_ignores_other_values(monkeypatch):
    monkeypatch.setenv("BRT_LLM_PROVIDER", "openai")
    assert LLMClient._offline_enabled() is False


# ---------------------------------------------------------------------------
# complete_text short-circuit
# ---------------------------------------------------------------------------

def test_complete_text_offline_never_builds_request(monkeypatch):
    """complete_text must not call _build_provider_request when offline."""
    monkeypatch.setenv("BRT_LLM_PROVIDER", "offline")
    # If the offline branch leaks into the network path, _build_provider_request
    # would be called. We replace it with a raiser to detect any leakage.
    with patch("llm_client.LLMClient._build_provider_request",
               side_effect=AssertionError("network path must not be reached")):
        result = LLMClient.complete_text(
            provider="openai",
            model="gpt-4o-mini",
            api_key="",
            system_prompt="sys",
            user_prompt="• cls_a\n• cls_b",
        )
    assert result == "cls_a"  # fallback: first bullet


def test_complete_text_offline_empty_prompt_returns_empty(monkeypatch):
    monkeypatch.setenv("BRT_LLM_PROVIDER", "offline")
    result = LLMClient.complete_text(
        provider="openai",
        model="gpt-4o-mini",
        api_key="",
        system_prompt="",
        user_prompt="no bullets here",
    )
    assert result == ""


# ---------------------------------------------------------------------------
# JSONL replay
# ---------------------------------------------------------------------------

def test_complete_text_offline_replay_hit(tmp_path, monkeypatch):
    """A matching entry in BRT_LLM_REPLAY wins over the bullet fallback."""
    monkeypatch.setenv("BRT_LLM_PROVIDER", "offline")

    key = LLMClient._cache_key(
        "openai", "gpt-4o-mini", "sys", "• a\n• b", 128, None,
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        json.dumps({"key": key, "response": "chosen_cls"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BRT_LLM_REPLAY", str(replay))

    result = LLMClient.complete_text(
        provider="openai",
        model="gpt-4o-mini",
        api_key="",
        system_prompt="sys",
        user_prompt="• a\n• b",
    )
    assert result == "chosen_cls"


def test_complete_text_offline_replay_miss_falls_back(tmp_path, monkeypatch):
    """Missing key in replay must fall through to the bullet fallback."""
    monkeypatch.setenv("BRT_LLM_PROVIDER", "offline")

    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        json.dumps({"key": "some_other_key", "response": "X"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BRT_LLM_REPLAY", str(replay))

    result = LLMClient.complete_text(
        provider="openai",
        model="gpt-4o-mini",
        api_key="",
        system_prompt="sys",
        user_prompt="• first_cls\n• second_cls",
    )
    assert result == "first_cls"


def test_complete_text_offline_corrupt_replay_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("BRT_LLM_PROVIDER", "offline")
    replay = tmp_path / "replay.jsonl"
    replay.write_text("this is not json\n\n{broken\n", encoding="utf-8")
    monkeypatch.setenv("BRT_LLM_REPLAY", str(replay))

    # Must not raise; falls back to bullet resolution.
    result = LLMClient.complete_text(
        provider="openai",
        model="gpt-4o-mini",
        api_key="",
        system_prompt="sys",
        user_prompt="• only_one",
    )
    assert result == "only_one"


# ---------------------------------------------------------------------------
# Integration with llm_reranker
# ---------------------------------------------------------------------------

def test_rerank_top_k_works_offline(monkeypatch):
    """rerank_top_k + BRT_LLM_PROVIDER=offline picks the first candidate
    without touching the network."""
    from llm_reranker import rerank_top_k

    monkeypatch.setenv("BRT_LLM_PROVIDER", "offline")
    # If any outbound request is attempted, urlopen would fail → raise.
    with patch("llm_client.LLMClient._build_provider_request",
               side_effect=AssertionError("must not call real provider")):
        out = rerank_top_k(
            texts=["карту заблокировали"],
            top_candidates=[["блокировка_карты", "задержка"]],
            argmax_labels=["задержка"],
            provider="openai",
            model="gpt-4o-mini",
            api_key="",
        )
    assert out == ["блокировка_карты"]
