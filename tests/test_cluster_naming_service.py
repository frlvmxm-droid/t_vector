"""Unit tests for `cluster_naming_service` (Phase 11 port)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cluster_naming_service import (
    DEFAULT_SYSTEM_PROMPT,
    name_clusters_with_llm,
)


def _stub_complete(name_for_cluster: dict[int, str]):
    """Return a fake `complete_fn` that picks a name by inspecting the prompt."""
    calls: list[dict] = []

    def fn(**kwargs: Any) -> str:
        calls.append(kwargs)
        prompt = kwargs.get("user_prompt", "")
        # Cluster id is encoded into our test keywords as "kw_<id>"
        for cid, name in name_for_cluster.items():
            if f"kw_{cid}" in prompt:
                return name
        return ""

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def test_name_clusters_with_llm_dict_keywords():
    labels = np.array([0, 0, 1, 1, 2, 2])
    texts = ["a1", "a2", "b1", "b2", "c1", "c2"]
    keywords = {0: "kw_0 cards", 1: "kw_1 mobile", 2: "kw_2 transfer"}
    fn = _stub_complete({0: "Карты", 1: "Мобильное приложение", 2: "Переводы"})

    out = name_clusters_with_llm(
        labels, texts,
        keywords=keywords,
        provider="offline", model="stub", api_key="",
        complete_fn=fn,
    )
    assert out == {0: "Карты", 1: "Мобильное приложение", 2: "Переводы"}
    assert len(fn.calls) == 3  # type: ignore[attr-defined]
    assert all(c["system_prompt"] == DEFAULT_SYSTEM_PROMPT for c in fn.calls)  # type: ignore[attr-defined]


def test_name_clusters_with_llm_sequence_keywords():
    labels = np.array([0, 0, 1, 1])
    texts = ["a", "b", "c", "d"]
    keywords = ["kw_0 alpha", "kw_1 beta"]
    fn = _stub_complete({0: "Альфа-кластер", 1: "Бета-кластер"})

    out = name_clusters_with_llm(
        labels, texts, keywords=keywords,
        provider="offline", model="stub", api_key="", complete_fn=fn,
    )
    assert out == {0: "Альфа-кластер", 1: "Бета-кластер"}


def test_name_clusters_with_llm_skips_empty_keyword():
    labels = np.array([0, 0, 1, 1])
    texts = ["a", "b", "c", "d"]
    keywords = {0: "kw_0 alpha", 1: ""}
    fn = _stub_complete({0: "Альфа", 1: "не-должен-вызваться"})
    out = name_clusters_with_llm(
        labels, texts, keywords=keywords,
        provider="offline", model="stub", api_key="", complete_fn=fn,
    )
    assert out == {0: "Альфа"}
    assert len(fn.calls) == 1  # type: ignore[attr-defined]


def test_name_clusters_with_llm_strips_quotes_and_chevrons():
    labels = np.array([0, 0])
    keywords = {0: "kw_0"}
    fn = _stub_complete({0: '«Списание комиссии»'})
    out = name_clusters_with_llm(
        labels, ["x", "y"], keywords=keywords,
        provider="offline", model="stub", api_key="", complete_fn=fn,
    )
    assert out == {0: "Списание комиссии"}


def test_name_clusters_with_llm_continues_on_per_row_error():
    labels = np.array([0, 1])

    def flaky(**kwargs):
        if "kw_0" in kwargs["user_prompt"]:
            raise RuntimeError("boom")
        return "OK-1"

    log_lines: list[str] = []
    out = name_clusters_with_llm(
        labels, ["a", "b"],
        keywords={0: "kw_0", 1: "kw_1"},
        provider="offline", model="stub", api_key="",
        complete_fn=flaky,
        log_cb=log_lines.append,
    )
    assert out == {1: "OK-1"}
    assert any("boom" in line for line in log_lines)


def test_name_clusters_with_llm_offline_default(monkeypatch: pytest.MonkeyPatch):
    """When BRT_LLM_PROVIDER=offline, default LLMClient.complete_text returns
    a deterministic stub. We assert the call simply doesn't raise (real
    network would fail in CI), and result type is a dict.
    """
    monkeypatch.setenv("BRT_LLM_PROVIDER", "offline")
    labels = np.array([0, 1])
    out = name_clusters_with_llm(
        labels, ["text-a", "text-b"],
        keywords={0: "kw_a", 1: "kw_b"},
        provider="anthropic", model="claude-sonnet-4-6",
    )
    assert isinstance(out, dict)
