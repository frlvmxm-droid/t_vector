"""LLM reranker fallback / caching paths.

Wave 6.5 Block 3. Дополняет ``test_llm_reranker.py`` ветками:
  * `_parse_rerank_response` — whitespace, multiline, ``Класс:`` prefix,
    response, который после trim становится пустым.
  * `_cache_read` — отсутствие файла, битый JSON, отсутствующий ключ
    ``label``, нестроковое значение.
  * `_cache_write` — OSError проглатывается (read-only dir).
  * `_cache_key` — детерминирован, чувствителен к каждому полю.
  * `rerank_top_k` — in-batch dedup (один и тот же (text, candidates)
    → один LLM-вызов), disk-cache hit/miss + write-back, generic
    Exception (не только FeatureBuildError) ловится.

Все LLM-вызовы замоканы; реальная сеть не используется.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_reranker import (  # noqa: E402
    _cache_key,
    _cache_read,
    _cache_write,
    _parse_rerank_response,
    rerank_top_k,
)


# ---------------------------------------------------------------------------
# _parse_rerank_response — edge cases not covered in test_llm_reranker.py
# ---------------------------------------------------------------------------


def test_parse_whitespace_only_response_returns_fallback() -> None:
    """Только пробелы/переносы строк → fallback."""
    assert _parse_rerank_response("   \n\t  ", ["a", "b"], "a") == "a"


def test_parse_multiline_uses_first_nonempty_line() -> None:
    """LLM иногда возвращает 'класс\\n\\nобоснование' — берём первую непустую."""
    response = "\n\nкласс_a\n\nпотому что текст содержит ключевые слова..."
    assert _parse_rerank_response(response, ["класс_a", "класс_b"], "класс_b") == "класс_a"


def test_parse_strips_prefix_klass_colon() -> None:
    """Префикс 'Класс:' / 'Class -' / 'Ответ —' снимается."""
    assert _parse_rerank_response("Класс: блокировка", ["блокировка", "иное"], "иное") == "блокировка"
    assert _parse_rerank_response("Class - блокировка", ["блокировка", "иное"], "иное") == "блокировка"
    assert _parse_rerank_response("Ответ — блокировка", ["блокировка", "иное"], "иное") == "блокировка"


def test_parse_response_only_quotes_and_punctuation_returns_fallback() -> None:
    """Ответ из одних маркеров после trim → пустой → fallback."""
    assert _parse_rerank_response('«»".•-', ["a", "b"], "b") == "b"


def test_parse_none_response_returns_fallback() -> None:
    """None как response → fallback (defensive)."""
    assert _parse_rerank_response(None, ["a", "b"], "a") == "a"  # type: ignore[arg-type]


def test_parse_substring_prefers_longer_when_both_present() -> None:
    """'a' и 'aa' оба входят в ответ — выбирается длинный."""
    candidates = ["a", "abcdef"]
    assert _parse_rerank_response("ответ: abcdef", candidates, "a") == "abcdef"


# ---------------------------------------------------------------------------
# _cache_key — determinism and field sensitivity
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic() -> None:
    k1 = _cache_key("текст", ["a", "b"], "openai", "gpt-4o-mini")
    k2 = _cache_key("текст", ["a", "b"], "openai", "gpt-4o-mini")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_cache_key_changes_with_each_field() -> None:
    base = _cache_key("t", ["a"], "p", "m")
    assert _cache_key("T", ["a"], "p", "m") != base       # text
    assert _cache_key("t", ["b"], "p", "m") != base       # candidates
    assert _cache_key("t", ["a"], "P", "m") != base       # provider
    assert _cache_key("t", ["a"], "p", "M") != base       # model
    assert _cache_key("t", ["a", "b"], "p", "m") != base  # candidate count


# ---------------------------------------------------------------------------
# _cache_read — missing file / invalid JSON / wrong shape
# ---------------------------------------------------------------------------


def test_cache_read_missing_file_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    assert _cache_read("ab" + "0" * 62) is None


def test_cache_read_invalid_json_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    key = "cd" + "0" * 62
    p = tmp_path / key[:2] / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ broken json", encoding="utf-8")
    assert _cache_read(key) is None


def test_cache_read_missing_label_key_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    key = "ef" + "0" * 62
    p = tmp_path / key[:2] / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"other": "val"}), encoding="utf-8")
    assert _cache_read(key) is None


def test_cache_read_non_string_label_returns_none(tmp_path, monkeypatch) -> None:
    """label=123 (число) → отбрасывается, чтобы не вернуть int как str."""
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    key = "ab" + "1" * 62
    p = tmp_path / key[:2] / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"label": 123}), encoding="utf-8")
    assert _cache_read(key) is None


def test_cache_read_valid_label_returns_string(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    key = "ab" + "2" * 62
    p = tmp_path / key[:2] / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"label": "блокировка"}), encoding="utf-8")
    assert _cache_read(key) == "блокировка"


# ---------------------------------------------------------------------------
# _cache_write — round-trip + OSError swallow
# ---------------------------------------------------------------------------


def test_cache_write_then_read_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    key = "ff" + "9" * 62
    _cache_write(key, "выбранный_класс")
    assert _cache_read(key) == "выбранный_класс"


def test_cache_write_oserror_is_swallowed(tmp_path, monkeypatch) -> None:
    """Запись в read-only директорию → no raise (best-effort)."""
    if os.name == "nt":
        pytest.skip("chmod-based read-only emulation на Windows ненадёжен")
    ro = tmp_path / "ro_cache"
    ro.mkdir()
    ro.chmod(stat.S_IREAD | stat.S_IEXEC)  # 0o500 — нельзя писать
    try:
        monkeypatch.setattr("llm_reranker._CACHE_DIR", ro)
        # Не должно поднять исключение, даже при OSError на mkdir/open/replace.
        _cache_write("aa" + "0" * 62, "label")
    finally:
        ro.chmod(stat.S_IRWXU)  # вернуть права для cleanup tmp_path


# ---------------------------------------------------------------------------
# rerank_top_k — in-batch dedup, disk-cache, generic-exception path
# ---------------------------------------------------------------------------


def test_rerank_top_k_in_batch_dedup_one_llm_call_per_unique_key() -> None:
    """3 строки с одинаковым (text, candidates) → один LLM-вызов."""
    call_count = {"n": 0}

    def _spy(**_kw):
        call_count["n"] += 1
        return "cls_a"

    with patch("llm_reranker.LLMClient.complete_text", side_effect=_spy):
        result = rerank_top_k(
            texts=["одинаковый текст", "одинаковый текст", "одинаковый текст"],
            top_candidates=[["cls_a", "cls_b"]] * 3,
            argmax_labels=["cls_b", "cls_b", "cls_b"],
            provider="openai", model="gpt-4o-mini", api_key="x",
        )
    assert call_count["n"] == 1, "in-batch dedup должен сделать ровно 1 LLM-вызов"
    assert result == ["cls_a", "cls_a", "cls_a"]


def test_rerank_top_k_generic_exception_falls_back() -> None:
    """Не-FeatureBuildError (произвольный ValueError) → fallback (BLE001-safety net)."""
    with patch("llm_reranker.LLMClient.complete_text", side_effect=ValueError("boom")):
        result = rerank_top_k(
            texts=["t"],
            top_candidates=[["a", "b"]],
            argmax_labels=["b"],
            provider="openai", model="gpt-4o-mini", api_key="x",
        )
    assert result == ["b"]


def test_rerank_top_k_disk_cache_hit_skips_llm(tmp_path, monkeypatch) -> None:
    """use_disk_cache=True + предзаписанный кэш → LLM не вызывается."""
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    # Предзаписать ответ в кэш
    key = _cache_key("текст", ["a", "b"], "openai", "gpt-4o-mini")
    _cache_write(key, "a")

    with patch("llm_reranker.LLMClient.complete_text") as mock_llm:
        result = rerank_top_k(
            texts=["текст"],
            top_candidates=[["a", "b"]],
            argmax_labels=["b"],
            provider="openai", model="gpt-4o-mini", api_key="x",
            use_disk_cache=True,
        )
    mock_llm.assert_not_called()
    assert result == ["a"]


def test_rerank_top_k_disk_cache_miss_writes_back(tmp_path, monkeypatch) -> None:
    """use_disk_cache=True, кэш пуст → LLM вызвается, ответ пишется на диск."""
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    with patch("llm_reranker.LLMClient.complete_text", return_value="b"):
        result = rerank_top_k(
            texts=["другой текст"],
            top_candidates=[["a", "b"]],
            argmax_labels=["a"],
            provider="openai", model="gpt-4o-mini", api_key="x",
            use_disk_cache=True,
        )
    assert result == ["b"]
    # Кэш должен теперь содержать 'b'
    key = _cache_key("другой текст", ["a", "b"], "openai", "gpt-4o-mini")
    assert _cache_read(key) == "b"


def test_rerank_top_k_disk_cache_returns_label_not_in_candidates_is_ignored(
    tmp_path, monkeypatch
) -> None:
    """Если в кэше label, которого больше нет в candidates → запросить LLM заново."""
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    key = _cache_key("t", ["new_a", "new_b"], "openai", "gpt-4o-mini")
    _cache_write(key, "old_label_no_longer_valid")

    with patch("llm_reranker.LLMClient.complete_text", return_value="new_a") as mock_llm:
        result = rerank_top_k(
            texts=["t"],
            top_candidates=[["new_a", "new_b"]],
            argmax_labels=["new_b"],
            provider="openai", model="gpt-4o-mini", api_key="x",
            use_disk_cache=True,
        )
    mock_llm.assert_called_once()
    assert result == ["new_a"]


def test_rerank_top_k_zero_candidates_uses_argmax() -> None:
    """Пустой список кандидатов (len=0) → skip-путь, fallback к argmax."""
    with patch("llm_reranker.LLMClient.complete_text") as mock_llm:
        result = rerank_top_k(
            texts=["t"],
            top_candidates=[[]],
            argmax_labels=["fb"],
            provider="openai", model="gpt-4o-mini", api_key="x",
        )
    mock_llm.assert_not_called()
    assert result == ["fb"]


def test_rerank_top_k_log_fn_reports_dedup_and_disk_counters(tmp_path, monkeypatch) -> None:
    """log_fn получает строку с счётчиками rerank/disk-кэш/in-batch."""
    monkeypatch.setattr("llm_reranker._CACHE_DIR", tmp_path)
    logs: list[str] = []
    with patch("llm_reranker.LLMClient.complete_text", return_value="a"):
        rerank_top_k(
            texts=["t1", "t1", "t2"],  # t1 дублируется → in-batch dedup
            top_candidates=[["a", "b"], ["a", "b"], ["a", "b"]],
            argmax_labels=["b", "b", "b"],
            provider="openai", model="gpt-4o-mini", api_key="x",
            log_fn=logs.append,
            use_disk_cache=True,
        )
    assert len(logs) == 1
    msg = logs[0]
    assert "in-batch" in msg
    assert "disk" in msg
