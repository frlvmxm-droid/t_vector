# -*- coding: utf-8 -*-
"""Comprehensive unit tests for ml_augment.py."""
from __future__ import annotations

import threading
from typing import List

import pytest

from ml_augment import (
    _parse_lines,
    augment_rare_classes,
    generate_class_paraphrases,
)


# ---------------------------------------------------------------------------
# Shared mock LLM function
# ---------------------------------------------------------------------------

def mock_llm(provider, model, api_key, system_prompt, user_prompt, max_tokens):
    """Returns a fixed LLM response with three numbered lines."""
    return "1. Первый вариант\n2. Второй вариант\n3. Третий вариант"


def mock_llm_bullet(provider, model, api_key, system_prompt, user_prompt, max_tokens):
    """Returns bullet-prefixed paraphrases."""
    return "- Первый вариант\n- Второй вариант\n- Третий вариант"


def mock_llm_empty(provider, model, api_key, system_prompt, user_prompt, max_tokens):
    """Returns an empty string."""
    return ""


def mock_llm_raises(provider, model, api_key, system_prompt, user_prompt, max_tokens):
    """Always raises an exception to simulate LLM error."""
    raise RuntimeError("LLM service unavailable")


# ===========================================================================
# _parse_lines() internal helper
# ===========================================================================

class TestParseLines:

    def test_numbered_lines_are_preserved_as_is(self):
        # _parse_lines only strips leading -, •, * — numbered lines are kept verbatim
        text = "1. First\n2. Second\n3. Third"
        result = _parse_lines(text)
        assert len(result) == 3
        assert "First" in result[0]
        assert "Second" in result[1]
        assert "Third" in result[2]

    def test_strips_bullet_prefix(self):
        text = "- First\n- Second"
        result = _parse_lines(text)
        assert result == ["First", "Second"]

    def test_strips_asterisk_prefix(self):
        text = "* One\n* Two"
        result = _parse_lines(text)
        assert result == ["One", "Two"]

    def test_empty_string_returns_empty_list(self):
        assert _parse_lines("") == []

    def test_blank_lines_skipped(self):
        text = "\nFirst\n\nSecond\n"
        result = _parse_lines(text)
        assert "First" in result
        assert "Second" in result
        assert "" not in result

    def test_mixed_prefixes(self):
        text = "1. Alpha\n- Beta\n• Gamma"
        result = _parse_lines(text)
        assert len(result) == 3


# ===========================================================================
# generate_class_paraphrases()
# ===========================================================================

class TestGenerateClassParaphrases:

    def test_returns_list(self):
        result = generate_class_paraphrases(
            examples=["Пример обращения"],
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        assert isinstance(result, list)

    def test_empty_examples_returns_empty_list(self):
        result = generate_class_paraphrases(
            examples=[],
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        assert result == []

    def test_returns_nonempty_list_for_valid_examples(self):
        result = generate_class_paraphrases(
            examples=["Клиент хочет закрыть счет"],
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        assert len(result) > 0

    def test_returns_strings_in_list(self):
        result = generate_class_paraphrases(
            examples=["Клиент хочет закрыть счет"],
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        assert all(isinstance(t, str) for t in result)

    def test_cancel_event_set_returns_empty(self):
        evt = threading.Event()
        evt.set()
        result = generate_class_paraphrases(
            examples=["Пример"],
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            cancel_event=evt,
        )
        assert result == []

    def test_cancel_event_not_set_proceeds_normally(self):
        evt = threading.Event()
        # NOT set — should proceed
        result = generate_class_paraphrases(
            examples=["Пример"],
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            cancel_event=evt,
        )
        assert len(result) > 0

    def test_llm_exception_returns_empty_list(self):
        result = generate_class_paraphrases(
            examples=["Пример"],
            n_paraphrases=3,
            llm_complete_fn=mock_llm_raises,
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        assert result == []

    def test_llm_empty_response_returns_empty_list(self):
        result = generate_class_paraphrases(
            examples=["Пример"],
            n_paraphrases=3,
            llm_complete_fn=mock_llm_empty,
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )
        assert result == []

    def test_uses_at_most_three_seed_examples(self):
        """The prompt must reference only up to 3 examples even if more are provided."""
        calls: list[dict] = []

        def capturing_llm(provider, model, api_key, system_prompt, user_prompt, max_tokens):
            calls.append({"user_prompt": user_prompt})
            return "- Вариант"

        examples = ["Ex1", "Ex2", "Ex3", "Ex4", "Ex5"]
        generate_class_paraphrases(
            examples=examples,
            n_paraphrases=3,
            llm_complete_fn=capturing_llm,
            provider="openai",
            model="gpt-4",
            api_key="key",
        )
        assert len(calls) == 1
        # Only "Пример 1/2/3" should appear, not "Пример 4"
        assert "Пример 4" not in calls[0]["user_prompt"]
        assert "Пример 5" not in calls[0]["user_prompt"]

    def test_bullet_response_parsed_correctly(self):
        result = generate_class_paraphrases(
            examples=["Пример"],
            n_paraphrases=3,
            llm_complete_fn=mock_llm_bullet,
            provider="openai",
            model="gpt-4",
            api_key="key",
        )
        assert len(result) == 3
        assert all("вариант" in r.lower() for r in result)

    def test_cancel_event_none_does_not_error(self):
        result = generate_class_paraphrases(
            examples=["Пример"],
            n_paraphrases=2,
            llm_complete_fn=mock_llm,
            provider="openai",
            model="gpt-4",
            api_key="key",
            cancel_event=None,
        )
        assert isinstance(result, list)


# ===========================================================================
# augment_rare_classes()
# ===========================================================================

class TestAugmentRareClasses:

    # -----------------------------------------------------------------------
    # No rare classes
    # -----------------------------------------------------------------------

    def test_no_rare_classes_returns_original_X(self):
        X = ["a", "b", "c", "d"]
        y = ["cat", "cat", "dog", "dog"]
        X_aug, y_aug, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert X_aug == list(X)
        assert y_aug == list(y)

    def test_no_rare_classes_report_zeros(self):
        X = ["a", "b", "c", "d"]
        y = ["cat", "cat", "dog", "dog"]
        _, _, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert report["classes_augmented"] == 0
        assert report["rows_added"] == 0

    def test_no_rare_classes_report_has_skipped_field(self):
        X = ["a", "b"]
        y = ["cat", "cat"]
        _, _, report = augment_rare_classes(
            X, y,
            min_samples_threshold=1,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert "skipped" in report
        assert isinstance(report["skipped"], list)

    # -----------------------------------------------------------------------
    # With rare classes
    # -----------------------------------------------------------------------

    def test_rare_class_gets_examples_added(self):
        X = ["обычный текст", "кот", "кот", "собака"]
        y = ["common", "common", "common", "rare"]
        X_aug, y_aug, report = augment_rare_classes(
            X, y,
            min_samples_threshold=3,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert len(X_aug) > len(X)
        assert len(y_aug) > len(y)

    def test_rare_class_label_added_correctly(self):
        X = ["text1", "t2", "t3", "rare_text"]
        y = ["common", "common", "common", "rare"]
        X_aug, y_aug, _ = augment_rare_classes(
            X, y,
            min_samples_threshold=3,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        new_labels = y_aug[len(y):]
        assert all(lbl == "rare" for lbl in new_labels)

    def test_report_classes_augmented_count(self):
        X = ["t1", "t2", "t3", "r1", "s1"]
        y = ["common", "common", "common", "rare1", "rare2"]
        _, _, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert report["classes_augmented"] == 2

    def test_report_rows_added_positive(self):
        X = ["t1", "t2", "t3", "r1"]
        y = ["common", "common", "common", "rare"]
        _, _, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert report["rows_added"] > 0

    def test_report_contains_required_fields(self):
        X = ["t1", "t2", "t3", "r1"]
        y = ["common", "common", "common", "rare"]
        _, _, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert "classes_augmented" in report
        assert "rows_added" in report
        assert "skipped" in report

    def test_llm_error_skips_class_returns_original(self):
        X = ["t1", "t2", "t3", "r1"]
        y = ["common", "common", "common", "rare"]
        X_aug, y_aug, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm_raises,
            provider="openai", model="gpt-4", api_key="key",
        )
        # Should still return lists of same or similar length (no crash)
        assert isinstance(X_aug, list)
        assert isinstance(y_aug, list)
        assert len(X_aug) == len(y_aug)
        # LLM failed → class skipped
        assert "rare" in report["skipped"]

    def test_cancel_event_stops_processing(self):
        evt = threading.Event()
        evt.set()

        X = ["t1", "t2", "t3", "r1", "s1"]
        y = ["common", "common", "common", "rare1", "rare2"]
        X_aug, y_aug, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
            cancel_event=evt,
        )
        # Cancelled before any work → original data returned unchanged
        assert len(X_aug) == len(X)
        assert report["rows_added"] == 0

    def test_log_fn_called_for_rare_class(self):
        log_messages: list[str] = []

        def log_fn(msg: str):
            log_messages.append(msg)

        X = ["t1", "t2", "t3", "r1"]
        y = ["common", "common", "common", "rare"]
        augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
            log_fn=log_fn,
        )
        assert len(log_messages) >= 1
        assert any("rare" in msg for msg in log_messages)

    def test_original_data_not_mutated(self):
        X = ["t1", "t2", "t3", "r1"]
        y = ["common", "common", "common", "rare"]
        X_copy = list(X)
        y_copy = list(y)
        augment_rare_classes(
            X, y,
            min_samples_threshold=2,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert X == X_copy
        assert y == y_copy

    def test_returns_tuple_of_three(self):
        X = ["t1", "t2"]
        y = ["a", "a"]
        result = augment_rare_classes(
            X, y,
            min_samples_threshold=1,
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_x_and_y_lengths_always_match(self):
        X = ["t1", "t2", "t3", "r1", "r2", "s1"]
        y = ["a", "a", "a", "b", "b", "c"]
        X_aug, y_aug, _ = augment_rare_classes(
            X, y,
            min_samples_threshold=3,
            n_paraphrases=5,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert len(X_aug) == len(y_aug)

    def test_threshold_boundary_exact_match_not_rare(self):
        """A class with exactly min_samples_threshold examples is NOT rare."""
        X = ["t1", "t2"]
        y = ["cat", "cat"]
        _, _, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,  # exactly 2 == threshold → not rare
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        assert report["classes_augmented"] == 0

    def test_threshold_one_below_is_rare(self):
        """A class with threshold-1 samples IS rare."""
        X = ["t1"]
        y = ["cat"]
        _, _, report = augment_rare_classes(
            X, y,
            min_samples_threshold=2,  # 1 < 2 → rare
            n_paraphrases=3,
            llm_complete_fn=mock_llm,
            provider="openai", model="gpt-4", api_key="key",
        )
        # classes_augmented should be 1 (LLM succeeded with mock)
        assert report["classes_augmented"] == 1 or report["skipped"] == ["cat"]
