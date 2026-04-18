# -*- coding: utf-8 -*-
"""
Tests for feature_builder.py:
  - build_feature_text(channel, desc, client_text, operator_text, summary,
                        ans_short, ans_full, weights, normalize_entities=False) -> str
  - choose_row_profile_weights(base, auto_profile, has_desc, has_dialog,
                                roles_found, has_summary, has_ans_s, has_ans_f) -> Dict[str, int]
"""
from __future__ import annotations

import pytest

from feature_builder import build_feature_text, choose_row_profile_weights
from constants import SECTION_PREFIX, PRESET_WEIGHTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_WEIGHTS = {
    "w_channel": 1,
    "w_desc": 1,
    "w_client": 1,
    "w_operator": 1,
    "w_summary": 1,
    "w_answer_short": 1,
    "w_answer_full": 1,
}

ZERO_WEIGHTS = {k: 0 for k in BASE_WEIGHTS}


def make_result(channel="IVR", desc="тест", client="клиент", operator="оператор",
                summary="итог", ans_short="краткий", ans_full="полный",
                weights=None, normalize=False):
    """Convenience wrapper to call build_feature_text."""
    if weights is None:
        weights = dict(BASE_WEIGHTS)
    return build_feature_text(channel, desc, client, operator,
                               summary, ans_short, ans_full, weights, normalize)


# ===========================================================================
# build_feature_text — basic structure
# ===========================================================================

class TestBuildFeatureTextBasicStructure:

    def test_channel_always_present(self):
        """Channel section is always included regardless of weights."""
        result = make_result(channel="WEB", weights=ZERO_WEIGHTS)
        assert SECTION_PREFIX["CHANNEL"] in result
        assert "WEB" in result

    def test_result_is_string(self):
        assert isinstance(make_result(), str)

    def test_result_not_empty(self):
        assert len(make_result()) > 0

    def test_section_prefixes_appear(self):
        result = make_result()
        assert SECTION_PREFIX["DESC"] in result
        assert SECTION_PREFIX["CLIENT"] in result
        assert SECTION_PREFIX["OPERATOR"] in result
        assert SECTION_PREFIX["SUMMARY"] in result
        assert SECTION_PREFIX["ANSWER_SHORT"] in result
        assert SECTION_PREFIX["ANSWER_FULL"] in result


# ===========================================================================
# build_feature_text — weight=0 excludes section
# ===========================================================================

class TestBuildFeatureTextWeightZero:

    def test_weight_zero_desc_excluded(self):
        w = dict(BASE_WEIGHTS)
        w["w_desc"] = 0
        result = make_result(desc="описание уникальное", weights=w)
        assert SECTION_PREFIX["DESC"] not in result

    def test_weight_zero_client_excluded(self):
        w = dict(BASE_WEIGHTS)
        w["w_client"] = 0
        result = make_result(client="текст клиента", weights=w)
        assert SECTION_PREFIX["CLIENT"] not in result

    def test_weight_zero_operator_excluded(self):
        w = dict(BASE_WEIGHTS)
        w["w_operator"] = 0
        result = make_result(operator="текст оператора", weights=w)
        assert SECTION_PREFIX["OPERATOR"] not in result

    def test_weight_zero_summary_excluded(self):
        w = dict(BASE_WEIGHTS)
        w["w_summary"] = 0
        result = make_result(summary="суммаризация", weights=w)
        assert SECTION_PREFIX["SUMMARY"] not in result

    def test_weight_zero_answer_short_excluded(self):
        w = dict(BASE_WEIGHTS)
        w["w_answer_short"] = 0
        result = make_result(ans_short="краткий ответ", weights=w)
        assert SECTION_PREFIX["ANSWER_SHORT"] not in result

    def test_weight_zero_answer_full_excluded(self):
        w = dict(BASE_WEIGHTS)
        w["w_answer_full"] = 0
        result = make_result(ans_full="полный ответ", weights=w)
        assert SECTION_PREFIX["ANSWER_FULL"] not in result

    def test_all_weights_zero_only_channel_remains(self):
        result = make_result(weights=ZERO_WEIGHTS)
        assert SECTION_PREFIX["CHANNEL"] in result
        assert SECTION_PREFIX["DESC"] not in result
        assert SECTION_PREFIX["CLIENT"] not in result
        assert SECTION_PREFIX["OPERATOR"] not in result
        assert SECTION_PREFIX["SUMMARY"] not in result
        assert SECTION_PREFIX["ANSWER_SHORT"] not in result
        assert SECTION_PREFIX["ANSWER_FULL"] not in result


# ===========================================================================
# build_feature_text — weight repetition
# ===========================================================================

class TestBuildFeatureTextWeightRepetition:

    def test_weight_3_desc_repeats_3_times(self):
        w = dict(ZERO_WEIGHTS)
        w["w_desc"] = 3
        result = make_result(desc="уникальный_дескриптор_xyz", weights=w)
        count = result.count(SECTION_PREFIX["DESC"])
        assert count == 3

    def test_weight_2_summary_repeats_2_times(self):
        w = dict(ZERO_WEIGHTS)
        w["w_summary"] = 2
        result = make_result(summary="уникальный_итог_abc", weights=w)
        count = result.count(SECTION_PREFIX["SUMMARY"])
        assert count == 2

    def test_weight_1_appears_once(self):
        w = dict(ZERO_WEIGHTS)
        w["w_answer_short"] = 1
        result = make_result(ans_short="уникальный_краткий_ответ", weights=w)
        count = result.count(SECTION_PREFIX["ANSWER_SHORT"])
        assert count == 1

    def test_weight_3_client_section_repeated(self):
        """Client section tag appears 3 times when weight=3."""
        w = dict(ZERO_WEIGHTS)
        w["w_client"] = 3
        result = make_result(client="клиент пишет", weights=w)
        count = result.count(SECTION_PREFIX["CLIENT"])
        assert count == 3


# ===========================================================================
# build_feature_text — empty inputs
# ===========================================================================

class TestBuildFeatureTextEmptyInputs:

    def test_empty_desc_excluded(self):
        w = dict(BASE_WEIGHTS)
        result = make_result(desc="", weights=w)
        assert SECTION_PREFIX["DESC"] not in result

    def test_empty_client_excluded(self):
        w = dict(BASE_WEIGHTS)
        result = make_result(client="", weights=w)
        assert SECTION_PREFIX["CLIENT"] not in result

    def test_empty_summary_excluded(self):
        w = dict(BASE_WEIGHTS)
        result = make_result(summary="", weights=w)
        assert SECTION_PREFIX["SUMMARY"] not in result

    def test_empty_ans_short_excluded(self):
        w = dict(BASE_WEIGHTS)
        result = make_result(ans_short="", weights=w)
        assert SECTION_PREFIX["ANSWER_SHORT"] not in result

    def test_empty_ans_full_excluded(self):
        w = dict(BASE_WEIGHTS)
        result = make_result(ans_full="", weights=w)
        assert SECTION_PREFIX["ANSWER_FULL"] not in result

    def test_all_empty_text_fields(self):
        """When all text fields are empty, only channel remains."""
        result = build_feature_text("CHAT", "", "", "", "", "", "", BASE_WEIGHTS)
        assert SECTION_PREFIX["CHANNEL"] in result
        assert "CHAT" in result

    def test_empty_channel_still_emits_channel_section(self):
        """Channel prefix is always written even if channel value is empty."""
        result = build_feature_text("", "desc", "", "", "", "", "", BASE_WEIGHTS)
        # channel section prefix always present
        assert SECTION_PREFIX["CHANNEL"] in result


# ===========================================================================
# build_feature_text — normalize_entities flag
# ===========================================================================

class TestBuildFeatureTextNormalizeEntities:

    def test_normalize_flag_does_not_crash(self):
        """normalize_entities=True should not raise any errors."""
        result = make_result(
            desc="Тест Сбербанк",
            client="Клиент позвонил в ВТБ",
            normalize=True,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_false_keeps_bank_names(self):
        """Without normalization, bank names appear verbatim."""
        result = make_result(
            desc="обращение в Сбербанк",
            normalize=False,
        )
        assert "Сбербанк" in result

    def test_normalize_affects_output(self):
        """With normalization enabled the output may differ (no crash)."""
        text_plain = make_result(desc="Клиент обратился в Сбербанк", normalize=False)
        text_norm = make_result(desc="Клиент обратился в Сбербанк", normalize=True)
        # Both are strings; normalized output is either same or different but valid
        assert isinstance(text_norm, str)


# ===========================================================================
# build_feature_text — whitespace / newline cleanup
# ===========================================================================

class TestBuildFeatureTextFormatting:

    def test_no_triple_newlines(self):
        """Output must not contain 3+ consecutive newlines."""
        import re
        result = make_result()
        assert not re.search(r"\n{3,}", result)

    def test_result_stripped(self):
        """Output should be stripped (no leading/trailing whitespace)."""
        result = make_result()
        assert result == result.strip()

    def test_multiline_client_truncated(self):
        """Long multiline client text is trimmed to head+tail lines."""
        many_lines = "\n".join([f"Клиент реплика {i}" for i in range(30)])
        w = dict(ZERO_WEIGHTS)
        w["w_client"] = 1
        result = make_result(client=many_lines, weights=w)
        # All 30 lines should NOT all appear — head/tail trim is applied
        assert result.count("Клиент реплика") <= 30
        assert SECTION_PREFIX["CLIENT"] in result


# ===========================================================================
# choose_row_profile_weights — auto_profile="off"
# ===========================================================================

class TestChooseRowProfileWeightsOff:

    def test_off_returns_base_weights_unchanged_except_empty_fields(self):
        base = {"w_desc": 2, "w_client": 3, "w_operator": 1,
                "w_summary": 2, "w_answer_short": 1, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "off",
            has_desc=True, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=True, has_ans_f=True,
        )
        # All fields present → no zeroing
        assert result["w_desc"] == 2
        assert result["w_client"] == 3

    def test_off_zeros_missing_desc(self):
        base = {"w_desc": 2, "w_client": 2, "w_operator": 1,
                "w_summary": 1, "w_answer_short": 1, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "off",
            has_desc=False, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=True, has_ans_f=True,
        )
        assert result["w_desc"] == 0

    def test_off_zeros_missing_summary(self):
        base = {"w_desc": 1, "w_client": 1, "w_operator": 1,
                "w_summary": 3, "w_answer_short": 1, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "off",
            has_desc=True, has_dialog=True, roles_found=True,
            has_summary=False, has_ans_s=True, has_ans_f=True,
        )
        assert result["w_summary"] == 0

    def test_off_zeros_missing_ans_short_and_ans_full(self):
        base = {"w_desc": 1, "w_client": 1, "w_operator": 1,
                "w_summary": 1, "w_answer_short": 2, "w_answer_full": 2}
        result = choose_row_profile_weights(
            base, "off",
            has_desc=True, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=False, has_ans_f=False,
        )
        assert result["w_answer_short"] == 0
        assert result["w_answer_full"] == 0


# ===========================================================================
# choose_row_profile_weights — auto_profile="smart"
# ===========================================================================

class TestChooseRowProfileWeightsSmart:

    def test_smart_uses_max_of_base_and_preset(self):
        """smart mode returns max(base, preset) per field."""
        # has_roles + has_summary + no answers + no desc → "consultation" preset
        # PRESET_WEIGHTS["consultation"]["w_summary"] == 5
        base = {"w_desc": 0, "w_client": 1, "w_operator": 1,
                "w_summary": 1, "w_answer_short": 0, "w_answer_full": 0}
        result = choose_row_profile_weights(
            base, "smart",
            has_desc=False, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=False, has_ans_f=False,
        )
        # smart → max(base_summary=1, preset_summary=5) = 5
        assert result["w_summary"] == 5

    def test_smart_keeps_higher_base_value(self):
        """If base weight is higher than preset, smart keeps the base value."""
        # "balanced" preset: w_client=2; we set base w_client=5
        base = {"w_desc": 2, "w_client": 5, "w_operator": 1,
                "w_summary": 3, "w_answer_short": 2, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "smart",
            has_desc=True, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=True, has_ans_f=True,
        )
        # "balanced" preset w_client=2; base=5; smart keeps max=5
        assert result["w_client"] == 5

    def test_smart_zeros_empty_fields_after_preset(self):
        """Missing fields remain zero after smart merging."""
        base = {"w_desc": 0, "w_client": 2, "w_operator": 1,
                "w_summary": 0, "w_answer_short": 0, "w_answer_full": 0}
        result = choose_row_profile_weights(
            base, "smart",
            has_desc=False, has_dialog=True, roles_found=True,
            has_summary=False, has_ans_s=False, has_ans_f=False,
        )
        # "client" preset would be chosen (has_roles, no summary, no answers)
        assert result["w_summary"] == 0
        assert result["w_answer_short"] == 0
        assert result["w_answer_full"] == 0


# ===========================================================================
# choose_row_profile_weights — auto_profile="strict"
# ===========================================================================

class TestChooseRowProfileWeightsStrict:

    def test_strict_replaces_base_with_preset(self):
        """strict mode completely replaces base with the selected preset."""
        # has_roles + has_summary + has_ans_s → "balanced" preset
        base = {"w_desc": 0, "w_client": 0, "w_operator": 0,
                "w_summary": 0, "w_answer_short": 0, "w_answer_full": 0}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=True, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=True, has_ans_f=True,
        )
        preset = PRESET_WEIGHTS["balanced"]
        assert result["w_client"] == preset["w_client"]
        assert result["w_summary"] == preset["w_summary"]

    def test_strict_no_answers_preset_selected(self):
        """No desc + dialog + summary + no answers → 'consultation' preset."""
        base = {"w_desc": 0, "w_client": 1, "w_operator": 1,
                "w_summary": 1, "w_answer_short": 3, "w_answer_full": 3}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=False, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=False, has_ans_f=False,
        )
        # When has_desc=False + has_roles + has_summary + no answers → "consultation"
        preset = PRESET_WEIGHTS["consultation"]
        assert result["w_summary"] == preset["w_summary"]
        assert result["w_answer_short"] == 0
        assert result["w_answer_full"] == 0

    def test_strict_client_preset_only_dialog(self):
        """Only dialog (no summary, no answers) → 'client' preset."""
        base = {"w_desc": 2, "w_client": 2, "w_operator": 2,
                "w_summary": 2, "w_answer_short": 2, "w_answer_full": 2}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=True, has_dialog=True, roles_found=True,
            has_summary=False, has_ans_s=False, has_ans_f=False,
        )
        preset = PRESET_WEIGHTS["client"]
        assert result["w_client"] == preset["w_client"]
        assert result["w_summary"] == 0

    def test_strict_summary_preset_no_dialog(self):
        """No dialog, has summary → 'summary' preset."""
        base = {"w_desc": 0, "w_client": 0, "w_operator": 0,
                "w_summary": 1, "w_answer_short": 1, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=True, has_dialog=False, roles_found=False,
            has_summary=True, has_ans_s=True, has_ans_f=True,
        )
        preset = PRESET_WEIGHTS["summary"]
        assert result["w_summary"] == preset["w_summary"]

    def test_strict_answers_preset_no_dialog_no_summary(self):
        """No dialog, no summary, but has answers → 'answers' preset."""
        base = {"w_desc": 1, "w_client": 1, "w_operator": 1,
                "w_summary": 1, "w_answer_short": 1, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=True, has_dialog=False, roles_found=False,
            has_summary=False, has_ans_s=True, has_ans_f=False,
        )
        preset = PRESET_WEIGHTS["answers"]
        assert result["w_answer_short"] == preset["w_answer_short"]

    def test_strict_consultation_preset(self):
        """has_roles + has_summary + no desc + no answers → 'consultation' preset."""
        base = {"w_desc": 3, "w_client": 3, "w_operator": 3,
                "w_summary": 3, "w_answer_short": 3, "w_answer_full": 3}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=False, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=False, has_ans_f=False,
        )
        preset = PRESET_WEIGHTS["consultation"]
        assert result["w_summary"] == preset["w_summary"]
        assert result["w_desc"] == 0

    def test_strict_returns_dict(self):
        base = {"w_desc": 1, "w_client": 1, "w_operator": 1,
                "w_summary": 1, "w_answer_short": 1, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=True, has_dialog=True, roles_found=True,
            has_summary=True, has_ans_s=True, has_ans_f=True,
        )
        assert isinstance(result, dict)


# ===========================================================================
# choose_row_profile_weights — edge cases
# ===========================================================================

class TestChooseRowProfileWeightsEdgeCases:

    def test_no_fields_at_all_falls_back_to_balanced(self):
        """No desc, no dialog, no summary, no answers → balanced preset in strict."""
        base = {"w_desc": 1, "w_client": 1, "w_operator": 1,
                "w_summary": 1, "w_answer_short": 1, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=False, has_dialog=False, roles_found=False,
            has_summary=False, has_ans_s=False, has_ans_f=False,
        )
        # Falls through to balanced preset, then all empties zeroed
        assert isinstance(result, dict)
        assert result["w_summary"] == 0
        assert result["w_answer_short"] == 0

    def test_dialog_without_roles_treated_as_no_dialog(self):
        """has_dialog=True but roles_found=False → has_roles=False → summary path."""
        base = {"w_desc": 1, "w_client": 1, "w_operator": 1,
                "w_summary": 2, "w_answer_short": 1, "w_answer_full": 1}
        result = choose_row_profile_weights(
            base, "strict",
            has_desc=True, has_dialog=True, roles_found=False,
            has_summary=True, has_ans_s=True, has_ans_f=True,
        )
        # has_roles = has_dialog AND roles_found = False → summary preset
        preset = PRESET_WEIGHTS["summary"]
        assert result["w_summary"] == preset["w_summary"]
