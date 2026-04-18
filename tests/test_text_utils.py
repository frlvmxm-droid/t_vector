# -*- coding: utf-8 -*-
"""
Unit tests for text_utils module.

Functions under test:
  - clean_masks(s)
  - strip_html(x)
  - normalize_text(x)
  - clean_answer_text(x)
  - parse_dialog_roles(text, ignore_chatbot)
"""
import pytest
from text_utils import (
    clean_masks,
    strip_html,
    normalize_text,
    clean_answer_text,
    parse_dialog_roles,
)


# ---------------------------------------------------------------------------
# clean_masks
# ---------------------------------------------------------------------------

class TestCleanMasks:
    def test_none_returns_empty_string(self):
        assert clean_masks(None) == ""

    def test_empty_string_unchanged(self):
        assert clean_masks("") == ""

    def test_removes_single_mask(self):
        result = clean_masks("Звонил !phone! по вопросу")
        assert "!phone!" not in result

    def test_removes_fio_mask(self):
        result = clean_masks("Клиент !fio! обратился")
        assert "!fio!" not in result

    def test_removes_multiple_masks(self):
        result = clean_masks("!fio! живёт по адресу !data! телефон !num!")
        assert "!" not in result

    def test_normalizes_spaces_after_mask_removal(self):
        result = clean_masks("Слово  !mask!  конец")
        assert "  " not in result

    def test_strips_leading_trailing_spaces(self):
        result = clean_masks("  текст  ")
        assert result == "текст"

    def test_plain_text_unchanged(self):
        text = "Обычный текст без масок"
        assert clean_masks(text) == text

    def test_mask_at_start(self):
        result = clean_masks("!email! написал письмо")
        assert result.startswith("написал") or result[0] != "!"

    def test_mask_at_end(self):
        result = clean_masks("Паспорт !data!")
        assert "!" not in result


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_none_returns_empty_string(self):
        assert strip_html(None) == ""

    def test_empty_string(self):
        assert strip_html("") == ""

    def test_removes_simple_tags(self):
        result = strip_html("<p>Привет</p>")
        assert "<p>" not in result
        assert "Привет" in result

    def test_removes_br_tag(self):
        result = strip_html("Строка1<br>Строка2")
        assert "<br>" not in result
        assert "Строка1" in result
        assert "Строка2" in result

    def test_removes_nbsp_entity(self):
        result = strip_html("Слово&nbsp;ещё")
        assert "&nbsp;" not in result

    def test_removes_html_attributes(self):
        result = strip_html('<span class="bold">Текст</span>')
        assert "<span" not in result
        assert "Текст" in result

    def test_normalizes_whitespace(self):
        result = strip_html("<p>  много   пробелов  </p>")
        assert "  " not in result

    def test_non_string_input_converted(self):
        result = strip_html(42)
        assert result == "42"

    def test_nested_tags(self):
        result = strip_html("<div><p><b>Текст</b></p></div>")
        assert "<" not in result
        assert "Текст" in result

    def test_plain_text_unchanged(self):
        text = "Обычный текст"
        assert strip_html(text) == text

    def test_removes_div_word_artifact(self):
        # PLAIN_HTML_WORDS_RE removes standalone "div", "br", "nbsp", "msonormal"
        result = strip_html("Текст div конец")
        assert "div" not in result

    def test_nbsp_unicode_removed(self):
        result = strip_html("Слово\u00a0конец")
        assert "\u00a0" not in result


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_none_returns_empty_string(self):
        assert normalize_text(None) == ""

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_crlf_normalized_to_lf(self):
        result = normalize_text("Строка1\r\nСтрока2")
        assert "\r" not in result

    def test_cr_normalized(self):
        result = normalize_text("Строка1\rСтрока2")
        assert "\r" not in result

    def test_removes_masks(self):
        result = normalize_text("Клиент !fio! обратился")
        assert "!fio!" not in result

    def test_e_umlaut_replaced(self):
        # ë (U+00EB) → е
        result = normalize_text("Клиёнт обратился")  # contains ë
        assert "\u00eb" not in result

    def test_en_dash_replaced(self):
        result = normalize_text("Период \u2013 год")
        assert "\u2013" not in result
        assert "-" in result

    def test_em_dash_replaced(self):
        result = normalize_text("Период \u2014 год")
        assert "\u2014" not in result
        assert "-" in result

    def test_nonbreaking_space_replaced(self):
        result = normalize_text("Слово\u00a0конец")
        assert "\u00a0" not in result

    def test_removes_empty_lines(self):
        result = normalize_text("Строка1\n\nСтрока2")
        assert "\n\n" not in result
        lines = result.splitlines()
        assert all(line.strip() for line in lines)

    def test_drops_system_line_sdk_request(self):
        result = normalize_text("sdk_request")
        assert "sdk_request" not in result

    def test_drops_operator_connecting_line(self):
        result = normalize_text("оператор подключается к чату")
        assert "оператор подключается" not in result

    def test_normalizes_multiple_spaces(self):
        result = normalize_text("Много    пробелов")
        assert "  " not in result

    def test_strips_leading_trailing_whitespace(self):
        result = normalize_text("  текст  ")
        assert result == "текст"

    def test_multiline_preserved(self):
        text = "Строка 1 с контентом\nСтрока 2 тоже"
        result = normalize_text(text)
        assert "Строка 1" in result
        assert "Строка 2" in result

    def test_non_string_input_converted(self):
        result = normalize_text(123)
        assert result == "123"


# ---------------------------------------------------------------------------
# clean_answer_text
# ---------------------------------------------------------------------------

class TestCleanAnswerText:
    def test_none_returns_empty_string(self):
        assert clean_answer_text(None) == ""

    def test_empty_string(self):
        assert clean_answer_text("") == ""

    def test_removes_html_tags(self):
        result = clean_answer_text("<p>Ответ банка</p>")
        assert "<p>" not in result
        assert "Ответ банка" in result

    def test_removes_masks(self):
        result = clean_answer_text("Уважаемый !fio! ваш запрос")
        assert "!fio!" not in result

    def test_removes_answer_header(self):
        text = "Мы рассмотрели ваше обращение и отвечаем 01.01.2024. Ответ: отказано"
        result = clean_answer_text(text)
        assert "рассмотрели" not in result
        assert "отказано" in result

    def test_normalizes_em_dash(self):
        result = clean_answer_text("Ответ \u2014 отказано")
        assert "\u2014" not in result
        assert "-" in result

    def test_normalizes_en_dash(self):
        result = clean_answer_text("Ответ \u2013 принято")
        assert "\u2013" not in result

    def test_normalizes_nbsp(self):
        result = clean_answer_text("Ответ\u00a0банка")
        assert "\u00a0" not in result

    def test_plain_text_preserved(self):
        text = "Ваша заявка одобрена"
        result = clean_answer_text(text)
        assert "одобрена" in result

    def test_non_string_input(self):
        result = clean_answer_text(42)
        assert result == "42"

    def test_html_nbsp_entity_removed(self):
        result = clean_answer_text("Ответ&nbsp;банка")
        assert "&nbsp;" not in result

    def test_normalizes_whitespace(self):
        result = clean_answer_text("Много    пробелов   в   тексте")
        assert "  " not in result


# ---------------------------------------------------------------------------
# parse_dialog_roles
# ---------------------------------------------------------------------------

class TestParseDialogRoles:
    def test_none_returns_empty_tuple(self):
        dialog, client, oper, found = parse_dialog_roles(None)
        assert dialog == ""
        assert client == ""
        assert oper == ""
        assert found is False

    def test_empty_string(self):
        dialog, client, oper, found = parse_dialog_roles("")
        assert dialog == ""
        assert found is False

    def test_roles_found_flag(self):
        text = "CLIENT: У меня проблема с картой и переводами\nOPERATOR: Сейчас разберёмся с этим вопросом"
        _, _, _, found = parse_dialog_roles(text)
        assert found is True

    def test_no_roles_found(self):
        text = "Просто обычный текст без ролей и меток"
        _, _, _, found = parse_dialog_roles(text)
        assert found is False

    def test_client_text_extracted(self):
        text = "CLIENT: Хочу закрыть вклад досрочно и получить деньги\nOPERATOR: Уточните пожалуйста данные"
        _, client, _, _ = parse_dialog_roles(text)
        assert "вклад" in client

    def test_operator_text_extracted(self):
        text = "CLIENT: У меня вопрос по переводу денег срочно\nOPERATOR: Произошла техническая ошибка в системе"
        _, _, oper, _ = parse_dialog_roles(text)
        assert "ошибка" in oper

    def test_chatbot_ignored_by_default(self):
        text = "CHATBOT: Добро пожаловать в чат поддержки\nCLIENT: Я хочу вернуть деньги за покупку"
        dialog, _, _, _ = parse_dialog_roles(text, ignore_chatbot=True)
        assert "Добро пожаловать" not in dialog

    def test_chatbot_included_when_not_ignored(self):
        text = "CHATBOT: Добро пожаловать в чат поддержки банка\nCLIENT: Нужна помощь с кредитной картой"
        dialog, _, _, _ = parse_dialog_roles(text, ignore_chatbot=False)
        assert "Добро пожаловать" in dialog

    def test_cyrillic_role_client(self):
        text = "КЛИЕНТ: Не могу войти в личный кабинет банка\nОПЕРАТОР: Уточните пожалуйста ваш логин"
        _, client, oper, found = parse_dialog_roles(text)
        assert found is True
        assert "кабинет" in client

    def test_cyrillic_role_operator(self):
        text = "КЛИЕНТ: Блокировка карты произошла без причины\nОПЕРАТОР: Карта заблокирована службой безопасности"
        _, _, oper, _ = parse_dialog_roles(text)
        assert "безопасности" in oper

    def test_dialog_clean_contains_both_roles(self):
        text = "CLIENT: У меня вопрос по балансу карты\nOPERATOR: Баланс по вашей карте доступен в приложении"
        dialog, _, _, _ = parse_dialog_roles(text)
        assert "CLIENT:" in dialog
        assert "OPERATOR:" in dialog

    def test_noise_utterance_filtered_from_client(self):
        # Short filler phrases like «спасибо большое» are noise for the client
        text = "CLIENT: Спасибо большое!\nCLIENT: Мне нужно срочно получить выписку по счёту"
        _, client, _, _ = parse_dialog_roles(text)
        # The noise line should be filtered, the meaningful line should remain
        assert "выписку" in client

    def test_noise_utterance_filtered_from_operator(self):
        # "Здравствуйте я вас слушаю." is a noise greeting
        text = "OPERATOR: Я вас слушаю.\nCLIENT: Потеряли платёж за перевод средств\nOPERATOR: Произошёл технический сбой в обработке"
        _, _, oper, _ = parse_dialog_roles(text)
        # The meaningful operator utterance should survive
        assert "сбой" in oper

    def test_multiline_client_utterance_merged(self):
        text = (
            "CLIENT: Мне нужна помощь с переводом средств\n"
            "продолжение реплики клиента\n"
            "OPERATOR: Помогу разобраться с вашим запросом сейчас"
        )
        _, client, _, _ = parse_dialog_roles(text)
        # Both lines should be merged into client text
        assert "переводом" in client
        assert "продолжение" in client

    def test_returns_normalized_text_when_no_roles(self):
        text = "Просто текст ë с тире \u2014 и пробелами"
        dialog, client, oper, found = parse_dialog_roles(text)
        assert found is False
        assert dialog  # Should be non-empty normalized text
        assert client == ""
        assert oper == ""

    def test_system_role_chatbot_cyrillic(self):
        text = "БОТ: Ваш запрос принят системой\nCLIENT: Хочу поговорить с оператором банка"
        _, _, _, found = parse_dialog_roles(text)
        assert found is True

    def test_bot_role_system_keyword(self):
        text = "SYSTEM: Соединяю с оператором\nCLIENT: Мне нужна помощь по кредиту срочно"
        _, _, _, found = parse_dialog_roles(text)
        assert found is True
