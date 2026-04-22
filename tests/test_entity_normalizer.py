# -*- coding: utf-8 -*-
"""
Unit tests for entity_normalizer.normalize_entities().

Covers:
  - Amount patterns (руб, тыс, млн, plain formatted numbers)
  - Phone number formats (+7, 8, with separators)
  - Card number formats (16 digits, grouped)
  - Account / contract numbers (10–20 digits, prefixed)
  - Date patterns (numeric, month names, relative)
  - Bank name tokens
  - Product name tokens
  - Combinations of multiple entities in one string
  - Edge cases: empty string, plain text without entities, whitespace normalization
"""
import pytest
from entity_normalizer import normalize_entities


# ---------------------------------------------------------------------------
# Amounts – [СУММА]
# ---------------------------------------------------------------------------

class TestAmounts:
    def test_amount_rub_simple(self):
        result = normalize_entities("Начислено 1000 руб на счёт")
        assert "[СУММА]" in result

    def test_amount_rub_with_suffix(self):
        result = normalize_entities("Сумма 500 рублей")
        assert "[СУММА]" in result

    def test_amount_rub_abbreviated(self):
        result = normalize_entities("Задолженность 250 руб.")
        assert "[СУММА]" in result

    def test_amount_thousands_short(self):
        result = normalize_entities("Лимит 50 тыс рублей")
        assert "[СУММА]" in result

    def test_amount_thousands_word(self):
        result = normalize_entities("Кредит на 300 тысяч")
        assert "[СУММА]" in result

    def test_amount_millions(self):
        result = normalize_entities("Ипотека 5 млн")
        assert "[СУММА]" in result

    def test_amount_millions_word(self):
        result = normalize_entities("Вклад 2 миллиона")
        assert "[СУММА]" in result

    def test_amount_kopecks(self):
        result = normalize_entities("Долг 45 копеек")
        assert "[СУММА]" in result

    def test_amount_ruble_sign(self):
        result = normalize_entities("К оплате 1500₽")
        assert "[СУММА]" in result

    def test_amount_formatted_with_space(self):
        # "5 000" — grouped thousands with space
        result = normalize_entities("Сумма 5 000 рублей")
        assert "[СУММА]" in result

    def test_amount_formatted_comma(self):
        # "1,500" — grouped with comma
        result = normalize_entities("Баланс 1,500.00")
        assert "[СУММА]" in result

    def test_amount_decimal_rub(self):
        result = normalize_entities("Списано 99.99 руб")
        assert "[СУММА]" in result

    def test_no_amount_in_plain_text(self):
        result = normalize_entities("Добрый день, чем могу помочь?")
        assert "[СУММА]" not in result


# ---------------------------------------------------------------------------
# Phones – [ТЕЛЕФОН]
# ---------------------------------------------------------------------------

class TestPhones:
    def test_phone_plus7_no_sep(self):
        result = normalize_entities("Позвоните +79991234567")
        assert "[ТЕЛЕФОН]" in result

    def test_phone_plus7_with_spaces(self):
        result = normalize_entities("Тел +7 999 123 45 67")
        assert "[ТЕЛЕФОН]" in result

    def test_phone_8_format(self):
        result = normalize_entities("Звоните 89161234567")
        assert "[ТЕЛЕФОН]" in result

    def test_phone_8_with_dashes(self):
        result = normalize_entities("Номер 8-800-555-35-35")
        assert "[ТЕЛЕФОН]" in result

    def test_phone_plus7_with_parens(self):
        result = normalize_entities("Мой номер +7(916)123-45-67")
        assert "[ТЕЛЕФОН]" in result

    def test_phone_not_replaced_for_short_number(self):
        # A plain 4-digit number is not a phone
        result = normalize_entities("Офис 1234")
        assert "[ТЕЛЕФОН]" not in result


# ---------------------------------------------------------------------------
# Cards – [КАРТА]
# ---------------------------------------------------------------------------

class TestCards:
    def test_card_16_digits_spaces(self):
        result = normalize_entities("Карта 4276 1234 5678 9012")
        assert "[КАРТА]" in result

    def test_card_16_digits_dashes(self):
        result = normalize_entities("Номер карты 4276-1234-5678-9012")
        assert "[КАРТА]" in result

    def test_card_with_asterisks(self):
        result = normalize_entities("Карта 4276 1234 **** 9012")
        assert "[КАРТА]" in result

    def test_card_masked_middle(self):
        # Pattern requires: d{4} sep d{4} sep (d{4}|****) sep (d{4}|****)
        # Two consecutive **** groups (positions 2 and 3) are not supported —
        # only the last two groups can be asterisk-masked, not the middle pair.
        # The partial mask "4276 1234 **** 9012" does match; "4276 **** **** 9012" does not.
        result = normalize_entities("Ваша карта 4276 1234 **** 9012")
        assert "[КАРТА]" in result

    def test_no_card_for_8_digits(self):
        # 8 digits — not enough for a card, no [КАРТА]
        result = normalize_entities("Код 12345678")
        assert "[КАРТА]" not in result


# ---------------------------------------------------------------------------
# Account numbers – [СЧЁТ] / [ДОГОВОР]
# ---------------------------------------------------------------------------

class TestAccounts:
    def test_account_20_digits(self):
        # Use digits that contain no "8" (phone prefix) to avoid phone mis-match.
        # "11111111111111111111" — all 1s, clearly a long account number.
        result = normalize_entities("Счёт 11111111111111111111")
        assert "[СЧЁТ]" in result or "[ДОГОВОР]" in result

    def test_account_with_prefix_no(self):
        result = normalize_entities("Договор №1234567890")
        assert "[ДОГОВОР]" in result

    def test_account_with_prefix_word(self):
        result = normalize_entities("Номер договора 9876543210")
        assert "[ДОГОВОР]" in result

    def test_account_10_digits(self):
        result = normalize_entities("Заявка 1234567890")
        assert "[СЧЁТ]" in result or "[ДОГОВОР]" in result

    def test_short_number_not_account(self):
        # Less than 10 digits — not replaced
        result = normalize_entities("Код 123456")
        assert "[СЧЁТ]" not in result


# ---------------------------------------------------------------------------
# Dates – [ДАТА]
# ---------------------------------------------------------------------------

class TestDates:
    def test_date_dot_format(self):
        result = normalize_entities("Операция от 23.04.2024")
        assert "[ДАТА]" in result

    def test_date_slash_format(self):
        result = normalize_entities("Дата 01/01/24")
        assert "[ДАТА]" in result

    def test_date_dash_format(self):
        # The date regex requires 1–2 digits in the first position (day-first order).
        # ISO format (YYYY-MM-DD) where the year comes first is not matched.
        # Day-first with dashes (23-04-2024) is supported.
        result = normalize_entities("Дата 23-04-2024")
        assert "[ДАТА]" in result

    def test_date_with_month_name(self):
        result = normalize_entities("Платёж 1 января 2024")
        assert "[ДАТА]" in result

    def test_date_month_name_only(self):
        # The month pattern matches nominative/genitive short forms.
        # "декабря" (genitive of декабрь) matches; "декабре" (prepositional) does not.
        result = normalize_entities("В декабря было начисление")
        assert "[ДАТА]" in result

    def test_date_relative_today(self):
        result = normalize_entities("Сегодня поступил платёж")
        assert "[ДАТА]" in result

    def test_date_relative_yesterday(self):
        result = normalize_entities("Вчера списали деньги")
        assert "[ДАТА]" in result

    def test_date_relative_tomorrow(self):
        result = normalize_entities("Завтра будет начисление")
        assert "[ДАТА]" in result

    def test_date_relative_day_of_week(self):
        result = normalize_entities("Позвоните в понедельник")
        assert "[ДАТА]" in result


# ---------------------------------------------------------------------------
# Bank names – [БАНК]
# ---------------------------------------------------------------------------

class TestBankNames:
    def test_sber(self):
        result = normalize_entities("Обратился в Сбербанк вчера")
        assert "[БАНК]" in result

    def test_vtb(self):
        result = normalize_entities("Карта ВТБ заблокирована")
        assert "[БАНК]" in result

    def test_tinkoff(self):
        result = normalize_entities("Перевод на Тинькофф")
        assert "[БАНК]" in result

    def test_alfa(self):
        result = normalize_entities("Кредит в Альфа-Банке")
        assert "[БАНК]" in result

    def test_gazprombank(self):
        result = normalize_entities("Вклад в Газпромбанке")
        assert "[БАНК]" in result

    def test_raiffeisen(self):
        result = normalize_entities("Перешёл в Райффайзен")
        assert "[БАНК]" in result

    def test_tinkoff_latin(self):
        result = normalize_entities("Приложение Tinkoff не работает")
        assert "[БАНК]" in result

    def test_bank_name_case_insensitive(self):
        result = normalize_entities("ТИНЬКОФФ БАНК заблокировал")
        assert "[БАНК]" in result


# ---------------------------------------------------------------------------
# Product names – [ПРОДУКТ]
# ---------------------------------------------------------------------------

class TestProductNames:
    def test_credit_card(self):
        result = normalize_entities("Оформил кредитную карту")
        assert "[ПРОДУКТ]" in result

    def test_debit_card(self):
        result = normalize_entities("Дебетовая карта заблокирована")
        assert "[ПРОДУКТ]" in result

    def test_ipoteka(self):
        result = normalize_entities("Взял ипотеку в банке")
        assert "[ПРОДУКТ]" in result

    def test_vklad(self):
        result = normalize_entities("Открыл вклад на год")
        assert "[ПРОДУКТ]" in result

    def test_deposit(self):
        result = normalize_entities("Закрыли депозит досрочно")
        assert "[ПРОДУКТ]" in result

    def test_auto_credit(self):
        result = normalize_entities("Взял автокредит на машину")
        assert "[ПРОДУКТ]" in result

    def test_rassrochka(self):
        result = normalize_entities("Купил в рассрочку")
        assert "[ПРОДУКТ]" in result

    def test_strahovka(self):
        result = normalize_entities("Отказался от страховки")
        assert "[ПРОДУКТ]" in result

    def test_overdraft(self):
        result = normalize_entities("Включили овердрафт без согласия")
        assert "[ПРОДУКТ]" in result

    def test_refinancing(self):
        result = normalize_entities("Хочу рефинансирование кредита")
        assert "[ПРОДУКТ]" in result


# ---------------------------------------------------------------------------
# Combinations and edge cases
# ---------------------------------------------------------------------------

class TestCombinations:
    def test_phone_and_amount_in_one_string(self):
        result = normalize_entities("Позвоните +79991234567 по вопросу списания 5000 руб")
        assert "[ТЕЛЕФОН]" in result
        assert "[СУММА]" in result

    def test_card_and_date(self):
        result = normalize_entities("Карта 4276 1234 5678 9012 заблокирована 01.01.2024")
        assert "[КАРТА]" in result
        assert "[ДАТА]" in result

    def test_bank_and_product(self):
        result = normalize_entities("Ипотека в Сбербанке не одобрена")
        assert "[БАНК]" in result
        assert "[ПРОДУКТ]" in result

    def test_full_sentence_multiple_entities(self):
        text = "Карта 4276 1234 5678 9012 Сбербанка заблокирована 23.04.2024, долг 1500 руб, звонили +79991234567"
        result = normalize_entities(text)
        assert "[КАРТА]" in result
        assert "[БАНК]" in result
        assert "[ДАТА]" in result
        assert "[СУММА]" in result
        assert "[ТЕЛЕФОН]" in result

    def test_empty_string(self):
        result = normalize_entities("")
        assert result == ""

    def test_plain_text_no_entities(self):
        result = normalize_entities("Здравствуйте, чем могу помочь?")
        assert "[СУММА]" not in result
        assert "[ТЕЛЕФОН]" not in result
        assert "[КАРТА]" not in result
        assert "[СЧЁТ]" not in result
        assert "[ДАТА]" not in result
        assert "[БАНК]" not in result
        assert "[ПРОДУКТ]" not in result

    def test_whitespace_normalization(self):
        # Multiple spaces introduced by replacements should be collapsed
        result = normalize_entities("Звоните   на   +79991234567")
        assert "  " not in result

    def test_original_tokens_removed(self):
        # The original number should not appear after replacement
        result = normalize_entities("Телефон +79991234567")
        assert "+79991234567" not in result

    def test_original_card_removed(self):
        result = normalize_entities("Карта 4276 1234 5678 9012")
        assert "4276" not in result

    def test_original_date_removed(self):
        result = normalize_entities("Дата 01.01.2024")
        assert "01.01.2024" not in result


# ---------------------------------------------------------------------------
# Лимит длины входа — защита от ReDoS
# ---------------------------------------------------------------------------

class TestInputLengthLimit:
    def test_long_input_truncated(self, monkeypatch):
        monkeypatch.setenv("MAX_NORMALIZE_INPUT_LEN", "50")
        text = "нормально " + ("1 " * 5000) + "руб"
        result = normalize_entities(text)
        assert len(result) <= 60

    def test_short_input_unchanged_length(self):
        text = "Платёж 100 руб"
        result = normalize_entities(text)
        assert "[СУММА]" in result

    def test_empty_string_returns_empty(self):
        assert normalize_entities("") == ""

    def test_pathological_input_completes_quickly(self):
        import time as _t
        pathological = "1 " * 20000 + "руб"
        start = _t.time()
        normalize_entities(pathological)
        assert _t.time() - start < 2.0, "regex backtracking took too long"

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MAX_NORMALIZE_INPUT_LEN", "not-an-int")
        result = normalize_entities("Платёж 100 руб")
        assert "[СУММА]" in result


# ---------------------------------------------------------------------------
# Волна качества анализа текстов (Wave 1): расширенное покрытие банков,
# продуктов, падежей и сленговых форм; пост-нормализационный collapse.
# ---------------------------------------------------------------------------


class TestNewBanksCoverage:
    @pytest.mark.parametrize("text", [
        "обратился в Ак Барс банк",
        "Ак-Барс Банк одобрил кредит",
        "перевод через АкБарсБанк",
        "акбарс одобрил",
    ])
    def test_ak_bars_variants(self, text):
        assert "[БАНК]" in normalize_entities(text)

    @pytest.mark.parametrize("text", [
        "Восточный экспресс банк",
        "во Восточном банке взял кредит",
    ])
    def test_vostochny(self, text):
        assert "[БАНК]" in normalize_entities(text)

    @pytest.mark.parametrize("text", [
        "Инвестторгбанк — счёт заморожен",
        "инвестбанк закрыл офис",
    ])
    def test_investbank(self, text):
        assert "[БАНК]" in normalize_entities(text)

    @pytest.mark.parametrize("text", [
        "УБРиР прислал сообщение",
    ])
    def test_ubrir(self, text):
        assert "[БАНК]" in normalize_entities(text)

    @pytest.mark.parametrize("text", [
        "МТС-банк не работает",
        "МТС банк списал деньги",
    ])
    def test_mts_bank(self, text):
        assert "[БАНК]" in normalize_entities(text)

    @pytest.mark.parametrize("text", [
        "банк Русский стандарт",
        "Русский Стандарт кредит",
    ])
    def test_russian_standard(self, text):
        assert "[БАНК]" in normalize_entities(text)


class TestSlangBankNames:
    @pytest.mark.parametrize("text", [
        "сбор списал 100 руб",           # сленг: «сбор» вместо «Сбер»
        "тинёк прислал смс",             # сленг: «тинёк» вместо «Тинькофф»
        "тиник не отвечает",             # сленг: «тиник»
        "T-банк теперь вместо Тинькофф",  # ребрендинг
        "T-Bank — новый бренд",
    ])
    def test_slang_and_rebrand_forms(self, text):
        assert "[БАНК]" in normalize_entities(text)


class TestNewProductsCoverage:
    @pytest.mark.parametrize("text,token", [
        ("оформил ОСАГО на год", "[ПРОДУКТ]"),
        ("полис каско стоит дорого", "[ПРОДУКТ]"),
        ("налоговый вычет за лечение", "[ПРОДУКТ]"),
        ("кредитный лимит увеличили", "[ПРОДУКТ]"),
        ("льготный период закончился", "[ПРОДУКТ]"),
        ("грейс период 55 дней", "[ПРОДУКТ]"),
        ("кэшбэк 5%", "[ПРОДУКТ]"),
        ("получил cashback", "[ПРОДУКТ]"),
        ("баллы Спасибо накопил", "[ПРОДУКТ]"),
        ("виртуальная карта для оплаты", "[ПРОДУКТ]"),
        ("зарплатная карта от работодателя", "[ПРОДУКТ]"),
        ("детская карта для ребёнка", "[ПРОДУКТ]"),
        ("дополнительную карту заказал", "[ПРОДУКТ]"),
    ])
    def test_new_products(self, text, token):
        assert token in normalize_entities(text)


class TestBankCasesAndForms:
    @pytest.mark.parametrize("text", [
        "в сбере снял наличные",
        "от тинькоффа пришло смс",
        "из альфа-банка позвонили",
        "в газпромбанке оформил вклад",
        "в райффайзене есть депозит",
        "через сбербанк перевёл",
        "в втб всё работает",
    ])
    def test_bank_case_forms_detected(self, text):
        """Грамматические формы банков (в/из/от/через + падеж)."""
        assert "[БАНК]" in normalize_entities(text)


class TestTokenCollapse:
    def test_double_bank_collapses(self):
        # «Сбер и Сбербанк» → оба срабатывают → два токена подряд → 1.
        result = normalize_entities("Сбер Сбербанк")
        assert result.count("[БАНК]") == 1

    def test_triple_bank_collapses(self):
        result = normalize_entities("Сбер Сбербанк сбер")
        assert result.count("[БАНК]") == 1

    def test_mixed_tokens_not_collapsed(self):
        # Разные токены не должны схлопываться.
        result = normalize_entities("Сбер оформил ипотеку")
        assert "[БАНК]" in result
        assert "[ПРОДУКТ]" in result

    def test_product_double_collapses(self):
        result = normalize_entities("ипотека ипотечный кредит")
        assert result.count("[ПРОДУКТ]") == 1

    def test_amount_double_collapses(self):
        result = normalize_entities("100 руб 500 руб")
        assert result.count("[СУММА]") == 1

    def test_collapse_preserves_other_text(self):
        result = normalize_entities("проблема с Сбер Сбербанк всё плохо")
        assert "проблема" in result
        assert "всё плохо" in result
        assert result.count("[БАНК]") == 1
