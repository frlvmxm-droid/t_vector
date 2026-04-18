# -*- coding: utf-8 -*-
"""
Утилиты обработки текста: очистка HTML, нормализация, парсинг ролей диалога.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from constants import (
    DROP_LINES, HTML_TAG, PLAIN_HTML_WORDS_RE, ROLE,
    MASK_RE, EMOJI_RE, ANSWER_HEADER_RE,
    OPERATOR_NOISE_RE, CLIENT_NOISE_RE,
    DIALOG_MIN_WORDS,
)


# ---------------------------------------------------------------------------
# Общая таблица замен спецсимволов
# ---------------------------------------------------------------------------
# Используется в normalize_text и clean_answer_text — единственное место
# для добавления новых символов-замен.
# str.maketrans + translate — один проход по строке вместо 7 последовательных replace().
_SYMBOL_TABLE = str.maketrans({
    "\u00a0": " ",   # non-breaking space
    "\ufffd": " ",   # replacement character (мусор кодировки)
    "\u00eb": "е",   # ë Latin → е (артефакт OCR / кодировки)
    "\u0451": "е",   # ё Cyrillic → е (явная нормализация перед лемматизацией)
    "\u0401": "Е",   # Ё Cyrillic → Е
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2022": " ",   # bullet •
    "\u2192": " ",   # arrow →
})


def _normalize_symbols(s: str) -> str:
    """Заменяет спецсимволы на нейтральные ASCII-эквиваленты (один проход)."""
    return s.translate(_SYMBOL_TABLE)


# ---------------------------------------------------------------------------
# Латиница-в-кириллице (visual confusables)
# ---------------------------------------------------------------------------
# «Сbер», «Banк», «Альфа-Банк» с латинской 'B' — артефакт набора на
# неправильной раскладке или копипасты из смешанных источников. Глобальная
# замена Latin→Cyrillic сломала бы английские слова («Visa», «Mastercard»),
# поэтому чиним только **слова, где Cyrillic и Latin буквы соседствуют**.
#
# Карта основана на Unicode confusables (TR 39 skeleton) — визуально
# неразличимые пары из ASCII и базовой кириллицы.
_LAT_TO_CYR = str.maketrans({
    "a": "а", "A": "А",
    "B": "В",
    "c": "с", "C": "С",
    "e": "е", "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О", "o": "о",
    "p": "р", "P": "Р",
    "T": "Т",
    "x": "х", "X": "Х",
    "y": "у", "Y": "У",
})

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# \w без re.UNICODE всё равно покрывает Unicode в Python 3, но явный класс
# проще читается и исключает цифры/знаки.
_MIXED_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def _normalize_confusables(s: str) -> str:
    """В словах, где смешаны кириллица и латиница, латинские буквы
    заменяются на визуально идентичные кириллические.

    Пример: «Сbер» → «Сбер», «Alfa Банк» → «Alfa Банк» (оба слова
    однородные — не трогаем).
    """
    if not s:
        return s

    def _fix(match: re.Match[str]) -> str:
        word = match.group(0)
        if _CYRILLIC_RE.search(word) and _LATIN_RE.search(word):
            return word.translate(_LAT_TO_CYR)
        return word

    return _MIXED_WORD_RE.sub(_fix, s)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def clean_masks(s: str) -> str:
    """Удаляет маски !xxx! и нормализует образовавшиеся пробелы."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", MASK_RE.sub(" ", s)).strip()


def _is_noise_utt(utt: str, patterns: list) -> bool:
    """
    True если реплика является шумом.

    Порядок проверок (от дешёвых к дорогим):
      1. Пустая строка после удаления масок
      2. Длина < DIALOG_MIN_WORDS — односложные/двусложные ответы на уточнения
         («ваш», «в москве», «две двадцать», «жду», «до свидания» и т.п.)
      3. Совпадение с явными паттернами из списка (прощания, шаблоны оператора)
    """
    cleaned = clean_masks(utt)
    if not cleaned:
        return True
    if len(cleaned.split()) < DIALOG_MIN_WORDS:
        return True
    return any(p.search(cleaned) for p in patterns)


# ---------------------------------------------------------------------------
# HTML / спецсимволы
# ---------------------------------------------------------------------------

def strip_html(x: Any) -> str:
    """Удаляет HTML-теги, текстовые HTML-артефакты и нормализует пробелы."""
    if x is None:
        return ""
    s = str(x)
    s = HTML_TAG.sub(" ", s)
    s = s.replace("&nbsp;", " ").replace("\u00a0", " ")
    s = PLAIN_HTML_WORDS_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Нормализация текста
# ---------------------------------------------------------------------------

def normalize_text(x: Any) -> str:
    """
    Приводит текст к единому виду:

    1. CR/CRLF → LF
    2. Маски !xxx! → пробел
    3. Emoji и мусорные Unicode-символы → пробел
    4. Неразрывные пробелы \\xa0 → пробел; ë → е; – / — → -
    5. Удаление строк из DROP_LINES
    6. Нормализация пробелов
    """
    if x is None:
        return ""
    s = str(x).replace("\r\n", "\n").replace("\r", "\n")

    # 1. Маски персональных данных
    s = MASK_RE.sub(" ", s)

    # 2. Emoji и технические символы
    s = EMOJI_RE.sub(" ", s)

    # 3. Нормализация спецсимволов
    s = _normalize_symbols(s)

    # 3a. Латинские буквы в русских словах (Сbер → Сбер)
    s = _normalize_confusables(s)

    # 4. Нормализуем пробелы ВНУТРИ строк (до разбивки, чтобы не мешать splitlines)
    s = re.sub(r"[ \t]+", " ", s)

    out: List[str] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(p.match(line) for p in DROP_LINES):
            continue
        out.append(line)

    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# Очистка текста ответа банка
# ---------------------------------------------------------------------------

def clean_answer_text(x: Any) -> str:
    """
    Специализированная очистка колонок ответа банка:

    1. Удаление HTML-тегов
    2. Удаление масок !xxx!
    3. Удаление шаблонного заголовка письма
       («мы рассмотрели ваше обращение … DD.MM.YYYY.»)
    4. Нормализация спецсимволов и пробелов
    """
    s = strip_html(x)
    if not s:
        return ""

    # Убираем маски
    s = MASK_RE.sub(" ", s)

    # Убираем стандартный шаблонный заголовок ответа
    s = ANSWER_HEADER_RE.sub(" ", s)

    # Нормализуем остальные спецсимволы
    s = _normalize_symbols(s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Парсинг ролей диалога
# ---------------------------------------------------------------------------

def parse_dialog_roles(
    text: Any,
    ignore_chatbot: bool = True,
) -> Tuple[str, str, str, bool]:
    """
    Парсит текст диалога, разделяя реплики по ролям CLIENT / OPERATOR / CHATBOT.

    Шумовые реплики (короткие филлеры, приветствия, тех. сообщения) автоматически
    исключаются из `client_text` и `operator_text` на основе OPERATOR_NOISE_RE
    и CLIENT_NOISE_RE.

    Returns:
        dialog_clean  — очищенный текст диалога (без игнорируемых ролей)
        client_text   — только содержательные реплики клиента
        operator_text — только содержательные реплики оператора
        roles_found   — True если хотя бы одна роль найдена

    Если роли не найдены — dialog_clean = нормализованный текст, остальные пустые.
    """
    s = normalize_text(text)
    if not s:
        return "", "", "", False

    kept: List[str] = []
    client: List[str] = []
    oper: List[str] = []
    roles_found = False
    last_role: Optional[str] = None

    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue

        # --- CHATBOT ---
        m = ROLE["CHATBOT"].match(line)
        if m:
            roles_found = True
            last_role = "CHATBOT"
            if not ignore_chatbot:
                utt = m.group(2).strip()
                if utt and not _is_noise_utt(utt, OPERATOR_NOISE_RE):
                    kept.append(utt)
            continue

        # --- CLIENT ---
        m = ROLE["CLIENT"].match(line)
        if m:
            roles_found = True
            utt = m.group(2).strip()
            last_role = "CLIENT"
            if utt and not _is_noise_utt(utt, CLIENT_NOISE_RE):
                client.append(utt)
                kept.append("CLIENT: " + utt)
            continue

        # --- OPERATOR ---
        m = ROLE["OPERATOR"].match(line)
        if m:
            roles_found = True
            utt = m.group(2).strip()
            last_role = "OPERATOR"
            if utt and not _is_noise_utt(utt, OPERATOR_NOISE_RE):
                oper.append(utt)
                kept.append("OPERATOR: " + utt)
            continue

        # --- Продолжение предыдущей реплики (строка без метки роли) ---
        if last_role == "CLIENT" and client:
            client[-1] += " " + line
            if kept and kept[-1].startswith("CLIENT: "):
                kept[-1] += " " + line
        elif last_role == "OPERATOR" and oper:
            oper[-1] += " " + line
            if kept and kept[-1].startswith("OPERATOR: "):
                kept[-1] += " " + line
        elif last_role == "CHATBOT":
            # Продолжение реплики чат-бота: добавляем только если бот не игнорируется
            if not ignore_chatbot:
                kept.append(line)
        else:
            kept.append(line)

    if roles_found:
        return "\n".join(kept).strip(), "\n".join(client).strip(), "\n".join(oper).strip(), True
    return s, "", "", False
