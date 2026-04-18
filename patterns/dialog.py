# -*- coding: utf-8 -*-
"""
patterns/dialog.py — регулярные выражения для разбора диалогов.

Содержит:
  ROLE             — паттерны ролей (CLIENT, OPERATOR, CHATBOT)
  DROP_LINES       — системные строки, которые нужно отбрасывать целиком
  HTML_TAG         — HTML-теги в тексте
  PLAIN_HTML_WORDS_RE — HTML-артефакты в plain-text (div, br, nbsp, class=...)
  MASK_RE          — маски персональных данных (!fio!, !phone!, ...)
  EMOJI_RE         — emoji и технические Unicode-символы
  ANSWER_HEADER_RE — шаблонный заголовок письма-ответа банка
  DIALOG_MIN_WORDS — минимальная длина реплики в словах
  SECTION_PREFIX   — теги секций для feature-текста ([DESC], [CLIENT], ...)
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Роли участников диалога
# ---------------------------------------------------------------------------
ROLE = {
    "CLIENT":   re.compile(r"^\s*(CLIENT|КЛИЕНТ)\s*:\s*(.*)$", re.IGNORECASE),
    "OPERATOR": re.compile(r"^\s*(OPERATOR|ОПЕРАТОР)\s*:\s*(.*)$", re.IGNORECASE),
    "CHATBOT":  re.compile(r"^\s*(CHATBOT|БОТ|SYSTEM|СИСТЕМА)\s*:\s*(.*)$", re.IGNORECASE),
}

# ---------------------------------------------------------------------------
# Системные строки — отбрасываются полностью до обработки ролей
# ---------------------------------------------------------------------------
DROP_LINES = [
    re.compile(r"^\s*sdk_request\s*$", re.IGNORECASE),
    re.compile(r"^\s*оператор подключается.*$", re.IGNORECASE),
    re.compile(r"^\s*после диалога оцените.*$", re.IGNORECASE),
    re.compile(r"^\s*оператор подключится в течение.*$", re.IGNORECASE),
    re.compile(r"^\s*уже переключили скоро вам ответят.*$", re.IGNORECASE),
    re.compile(r"^\s*вас проконсультирует.*$", re.IGNORECASE),
    re.compile(r"^\s*оператор\s+\S+\s+отозван\s*$", re.IGNORECASE),
    # Системные сообщения очереди / ожидания (чат-бот)
    re.compile(r"^\s*из-за\s+временной\s+нагрузки\b", re.IGNORECASE),
    # Артефакты мессенджера (стикеры Telegram/Сбербанк-чат)
    re.compile(r"^\s*стикер\s+pack=\d+\s+id=\d+\s*$", re.IGNORECASE),
    # GigaVoice IVR: соединение, запись, оценка специалиста
    re.compile(r"^\s*gigavoice\b", re.IGNORECASE),
    re.compile(r"^\s*соединяю\s+(вас\s+)?со\s+специалистом\b", re.IGNORECASE),
    re.compile(r"\bразговор\b.{0,25}\bзаписывается\b", re.IGNORECASE),
    re.compile(r"^\s*оставайтесь\s+на\s+линии\b", re.IGNORECASE),
    re.compile(r"^\s*после\s+завершения\s+разговора\b", re.IGNORECASE),
    re.compile(r"^\s*оцените\s+специалиста\.?$", re.IGNORECASE),
    # Системное сообщение чата о возврате обращения в работу
    re.compile(r"^\s*ваше\s+обращение\s+возвращено\s+в\s+работу\b", re.IGNORECASE),
    # Статусные строки чата (дождитесь оператора, оценка)
    re.compile(r"^\s*после\s+диалога\s+оцените\b", re.IGNORECASE),
    re.compile(r"^\s*переведи\s+на\s+оператора\.?$", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# HTML и спецсимволы
# ---------------------------------------------------------------------------
HTML_TAG = re.compile(r"<[^>]+>")

# Артефакты текстового HTML (div, br, nbsp, class=...) из писем Outlook/Word
PLAIN_HTML_WORDS_RE = re.compile(
    r"\b(?:div|br|nbsp|msonormal)\b|class=[^ ]*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Маски персональных данных и спецсимволы
# ---------------------------------------------------------------------------

# !fio!, !num!, !phone!, !data!, !email!, !SW! и любые другие !xxx!
MASK_RE = re.compile(r"![a-zA-Z0-9_]+!", re.IGNORECASE)

# Emoji и технические Unicode-символы без смысловой нагрузки
EMOJI_RE = re.compile(
    r"["
    r"\u2300-\u27BF"           # Misc Technical, Symbols, Dingbats (⏳✔ и т.д.)
    r"\u2B00-\u2BFF"           # Misc Symbols and Arrows (⭐ и т.д.)
    r"\U0001F000-\U0001FFFF"   # Emoji supplementary planes
    r"\uFE00-\uFE0F"           # Variation Selectors
    r"\uFFFD"                  # Replacement Character (мусор из кодировки)
    r"]",
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Шаблонный заголовок письма-ответа банка
# ---------------------------------------------------------------------------
ANSWER_HEADER_RE = re.compile(
    r"мы\s+рассмотрели\s+ваше\s+обращение.{0,60}?\d{2}\.\d{2}\.\d{4}\.",
    re.IGNORECASE,
)

# Минимальное кол-во слов в реплике (после удаления масок).
DIALOG_MIN_WORDS = 3

# ---------------------------------------------------------------------------
# Префиксы секций для feature-текста
# ---------------------------------------------------------------------------
SECTION_PREFIX = {
    "CHANNEL":      "[CHANNEL]",
    "DESC":         "[DESC]",
    "CLIENT":       "[CLIENT]",
    "OPERATOR":     "[OPERATOR]",
    "SUMMARY":      "[SUMMARY]",
    "ANSWER_SHORT": "[ANSWER_SHORT]",
    "ANSWER_FULL":  "[ANSWER_FULL]",
}
