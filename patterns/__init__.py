# -*- coding: utf-8 -*-
"""
patterns — пакет текстовых паттернов и шумовых фильтров.

Структура:
  dialog.py     — ROLE, DROP_LINES, HTML_TAG, MASK_RE, EMOJI_RE, SECTION_PREFIX, ...
  noise.py      — OPERATOR_NOISE_RE, CLIENT_NOISE_RE, NOISE_TOKENS, NOISE_PHRASES
  stop_words.py — RUSSIAN_STOP_WORDS

Реэкспортирует всё для обратной совместимости:
  from patterns import NOISE_TOKENS   # новый стиль
  from constants import NOISE_TOKENS  # по-прежнему работает через constants.py-шим
"""
from __future__ import annotations

from patterns.dialog import (  # noqa: F401
    ROLE,
    DROP_LINES,
    HTML_TAG,
    PLAIN_HTML_WORDS_RE,
    MASK_RE,
    EMOJI_RE,
    ANSWER_HEADER_RE,
    DIALOG_MIN_WORDS,
    SECTION_PREFIX,
)

from patterns.noise import (  # noqa: F401
    OPERATOR_NOISE_RE,
    CLIENT_NOISE_RE,
    NOISE_TOKENS,
    NOISE_PHRASES,
)

from patterns.stop_words import RUSSIAN_STOP_WORDS  # noqa: F401
