# -*- coding: utf-8 -*-
"""
config/exclusions.py — управление пользовательскими словами-исключениями.

Хранит три категории исключений в user_exclusions.json:
  • stop_words    — добавляются к RUSSIAN_STOP_WORDS (исключаются из словаря TF-IDF)
  • noise_tokens  — добавляются к NOISE_TOKENS (отдельные токены-шум)
  • noise_phrases — добавляются к NOISE_PHRASES (фразы, удаляемые до токенизации)

Поддерживает миграцию из legacy custom_stop_words.json (только stop_words).
Кэширует загруженные данные — файл читается один раз за сессию.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Типы и внутренний кэш
# ---------------------------------------------------------------------------
ExclusionsDict = Dict[str, List[str]]

_cache: Optional[ExclusionsDict] = None


def _empty() -> ExclusionsDict:
    return {"stop_words": [], "noise_tokens": [], "noise_phrases": []}


def _get_files() -> tuple[Path, Path]:
    """Возвращает (USER_EXCLUSIONS_FILE, CUSTOM_STOP_WORDS_FILE)."""
    from config.paths import USER_EXCLUSIONS_FILE, CUSTOM_STOP_WORDS_FILE
    return USER_EXCLUSIONS_FILE, CUSTOM_STOP_WORDS_FILE


# ---------------------------------------------------------------------------
# Загрузка / сохранение
# ---------------------------------------------------------------------------

def load_exclusions() -> ExclusionsDict:
    """Загружает исключения из файла. Кэширует результат.

    При первом запуске пытается мигрировать из legacy custom_stop_words.json.
    """
    global _cache
    if _cache is not None:
        return _cache

    exc_file, legacy_file = _get_files()

    # Основной файл
    if exc_file.exists():
        try:
            data = json.loads(exc_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                result = _empty()
                for key in result:
                    if isinstance(data.get(key), list):
                        result[key] = [
                            str(w).strip() for w in data[key] if str(w).strip()
                        ]
                _cache = result
                return _cache
        except Exception:
            pass

    # Миграция из legacy custom_stop_words.json
    if legacy_file.exists():
        try:
            data = json.loads(legacy_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                words = [str(w).strip() for w in data if str(w).strip()]
                result = _empty()
                result["stop_words"] = words
                save_exclusions(result)  # сохраняем в новый формат
                _cache = result
                return _cache
        except Exception:
            pass

    _cache = _empty()
    return _cache


def save_exclusions(data: ExclusionsDict) -> None:
    """Сохраняет исключения в файл и обновляет кэш."""
    global _cache
    _cache = data
    exc_file, _ = _get_files()
    try:
        exc_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def invalidate_cache() -> None:
    """Сбрасывает кэш — следующий load_exclusions() перечитает файл."""
    global _cache
    _cache = None


# ---------------------------------------------------------------------------
# Эффективные наборы (built-in + user)
# ---------------------------------------------------------------------------

def get_effective_stop_words(base: set) -> set:
    """Возвращает base | пользовательские стоп-слова."""
    user = load_exclusions().get("stop_words", [])
    return base | {w.lower() for w in user if w}


def get_effective_noise_tokens(base: set) -> set:
    """Возвращает base | пользовательские токены-шум."""
    user = load_exclusions().get("noise_tokens", [])
    return base | {w.lower() for w in user if w}


def get_effective_noise_phrases(base: list) -> list:
    """Возвращает пользовательские фразы + built-in фразы.

    Пользовательские фразы идут ПЕРВЫМИ — они матчатся раньше коротких built-in.
    """
    user = load_exclusions().get("noise_phrases", [])
    return [p.lower() for p in user if p.strip()] + list(base)
