# -*- coding: utf-8 -*-
"""
config/paths.py — пути к папкам и файлам приложения (PyInstaller-aware).

В режиме frozen (.exe) APP_ROOT = папка рядом с .exe.
В режиме разработки APP_ROOT = корень проекта.
"""
from __future__ import annotations

import sys as _sys
from datetime import datetime
from pathlib import Path

if getattr(_sys, "frozen", False):
    # PyInstaller --onedir: .exe лежит в APP_ROOT
    APP_ROOT = Path(_sys.executable).resolve().parent
else:
    # Разработка: config/paths.py → поднимаемся на уровень выше
    APP_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR               = APP_ROOT / "model"
CLASS_DIR               = APP_ROOT / "classification"
CLUST_DIR               = APP_ROOT / "clustering"

# Пользовательские исключения (стоп-слова, токены, фразы)
USER_EXCLUSIONS_FILE    = APP_ROOT / "user_exclusions.json"
# Legacy: старый файл только с кастомными стоп-словами — используется для миграции
CUSTOM_STOP_WORDS_FILE  = APP_ROOT / "custom_stop_words.json"

for _d in (MODEL_DIR, CLASS_DIR, CLUST_DIR):
    _d.mkdir(exist_ok=True)


def now_stamp() -> str:
    """Возвращает метку времени для имён файлов: 20250101_120000."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
