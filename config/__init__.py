# -*- coding: utf-8 -*-
"""
config — пакет конфигурации приложения.

Структура:
  user_config.py   — пользовательские настройки (колонки, пресеты, SBERT)
  paths.py         — пути к файлам и папкам (PyInstaller-aware)
  model_config.py  — ModelConfig dataclass, версия схемы, backward-compat
  exclusions.py    — управление пользовательскими словами-исключениями

Этот __init__.py реэкспортирует всё для обратной совместимости:
  from config import DEFAULT_COLS        # работает как раньше
  from config.exclusions import load_exclusions  # новый стиль
"""
from __future__ import annotations

# user_config
from config.user_config import (  # noqa: F401
    DEFAULT_COLS,
    PRESET_WEIGHTS,
    PRESET_WEIGHTS_DESC,
    PRESET_ALGO_PARAMS,
    SBERT_MODELS,
    SBERT_MODELS_LIST,
    SBERT_DEFAULT,
    DEBERTA_MODELS,
    DEBERTA_MODELS_LIST,
    NEURAL_MODELS,
    NEURAL_MODELS_LIST,
    DEFAULT_OTHER_LABEL,
    SETFIT_MODELS,
    SETFIT_MODELS_LIST,
    SETFIT_DEFAULT,
)

# paths
from config.paths import (  # noqa: F401
    APP_ROOT,
    MODEL_DIR,
    CLASS_DIR,
    CLUST_DIR,
    USER_EXCLUSIONS_FILE,
    CUSTOM_STOP_WORDS_FILE,
    now_stamp,
)

# model_config
from config.model_config import (  # noqa: F401
    CONFIG_VERSION,
    _CONFIG_FIELD_DEFAULTS,
    upgrade_config_dict,
    ModelConfig,
)

# exclusions (load/save/get helpers)
from config.exclusions import (  # noqa: F401
    load_exclusions,
    save_exclusions,
    invalidate_cache as invalidate_exclusions_cache,
    get_effective_stop_words,
    get_effective_noise_tokens,
    get_effective_noise_phrases,
)
