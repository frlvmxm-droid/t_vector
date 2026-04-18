# -*- coding: utf-8 -*-
"""
constants.py — шим для обратной совместимости.

Весь код перенесён в пакеты:
  config/   — пути, ModelConfig, пользовательские настройки, исключения
  patterns/ — диалоговые паттерны, шум, стоп-слова

Этот файл реэкспортирует все имена, которые ранее были здесь определены,
чтобы существующий код (from constants import X) продолжал работать без изменений.
"""
from __future__ import annotations

# ── config ───────────────────────────────────────────────────────────────────
from config.user_config import (  # noqa: F401
    DEFAULT_COLS,
    PRESET_WEIGHTS, PRESET_WEIGHTS_DESC,
    SBERT_MODELS, SBERT_MODELS_LIST, SBERT_DEFAULT,
    DEFAULT_OTHER_LABEL,
    SETFIT_MODELS, SETFIT_MODELS_LIST, SETFIT_DEFAULT,
)
from config.paths import (  # noqa: F401
    APP_ROOT, MODEL_DIR, CLASS_DIR, CLUST_DIR,
    USER_EXCLUSIONS_FILE, CUSTOM_STOP_WORDS_FILE,
    now_stamp,
)
from config.model_config import (  # noqa: F401
    CONFIG_VERSION, _CONFIG_FIELD_DEFAULTS,
    upgrade_config_dict, ModelConfig,
)
from config.exclusions import (  # noqa: F401
    load_exclusions, save_exclusions,
    get_effective_stop_words, get_effective_noise_tokens, get_effective_noise_phrases,
)

# ── patterns ─────────────────────────────────────────────────────────────────
from patterns.dialog import (  # noqa: F401
    ROLE, DROP_LINES,
    HTML_TAG, PLAIN_HTML_WORDS_RE,
    MASK_RE, EMOJI_RE,
    ANSWER_HEADER_RE, DIALOG_MIN_WORDS,
    SECTION_PREFIX,
)
from patterns.noise import (  # noqa: F401
    OPERATOR_NOISE_RE, CLIENT_NOISE_RE,
    NOISE_TOKENS, NOISE_PHRASES,
)
from patterns.stop_words import RUSSIAN_STOP_WORDS  # noqa: F401

# ── ml constants ─────────────────────────────────────────────────────────────
from config.ml_constants import (  # noqa: F401
    hf_cache_key,
    SMOTE_MAX_MULTIPLIER, SMOTE_IMBALANCE_RATIO,
    CONF_THRESH_90_PERCENTILE, CONF_THRESH_75_PERCENTILE, CONF_THRESH_50_PERCENTILE,
    PR_MIN_PRECISION,
    SBERT_IMPORT_MAX_RETRIES,
    KMEANS_BATCH_SIZE, HDBSCAN_NOISE_BATCH_SIZE,
)
