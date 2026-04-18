# -*- coding: utf-8 -*-
"""
config/model_config.py — схема конфигурации модели (ModelConfig dataclass).

Содержит:
  • CONFIG_VERSION          — версия схемы (увеличивать при новых обязательных полях)
  • _CONFIG_FIELD_DEFAULTS  — значения по умолчанию для backward-compat
  • upgrade_config_dict()   — заполняет старые конфиги недостающими полями
  • ModelConfig             — dataclass с полным описанием параметров модели
"""
from __future__ import annotations

from dataclasses import dataclass

from config.user_config import SBERT_DEFAULT


# ---------------------------------------------------------------------------
# Версия схемы конфигурации
# ---------------------------------------------------------------------------
CONFIG_VERSION: int = 11

# Значения по умолчанию для всех полей ModelConfig.
# Новые поля добавлять сюда — тогда старые модели загрузятся корректно.
_CONFIG_FIELD_DEFAULTS: dict = {
    "use_noise_tokens":      True,
    "use_noise_phrases":     True,
    "use_per_field":         True,
    "use_svd":               True,
    "svd_components":        200,
    "use_lemma":             True,
    "parent_model":          "",
    "use_sbert":             False,
    "sbert_model":           SBERT_DEFAULT,
    "use_sbert_hybrid":      False,
    "use_meta":              False,
    "calib_method":          "sigmoid",
    "auto_profile":          "off",
    "class_weight_balanced": True,
    "test_size":             0.2,
    "sublinear_tf":          True,
    "use_stop_words":        True,
    "ignore_chatbot":        True,
    "use_summary":           True,
    "char_ng_min":           2,
    "char_ng_max":           9,
    "word_ng_min":           1,
    "word_ng_max":           3,
    "min_df":                3,
    "max_features":          150_000,
    "use_smote":             True,
    # SetFit / нейросетевой классификатор (schema v11+)
    "classifier_backend":          "linearsvc",  # "linearsvc" | "setfit"
    "nn_model":                    "",
    "nn_epochs":                   3,
    "nn_batch_size":               8,
    "nn_num_iterations":           20,
    "nn_fp16":                     True,
    "nn_gradient_checkpointing":   False,
    "C":                     3.0,
    "max_iter":              2000,
    "w_desc":                2,
    "w_client":              3,
    "w_operator":            1,
    "w_summary":             2,
    "w_answer_short":        1,
    "w_answer_full":         1,
}


def upgrade_config_dict(d: dict) -> dict:
    """Дополняет config-словарь из joblib значениями по умолчанию.

    Обеспечивает обратную совместимость: модели, обученные в старых версиях
    приложения (с меньшим набором полей), загружаются корректно.
    Возвращает новый словарь — оригинал не изменяется.
    """
    result = dict(_CONFIG_FIELD_DEFAULTS)
    result.update(d)
    return result


# ---------------------------------------------------------------------------
# Dataclass конфигурации модели
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    version: int
    created_at: str
    language: str

    classifier_type: str  # "LinearSVC+Calibrated" или "LogReg"

    desc_col: str
    call_col: str
    chat_col: str
    summary_col: str
    answer_short_col: str
    answer_full_col: str
    label_col: str

    use_summary: bool
    ignore_chatbot: bool
    auto_profile: str
    use_stop_words: bool

    w_desc: int
    w_client: int
    w_operator: int
    w_summary: int
    w_answer_short: int
    w_answer_full: int

    char_ng_min: int
    char_ng_max: int
    word_ng_min: int
    word_ng_max: int
    min_df: int
    max_features: int
    sublinear_tf: bool

    C: float
    max_iter: int
    class_weight_balanced: bool
    test_size: float

    parent_model: str = ""
    use_noise_tokens: bool = True
    use_noise_phrases: bool = True
    use_per_field: bool = True
    use_svd: bool = True
    svd_components: int = 200
    use_lemma: bool = True
    use_sbert: bool = False
    sbert_model: str = SBERT_DEFAULT
    use_sbert_hybrid: bool = False
    use_meta: bool = False
    calib_method: str = "sigmoid"
    use_smote: bool = True

    # SetFit / нейросетевой классификатор (schema v11+)
    classifier_backend: str = "linearsvc"  # "linearsvc" | "setfit"
    nn_model: str = ""
    nn_epochs: int = 3
    nn_batch_size: int = 8
    nn_num_iterations: int = 20
    nn_fp16: bool = True
    nn_gradient_checkpointing: bool = False

    def __post_init__(self):
        if not (self.test_size == 0.0 or 0.0 < self.test_size < 1.0):
            raise ValueError(f"test_size must be 0 (no validation) or in (0, 1), got {self.test_size}")
        if self.C <= 0:
            raise ValueError(f"C must be positive, got {self.C}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.svd_components < 1:
            raise ValueError(f"svd_components must be >= 1, got {self.svd_components}")
        if self.char_ng_min > self.char_ng_max:
            raise ValueError(
                f"char_ng_min ({self.char_ng_min}) > char_ng_max ({self.char_ng_max})"
            )
        if self.word_ng_min > self.word_ng_max:
            raise ValueError(
                f"word_ng_min ({self.word_ng_min}) > word_ng_max ({self.word_ng_max})"
            )
        if self.min_df < 1:
            raise ValueError(f"min_df must be >= 1, got {self.min_df}")
        if self.max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {self.max_features}")
        if self.classifier_backend not in ("linearsvc", "setfit"):
            raise ValueError(f"classifier_backend must be 'linearsvc' or 'setfit', got {self.classifier_backend!r}")
        if self.nn_epochs < 1:
            raise ValueError(f"nn_epochs must be >= 1, got {self.nn_epochs}")
        if self.nn_batch_size < 1:
            raise ValueError(f"nn_batch_size must be >= 1, got {self.nn_batch_size}")
