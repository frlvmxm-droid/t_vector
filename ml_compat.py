"""
Ранние патчи совместимости для torch и packaging.version.

Вызывается из bootstrap-контекста через
``ml_compat.apply_early_compat_patches()`` в `ml_vectorizers.py` до первого
использования sentence_transformers, чтобы torch.__version__ был гарантированно
установлен ДО первого импорта transformers.

Файл не содержит бизнес-логики и не должен импортировать другие ML-модули.

Governance:
  - lifecycle и критерии выключения патчей описаны в
    MONKEY_PATCH_GOVERNANCE.md.
"""
from __future__ import annotations

from app_logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Ранний патч torch.__version__
# На некоторых системах (Windows + Python 3.13) torch.__version__ == None.
# Это ломает sentence_transformers/transformers при первом импорте:
# packaging.version.parse(None) → TypeError → is_torch_available() → False
# → AutoModel.from_pretrained = None → TypeError при _ensure_model().
#
# Патчируем ДО первого импорта sentence_transformers (который происходит
# через SBERTVectorizer.is_available() во время построения UI), чтобы
# transformers корректно задетектировал torch при первом импорте.
# ---------------------------------------------------------------------------
_PATCH_APPLIED = False


def apply_early_compat_patches() -> None:
    """Применяет ранние патчи совместимости один раз за процесс."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    try:
        import sys as _sys_early
        _torch_early = _sys_early.modules.get("torch")
        if _torch_early is None:
            try:
                import torch as _torch_early
            except Exception as _e_torch_early:
                _log.warning(
                    "[ml_compat] ранний патч torch.__version__: torch не импортируется (%s). "
                    "Это нормально, если torch не установлен.",
                    _e_torch_early,
                )
                _torch_early = None
        if _torch_early is not None and getattr(_torch_early, "__version__", None) is None:
            try:
                import importlib.metadata as _imeta_early
                _torch_early.__version__ = _imeta_early.version("torch")
            except Exception:
                _torch_early.__version__ = "2.4.0"
        _PATCH_APPLIED = True
    except Exception as _e_early_patch:
        _log.warning(
            "[ml_compat] ранний патч torch.__version__ завершился с ошибкой: %s",
            _e_early_patch,
        )
