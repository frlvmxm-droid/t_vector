# -*- coding: utf-8 -*-
"""
core — пакет-шим для обратной совместимости.

.. deprecated::
    Все модули пакета ``core`` являются шимами и будут удалены в следующем
    мажорном релизе. Импортируйте напрямую из корневых модулей:

      * ``ml_core`` / ``ml_vectorizers`` / ``ml_training`` / ``ml_diagnostics``
      * ``feature_builder``
      * ``text_utils``
      * ``hw_profile``
"""
# Намеренно не импортируем шимы здесь — это предотвращает срабатывание
# DeprecationWarning при простом ``import core``.
# Предупреждение появляется только при ``from core.ml_core import ...`` и т.п.
