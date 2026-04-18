# -*- coding: utf-8 -*-
"""
Иерархия пользовательских исключений приложения.

  AppBaseError            — общий базовый класс
  ├─ ModelLoadError       — ошибка загрузки / валидации модели
  ├─ FeatureBuildError    — ошибка построения текстового признака / векторайзера
  └─ PredictPipelineError — ошибка в конвейере предсказания (применение модели)
"""
from __future__ import annotations


class AppBaseError(Exception):
    """Базовый класс для всех исключений приложения."""


class ModelLoadError(AppBaseError):
    """Не удалось загрузить, десериализовать или валидировать файл модели.

    Примеры использования::

        raise ModelLoadError(f"Файл {path} повреждён или несовместим с текущей версией.")
    """


class SchemaError(ModelLoadError):
    """Ошибка версии/контракта schema для модельного bundle.

    Используется для hard-fail случаев несовместимости контрактов:
    - отсутствует обязательный schema_version;
    - неверный тип schema_version;
    - bundle из будущей версии приложения;
    - artifact_type не совпадает с ожидаемым.
    """


class FeatureBuildError(AppBaseError):
    """Ошибка при построении признаков или инициализации векторайзера.

    Примеры использования::

        raise FeatureBuildError("Колонка 'text' не найдена в данных.")
    """


class PredictPipelineError(AppBaseError):
    """Ошибка внутри конвейера предсказания (применение обученной модели).

    Примеры использования::

        raise PredictPipelineError("Несовместимые размерности: ожидалось 512, получено 768.")
    """


class UnexpectedError(AppBaseError):
    """Непредвиденная ошибка верхнего уровня с инцидентным маркером в логах."""
