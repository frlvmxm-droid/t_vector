# -*- coding: utf-8 -*-
"""
ml_distillation — дистилляция знаний (teacher → student).

Позволяет передать знания большой модели-учителя (напр. deepvk/USER2-large)
в компактную модель-ученика (deepvk/USER-base или LinearSVC+TF-IDF), сохраняя
большую часть качества при меньшем времени инференса и потребляемой памяти.

Поддерживаемые режимы:
  1. soft_label  — «мягкие метки»: учитель предсказывает вероятности по всем
                   классам (temperature-softmax), ученик обучается имитировать
                   это распределение вместо hard one-hot меток.
  2. logit_match — ученик-нейросеть (SetFit/трансформер) минимизирует KL-дивергенцию
                   между своими логитами и логитами учителя.

Текущее состояние: режим soft_label реализован для sklearn-пайплайнов,
которые поддерживают predict_proba. logit_match — scaffolding для будущей
интеграции с setfit.Trainer.

Как использовать:
  # 1. Обучите учителя (большую SetFit-модель или LinearSVC+SBERT)
  teacher_pipe = ... # обученный sklearn Pipeline с predict_proba

  # 2. Создайте студента (компактная TF-IDF модель)
  from ml_vectorizers import make_hybrid_vectorizer
  from ml_training import make_classifier
  student_features = make_hybrid_vectorizer(...)
  student_clf, _ = make_classifier(y_train, C=1.0, max_iter=1000, balanced=True)
  student_pipe = Pipeline([('features', student_features), ('clf', student_clf)])

  # 3. Запустите дистилляцию
  from ml_distillation import distill_soft_labels
  distilled_pipe = distill_soft_labels(
      teacher_pipe, student_pipe, X_train, y_train,
      temperature=3.0, log_cb=print
  )

Требования:
  • sklearn>=1.0 (CalibratedClassifierCV для soft-label обучения)
  • Учитель должен поддерживать predict_proba.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.pipeline import Pipeline

from app_logger import get_logger

_log = get_logger(__name__)


def distill_soft_labels(
    teacher: Any,
    student_pipe: Pipeline,
    X_train: Sequence[str],
    y_train: Sequence[str],
    *,
    temperature: float = 3.0,
    alpha: float = 0.5,
    test_size: float = 0.15,
    random_state: int = 42,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Pipeline:
    """Обучает студента на мягких метках учителя (soft label distillation).

    Алгоритм:
      1. Учитель предсказывает P_teacher(y|x) для каждого примера.
      2. Вероятности «сглаживаются» через температуру T > 1 (сильнее сглаживает
         при T→∞, жёстче при T→1).
      3. Финальная метка = alpha * P_teacher + (1 - alpha) * one_hot(y_true).
         alpha=1.0 — чистая дистилляция; alpha=0.0 — обычное обучение.
      4. Студент обучается через LogisticRegression с sample_weight,
         имитируя взвешенное распределение мягких меток.

    Параметры:
        teacher       — обученная модель-учитель (predict_proba)
        student_pipe  — НЕобученный sklearn Pipeline (features + clf)
        X_train       — тексты обучающей выборки
        y_train       — метки обучающей выборки
        temperature   — температура сглаживания (2.0–5.0 типично)
        alpha         — вес мягких меток учителя (0..1)
        log_cb        — callback для логирования

    Возвращает обученный student_pipe.
    """
    if not hasattr(teacher, "predict_proba"):
        raise ValueError("teacher должен поддерживать predict_proba()")

    def _log_msg(msg: str) -> None:
        _log.info(msg)
        if log_cb:
            log_cb(msg)

    X = list(X_train)
    y = list(y_train)
    classes = sorted(set(y))
    n_classes = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    _log_msg(f"[Дистилляция] Учитель → предсказываю вероятности ({len(X)} примеров)…")

    teacher_classes = list(teacher.classes_) if hasattr(teacher, "classes_") else classes
    t_cls_to_idx = {c: i for i, c in enumerate(teacher_classes)}

    teacher_proba_raw = teacher.predict_proba(X)

    # Температурное сглаживание (работает на логитах, аппроксимируем через log)
    if temperature != 1.0:
        eps = 1e-10
        log_p = np.log(np.clip(teacher_proba_raw, eps, 1.0)) / float(temperature)
        log_p -= log_p.max(axis=1, keepdims=True)
        teacher_proba_raw = np.exp(log_p)
        teacher_proba_raw /= teacher_proba_raw.sum(axis=1, keepdims=True).clip(eps)

    # Выровнять классы учителя → порядку classes студента
    teacher_proba = np.zeros((len(X), n_classes))
    for j_t, cls in enumerate(teacher_classes):
        j_s = cls_to_idx.get(cls)
        if j_s is not None:
            teacher_proba[:, j_s] = teacher_proba_raw[:, j_t]

    # One-hot hard labels
    hard_labels = np.zeros((len(X), n_classes))
    for i, lbl in enumerate(y):
        j = cls_to_idx.get(lbl)
        if j is not None:
            hard_labels[i, j] = 1.0

    # Мягкие метки = смесь teacher + hard
    soft_proba = alpha * teacher_proba + (1.0 - alpha) * hard_labels

    # Для sklearn: обучаем на argmax + sample_weight (= уверенность в истинном классе)
    # Это аппроксимация дистилляции для hard-label классификаторов.
    soft_y = [classes[int(soft_proba[i].argmax())] for i in range(len(X))]
    true_cls_proba = np.array([
        float(soft_proba[i, cls_to_idx.get(y_i, 0)])
        for i, y_i in enumerate(y)
    ])
    # sample_weight: примеры, где мягкая метка совпадает с истинной, весят больше
    sample_weights = np.where(
        np.array(soft_y) == np.array(y),
        1.0 + true_cls_proba,
        1.0 - true_cls_proba + 0.1,
    )

    _log_msg(
        f"[Дистилляция] T={temperature} alpha={alpha} | "
        f"мягких меток ≠ истинных: "
        f"{int((np.array(soft_y) != np.array(y)).sum())} из {len(y)}"
    )
    _log_msg("[Дистилляция] Обучение студента…")

    student_pipe.fit(X, soft_y, clf__sample_weight=sample_weights)
    _log_msg("[Дистилляция] ✅ Студент обучен.")
    return student_pipe


def evaluate_distillation(
    teacher: Any,
    student: Any,
    X_test: Sequence[str],
    y_test: Sequence[str],
    log_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Сравнивает метрики учителя и студента на тестовой выборке.

    Возвращает словарь:
      teacher_f1, student_f1, f1_drop, teacher_acc, student_acc, acc_drop
    """
    from sklearn.metrics import f1_score, accuracy_score

    def _log_msg(msg: str) -> None:
        _log.info(msg)
        if log_cb:
            log_cb(msg)

    t_pred = teacher.predict(list(X_test))
    s_pred = student.predict(list(X_test))

    t_f1 = float(f1_score(y_test, t_pred, average="macro", zero_division=0))
    s_f1 = float(f1_score(y_test, s_pred, average="macro", zero_division=0))
    t_acc = float(accuracy_score(y_test, t_pred))
    s_acc = float(accuracy_score(y_test, s_pred))

    result = {
        "teacher_f1": t_f1,
        "student_f1": s_f1,
        "f1_drop":    round(t_f1 - s_f1, 4),
        "teacher_acc": t_acc,
        "student_acc": s_acc,
        "acc_drop":   round(t_acc - s_acc, 4),
    }
    _log_msg(
        f"[Дистилляция] Учитель: F1={t_f1:.3f} acc={t_acc:.3f} | "
        f"Студент: F1={s_f1:.3f} acc={s_acc:.3f} | "
        f"Потеря: F1={result['f1_drop']:+.3f}"
    )
    return result
