# -*- coding: utf-8 -*-
"""
dataset_analyzer — анализ обучающего датасета и подбор оптимальных параметров.

Публичный API:
    analyze_dataset(X, y, field_coverage) -> dict
    build_param_changes(recommendations, current_values)  -> list[dict]
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

# ── Пороговые константы ───────────────────────────────────────────────────────
_TINY   = 200       # < TINY   → крошечный датасет
_SMALL  = 800       # < SMALL  → маленький
_MEDIUM = 3_000     # < MEDIUM → средний
_LARGE  = 15_000    # < LARGE  → большой; иначе очень большой

_IMB_WARN   = 3.0   # дисбаланс: предупреждение
_IMB_SMOTE  = 6.0   # дисбаланс: нужен SMOTE
_RARE_MIN   = 10    # менее N примеров — редкий класс
_LOW_COV    = 0.45  # поле заполнено менее чем в 45% строк


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dataset(
    X: List[str],
    y: List[str],
    field_coverage: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Анализирует обучающий датасет и возвращает статистику, проблемы и рекомендации.

    Args:
        X: список фиче-текстов (или сырых текстов)
        y: список меток классов
        field_coverage: {имя_поля: доля_непустых_строк} — опционально

    Returns:
        {
          "stats":           {…числовые метрики…},
          "issues":          [{level, msg}, …],
          "recommendations": {param: value, …},
        }
    """
    n = len(X)
    counts = Counter(y)
    n_cls = len(counts)
    sizes = sorted(counts.values())
    max_c = sizes[-1] if sizes else 1
    min_c = sizes[0]  if sizes else 1
    imb   = round(max_c / max(min_c, 1), 2)
    rare  = [cls for cls, cnt in counts.items() if cnt < _RARE_MIN]

    text_lens  = [len(t) for t in X]
    avg_len    = float(np.mean(text_lens))    if text_lens else 0.0
    median_len = float(np.median(text_lens))  if text_lens else 0.0

    # Топ и хвост распределения классов
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top5   = sorted_counts[:5]
    tail5  = sorted_counts[-5:] if len(sorted_counts) > 5 else []

    stats: Dict[str, Any] = {
        "n_samples":       n,
        "n_classes":       n_cls,
        "class_counts":    dict(counts),
        "max_class_count": max_c,
        "min_class_count": min_c,
        "imbalance_ratio": imb,
        "rare_classes":    rare,
        "avg_text_len":    round(avg_len, 1),
        "median_text_len": round(median_len, 1),
        "top5_classes":    top5,
        "tail5_classes":   tail5,
        "field_coverage":  field_coverage or {},
    }

    return {
        "stats":           stats,
        "issues":          _build_issues(stats),
        "recommendations": _build_recommendations(stats),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Выявление проблем
# ─────────────────────────────────────────────────────────────────────────────

def _build_issues(stats: Dict[str, Any]) -> List[Dict[str, str]]:
    issues = []
    n      = stats["n_samples"]
    n_cls  = stats["n_classes"]
    imb    = stats["imbalance_ratio"]
    rare   = stats["rare_classes"]
    cov    = stats.get("field_coverage", {})

    # Объём данных
    if n < 30:
        issues.append({"level": "critical",
                        "msg": f"Критически мало примеров: {n}. Минимум для обучения — 50–100."})
    elif n < _TINY:
        issues.append({"level": "warning",
                        "msg": f"Мало примеров: {n}. Возможно переобучение — добавьте больше данных."})
    elif n < _SMALL:
        issues.append({"level": "info",
                        "msg": f"Небольшой датасет ({n} примеров). Качество может улучшиться при расширении."})

    # Число классов
    if n_cls < 2:
        issues.append({"level": "critical",
                        "msg": "Менее 2 классов — обучение невозможно."})
    elif n_cls > 50:
        issues.append({"level": "info",
                        "msg": f"Много классов ({n_cls}). Иерархическая классификация значительно улучшит качество."})
    elif n_cls > 20:
        issues.append({"level": "info",
                        "msg": f"Больше 20 классов ({n_cls}) — рекомендуется иерархическая классификация."})

    # Дисбаланс
    if imb > 20:
        issues.append({"level": "critical",
                        "msg": f"Критический дисбаланс классов ({imb:.0f}x). Модель будет игнорировать редкие классы."})
    elif imb > _IMB_SMOTE:
        issues.append({"level": "warning",
                        "msg": f"Сильный дисбаланс ({imb:.1f}x). Рекомендуется SMOTE + class_weight=balanced."})
    elif imb > _IMB_WARN:
        issues.append({"level": "info",
                        "msg": f"Умеренный дисбаланс ({imb:.1f}x). Включите class_weight=balanced."})

    # Редкие классы
    if rare:
        names = ", ".join(f"«{c}»" for c in rare[:4])
        extra = f" и ещё {len(rare) - 4}" if len(rare) > 4 else ""
        issues.append({"level": "warning",
                        "msg": f"Редкие классы (<{_RARE_MIN} примеров): {names}{extra}. "
                               "Добавьте примеры или используйте якорные тексты."})

    # Покрытие полей
    low = {f: round(p * 100) for f, p in cov.items() if p < _LOW_COV}
    if low:
        lf_str = ", ".join(f"{f} ({p}%)" for f, p in list(low.items())[:3])
        extra2 = f" и ещё {len(low) - 3}" if len(low) > 3 else ""
        issues.append({"level": "info",
                        "msg": f"Низкое заполнение полей: {lf_str}{extra2}. "
                               "Включите field-dropout для устойчивости к пропускам."})

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Генерация рекомендаций
# ─────────────────────────────────────────────────────────────────────────────

def _build_recommendations(stats: Dict[str, Any]) -> Dict[str, Any]:
    n      = stats["n_samples"]
    n_cls  = stats["n_classes"]
    imb    = stats["imbalance_ratio"]
    avg    = stats["avg_text_len"]
    rare   = stats["rare_classes"]
    cov    = stats.get("field_coverage", {})

    r: Dict[str, Any] = {}

    # ── TF-IDF и C в зависимости от размера датасета ─────────────────────────
    if n < _TINY:
        r.update(max_features=10_000, C=0.3, use_svd=False, svd_components=100,
                 model_label="LinearSVC (маленький датасет)")
    elif n < _SMALL:
        r.update(max_features=30_000, C=1.0, use_svd=False, svd_components=150,
                 model_label="LinearSVC")
    elif n < _MEDIUM:
        r.update(max_features=60_000, C=1.0, use_svd=True,  svd_components=200,
                 model_label="LinearSVC + SVD")
    elif n < _LARGE:
        r.update(max_features=100_000, C=0.7, use_svd=True, svd_components=300,
                 model_label="LinearSVC + SVD")
    else:
        r.update(max_features=150_000, C=0.3, use_svd=True, svd_components=400,
                 model_label="LinearSVC + SVD (крупный датасет)")

    # ── N-gram диапазон по средней длине текста ───────────────────────────────
    if avg < 60:
        r.update(char_ng_min=2, char_ng_max=4, word_ng_min=1, word_ng_max=1)
    elif avg < 200:
        r.update(char_ng_min=2, char_ng_max=6, word_ng_min=1, word_ng_max=2)
    else:
        r.update(char_ng_min=2, char_ng_max=6, word_ng_min=1, word_ng_max=3)

    # ── Дисбаланс ─────────────────────────────────────────────────────────────
    r["balanced"]   = imb > 2.5
    r["use_smote"]  = imb > _IMB_SMOTE

    # ── Много классов → иерархия и hard negatives ─────────────────────────────
    r["use_hierarchical"]  = n_cls >= 15
    r["use_hard_negatives"] = n_cls >= 8 and n >= 300

    # ── Пропуски полей → field-dropout ───────────────────────────────────────
    r["use_field_dropout"] = any(p < _LOW_COV for p in cov.values())

    # ── Ансамблевые методы ────────────────────────────────────────────────────
    r["use_kfold_ensemble"]     = n >= 600 and n_cls >= 3
    r["kfold_k"]                = 5
    r["use_confident_learning"] = n >= 800

    # ── Якорные тексты (информационная рекомендация) ──────────────────────────
    r["suggest_anchor_texts"] = len(rare) > 0

    return r


# ─────────────────────────────────────────────────────────────────────────────
# Утилита: список изменений параметров (до / после)
# ─────────────────────────────────────────────────────────────────────────────

# Человекочитаемые имена параметров
_PARAM_LABELS: Dict[str, str] = {
    "max_features":          "max_features (TF-IDF)",
    "C":                     "C (регуляризация)",
    "use_svd":               "Использовать SVD",
    "svd_components":        "SVD компонент",
    "char_ng_min":           "char n-gram мин",
    "char_ng_max":           "char n-gram макс",
    "word_ng_min":           "word n-gram мин",
    "word_ng_max":           "word n-gram макс",
    "balanced":              "class_weight=balanced",
    "use_smote":             "SMOTE (оверсэмплинг)",
    "use_hierarchical":      "Иерархическая классификация",
    "use_hard_negatives":    "Hard negative oversampling",
    "use_field_dropout":     "Field-dropout аугментация",
    "use_kfold_ensemble":    "K-fold ансамбль",
    "use_confident_learning": "Confident Learning",
}

# Параметры, которые показываем в таблице (в нужном порядке)
_SHOW_PARAMS = [
    "max_features", "C", "use_svd", "svd_components",
    "char_ng_min", "char_ng_max", "word_ng_min", "word_ng_max",
    "balanced", "use_smote",
    "use_hierarchical", "use_hard_negatives", "use_field_dropout",
    "use_kfold_ensemble", "use_confident_learning",
]


def build_param_changes(
    recommendations: Dict[str, Any],
    current_values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Сравнивает рекомендации с текущими значениями.

    Returns: список [{param, label, current, recommended, changed}]
    """
    rows = []
    for p in _SHOW_PARAMS:
        if p not in recommendations:
            continue
        rec_val = recommendations[p]
        cur_val = current_values.get(p)
        changed = cur_val != rec_val
        rows.append({
            "param":       p,
            "label":       _PARAM_LABELS.get(p, p),
            "current":     cur_val,
            "recommended": rec_val,
            "changed":     changed,
        })
    return rows


def _fmt(v: Any) -> str:
    """Форматирует значение параметра для отображения."""
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v) if v is not None else "—"
