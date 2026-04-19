# -*- coding: utf-8 -*-
"""
ml_training — обучение классификатора и подбор гиперпараметров.

Содержит:
  make_classifier            — фабрика LinearSVC / LogReg
  _maybe_skip_validation     — проверка минимального датасета
  _oversample_rare_classes   — текстовый оверсэмплинг
  _oversample_hard_negatives — оверсэмплинг граничных примеров между близкими классами
  _field_dropout_augment     — обучающие копии с случайно удалёнными полями (робастность к пропускам)
  _compute_validation_extras — метрики валидации
  train_model                — полный цикл обучения
  find_best_c                — GridSearch по C (кросс-валидация)
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config.ml_constants import (
    SMOTE_MAX_MULTIPLIER, SMOTE_IMBALANCE_RATIO,
    CONF_THRESH_90_PERCENTILE, CONF_THRESH_75_PERCENTILE, CONF_THRESH_50_PERCENTILE,
    PR_MIN_PRECISION,
)
from app_logger import get_logger

_log = get_logger(__name__)

# --- nn_mix (SMOTE-подобное смешение для текстов) --------------------------
# Параметры Beta(α, α): α<1 предпочитает крайние смеси (как classic SMOTE weight).
_NN_MIX_BETA_ALPHA = 0.4
# Порог косинусного расстояния для отбрасывания самого себя при поиске ближайшего соседа.
_NN_MIX_SELF_DIST_EPS = 1e-6
# TF-IDF для построения индекса NN: малая модель, char_wb 2..4.
_NN_MIX_NGRAM_RANGE = (2, 4)
_NN_MIX_MAX_FEATURES = 15_000
# ---------------------------------------------------------------------------


def make_classifier(
    y: List[str],
    C: float,
    max_iter: int,
    balanced: bool,
    calib_method: str = "sigmoid",
) -> Tuple[Any, str]:
    """
    Выбирает и создаёт классификатор в зависимости от размера датасета.

    - LinearSVC + CalibratedClassifierCV (cv динамический) если min_class >= 3
      и классов >= 2 → поддерживает predict_proba / top3
    - LogisticRegression как fallback для маленьких датасетов

    calib_method: "sigmoid" (Platt scaling, надёжен на малых данных)
                  "isotonic" (монотонная регрессия, точнее при >1000 сэмплов/класс)

    CV выбирается динамически: min(5, max(3, ceil(min_class / 3))).
    Это гарантирует ≥2 примера каждого класса в каждом фолде при малых датасетах
    и стандартный CV=5 при min_class >= 15.
    """
    cnt = Counter(y)
    min_class = min(cnt.values()) if cnt else 0
    # Динамический CV: уменьшаем фолды на малых датасетах, чтобы каждый фолд
    # имел ≥ 2 примера каждого класса → более надёжная калибровка.
    _CV = min(5, max(3, math.ceil(min_class / 3)))
    if min_class >= _CV and len(set(y)) >= 2:
        base = LinearSVC(
            C=float(C),
            max_iter=int(max_iter),
            class_weight=("balanced" if balanced else None),
        )
        if calib_method == "auto":
            # isotonic точнее при >= 200 сэмплов/класс, sigmoid надёжнее на малых данных
            avg_per_class = len(y) / max(len(cnt), 1)
            _method = "isotonic" if avg_per_class >= 200 else "sigmoid"
        else:
            _method = calib_method if calib_method in ("sigmoid", "isotonic") else "sigmoid"
        est = CalibratedClassifierCV(base, cv=_CV, method=_method)
        return est, f"LinearSVC+Calibrated({_method},cv={_CV})"
    # Fallback для очень маленьких датасетов
    est = LogisticRegression(
        C=float(C),
        max_iter=int(max_iter),
        class_weight=("balanced" if balanced else None),
    )
    return est, "LogReg"


def _maybe_skip_validation(
    X: List[str],
    y: List[str],
    test_size: float,
    pipe: Pipeline,
    clf_type: str,
    log_cb: Optional[Callable[[str], None]],
    progress_cb: Optional[Callable[[float, str], None]],
) -> Optional[Tuple[Pipeline, str, str, None, None, Dict]]:
    """Проверяет, можно ли провести стратифицированный split.

    Возвращает готовый результат train_model (без валидации) если данных
    недостаточно, иначе None — тогда вызывающий код продолжает обучение с holdout.
    """
    _cnt = Counter(y)
    _min_class = min(_cnt.values()) if _cnt else 0
    if test_size > 0 and len(set(y)) >= 2 and len(y) >= 30 and _min_class >= 2:
        return None  # достаточно данных — продолжаем штатный путь

    if progress_cb:
        progress_cb(78.0, "Обучение (без валидации)…")
    pipe.fit(X, y)
    reason = (
        "test_size=0" if test_size <= 0
        else f"класс с {_min_class} примером(ами) — нужно ≥2 для стратифицированного сплита"
        if _min_class < 2
        else "мало данных"
    )
    if log_cb:
        log_cb(f"[Обучение] Валидация пропущена: {reason}")
    return pipe, clf_type, f"ВАЛИДАЦИЯ ПРОПУЩЕНА ({reason}).", None, None, {}


def fuzzy_string_dedup(
    X: List[str],
    y: List[str],
    *,
    threshold: int = 92,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[str], List[str], int]:
    """Near-duplicate dedup в паре (X, y) по rapidfuzz.token_sort_ratio.

    Группирует строки одного класса с similarity ≥ threshold, оставляя
    по одному представителю на группу. Дубликаты разных классов не
    трогаем — они диагностически полезны как сигнал неоднозначности
    разметки. При отсутствии rapidfuzz — graceful fallback на exact
    dedup (strip+casefold).

    Returns: (X_dedup, y_dedup, n_removed).
    """
    if not X:
        return list(X), list(y), 0
    n0 = len(X)
    try:
        from rapidfuzz import fuzz  # type: ignore[import-not-found]
    except ImportError:
        seen: Dict[Tuple[str, str], bool] = {}
        out_x: List[str] = []
        out_y: List[str] = []
        for x, yi in zip(X, y):
            key = (str(x).strip().casefold(), str(yi))
            if key in seen:
                continue
            seen[key] = True
            out_x.append(x)
            out_y.append(yi)
        removed = n0 - len(out_x)
        if removed and log_cb:
            log_cb(f"[Dedup] Удалено точных дубликатов: {removed} (rapidfuzz недоступен)")
        return out_x, out_y, removed

    by_class: Dict[str, List[int]] = {}
    for i, yi in enumerate(y):
        by_class.setdefault(str(yi), []).append(i)

    keep_mask = [True] * n0
    for _cls, idxs in by_class.items():
        normalized = [str(X[i]).strip().casefold() for i in idxs]
        for a in range(len(idxs)):
            if not keep_mask[idxs[a]]:
                continue
            for b in range(a + 1, len(idxs)):
                if not keep_mask[idxs[b]]:
                    continue
                score = fuzz.token_sort_ratio(normalized[a], normalized[b])
                if score >= threshold:
                    keep_mask[idxs[b]] = False
    out_x = [X[i] for i, keep in enumerate(keep_mask) if keep]
    out_y = [y[i] for i, keep in enumerate(keep_mask) if keep]
    removed = n0 - len(out_x)
    if removed and log_cb:
        log_cb(f"[Dedup] Near-duplicate'ов (rapidfuzz ≥{threshold}): {removed}")
    return out_x, out_y, removed


def _oversample_rare_classes(
    Xtr: List[str],
    ytr: List[str],
    random_state: int,
    log_cb: Optional[Callable[[str], None]],
    progress_cb: Optional[Callable[[float, str], None]],
    max_dup_per_sample: int = 5,
    strategy: str = "cap",
) -> Tuple[List[str], List[str]]:
    """Текстовый оверсэмплинг редких классов (аналог SMOTE для строк).

    Если дисбаланс ≥ SMOTE_IMBALANCE_RATIO — дублирует редкие классы
    до размера мажоритарного (но не более чем SMOTE_MAX_MULTIPLIER раз).
    Работает до feature extraction, не требует imbalanced-learn.
    Возвращает (возможно расширенные) Xtr, ytr.
    """
    _rng = np.random.default_rng(int(random_state))
    _tr_cnt = Counter(ytr)
    if not _tr_cnt:
        return Xtr, ytr
    _max_cnt = max(_tr_cnt.values())
    _min_cnt = min(_tr_cnt.values())
    if _max_cnt / _min_cnt < SMOTE_IMBALANCE_RATIO:
        return Xtr, ytr

    def _augment_light_text(text: str) -> str:
        toks = str(text).split()
        if len(toks) >= 4:
            drop_idx = int(_rng.integers(0, len(toks)))
            toks = [t for i, t in enumerate(toks) if i != drop_idx]
            return " ".join(toks)
        if toks:
            return f"{' '.join(toks)} !"
        return text

    # nn_mix — SMOTE-подобное смешение: для редкого примера находим ближайшего
    # соседа того же класса по TF-IDF и порождаем синтетический текст, беря
    # часть токенов из каждого. Требуется TF-IDF фит по Xtr (один раз).
    _nn_cache: Dict[str, Any] = {}
    def _build_nn_index() -> None:
        if _nn_cache:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer as _TfIdf
            from sklearn.neighbors import NearestNeighbors
            _vec = _TfIdf(analyzer="char_wb", ngram_range=_NN_MIX_NGRAM_RANGE,
                          max_features=_NN_MIX_MAX_FEATURES, min_df=1)
            _M = _vec.fit_transform([str(t) for t in Xtr])
            _nn_cache["vec"] = _vec
            _nn_cache["M"] = _M
            _nn_cache["nn_by_cls"] = {}
            _nn_cache["idx_by_cls"] = {}
            import numpy as _np
            _y_arr = _np.asarray(ytr)
            for _cls in set(ytr):
                _idx = _np.where(_y_arr == _cls)[0]
                if len(_idx) < 2:
                    continue
                _nn = NearestNeighbors(n_neighbors=min(3, len(_idx)), metric="cosine")
                _nn.fit(_M[_idx])
                _nn_cache["nn_by_cls"][_cls] = _nn
                _nn_cache["idx_by_cls"][_cls] = _idx
        except Exception as _nn_exc:  # noqa: BLE001 — sklearn NN/vectorizer cold-start failures are diverse; fall back safely
            _log.warning("nn_mix index build failed: %s", _nn_exc)
            _nn_cache["failed"] = True

    def _nn_mix_text(base_text: str, cls: str) -> str:
        _build_nn_index()
        if _nn_cache.get("failed") or cls not in _nn_cache.get("nn_by_cls", {}):
            return _augment_light_text(base_text)
        try:
            _vec = _nn_cache["vec"]
            _nn = _nn_cache["nn_by_cls"][cls]
            _idx_cls = _nn_cache["idx_by_cls"][cls]
            _q = _vec.transform([base_text])
            _dist, _nbrs = _nn.kneighbors(_q, n_neighbors=min(3, len(_idx_cls)))
            # Берём ближайшего соседа (не сам пример — пропускаем расстояние 0)
            _nbr_i = None
            for _d, _ni in zip(_dist[0], _nbrs[0]):
                if _d > _NN_MIX_SELF_DIST_EPS:
                    _nbr_i = int(_idx_cls[_ni])
                    break
            if _nbr_i is None:
                return _augment_light_text(base_text)
            _neighbour = str(Xtr[_nbr_i])
            _a_toks = str(base_text).split()
            _b_toks = _neighbour.split()
            if not _a_toks or not _b_toks:
                return _augment_light_text(base_text)
            # λ ~ Beta(α, α) — предпочитает крайние смеси (как classic SMOTE weights)
            _lam = float(_rng.beta(_NN_MIX_BETA_ALPHA, _NN_MIX_BETA_ALPHA))
            _n_a = max(1, int(round(len(_a_toks) * _lam)))
            _take_a = list(_rng.choice(_a_toks, size=_n_a, replace=False)) if _n_a <= len(_a_toks) \
                else list(_a_toks)
            _n_b = max(1, int(round(len(_b_toks) * (1.0 - _lam))))
            _take_b = list(_rng.choice(_b_toks, size=_n_b, replace=False)) if _n_b <= len(_b_toks) \
                else list(_b_toks)
            _mixed = list(_take_a) + list(_take_b)
            _rng.shuffle(_mixed)
            return " ".join(str(t) for t in _mixed)
        except Exception as _mix_exc:  # noqa: BLE001 — best-effort augmentation; any failure falls back to augment_light
            _log.debug("nn_mix_text fallback to augment_light: %s", _mix_exc)
            return _augment_light_text(base_text)

    _Xaug: List[str] = list(Xtr)
    _yaug: List[str] = list(ytr)
    _dup_log: Dict[str, float] = {}
    _max_dup = max(1, int(max_dup_per_sample))
    _strategy = (strategy or "cap").strip().lower()
    if _strategy not in {"duplicate", "cap", "augment_light", "nn_mix"}:
        _strategy = "cap"
    for _cls, _cnt in _tr_cnt.items():
        if _max_cnt / _cnt >= SMOTE_IMBALANCE_RATIO:
            _target = min(_cnt * SMOTE_MAX_MULTIPLIER, _max_cnt)
            if _strategy in {"cap", "augment_light"}:
                _target = min(_target, _cnt * _max_dup)
            _n_add = _target - _cnt
            _src = [i for i, lbl in enumerate(ytr) if lbl == _cls]
            for _ in range(_n_add):
                _sample = Xtr[int(_rng.choice(_src))]
                if _strategy == "augment_light":
                    _sample = _augment_light_text(_sample)
                elif _strategy == "nn_mix":
                    _sample = _nn_mix_text(_sample, _cls)
                _Xaug.append(_sample)
                _yaug.append(_cls)
            _dup_log[_cls] = _target / max(_cnt, 1)
    if len(_Xaug) <= len(Xtr):
        return Xtr, ytr

    _orig = len(Xtr)
    # Single-pass permutation: O(N) index shuffle avoids building a list of
    # (x, label) tuples and re-allocating two result lists.
    _perm = _rng.permutation(len(_Xaug))
    Xtr = [_Xaug[i] for i in _perm]
    ytr = [_yaug[i] for i in _perm]
    if progress_cb:
        progress_cb(79.0, f"Балансировка: {len(Xtr)} примеров (было {_orig})")
    if log_cb:
        _aug_classes = sorted(
            {cls for cls, cnt in _tr_cnt.items() if _max_cnt / cnt >= SMOTE_IMBALANCE_RATIO}
        )
        log_cb(f"[Балансировка] оверсэмплинг: {_orig} → {len(Xtr)} примеров "
               f"| расширены классы: {_aug_classes} | стратегия={_strategy}")
        if _dup_log:
            _dup_parts = ", ".join(f"{cls}×{factor:.2f}" for cls, factor in sorted(_dup_log.items()))
            log_cb(f"[Балансировка] факторы дублирования: {_dup_parts}")
    return Xtr, ytr


def _oversample_hard_negatives(
    Xtr: List[str],
    ytr: List[str],
    random_state: int,
    log_cb: Optional[Callable[[str], None]] = None,
    n_top_per_pair: int = 10,
    pair_sim_threshold: float = 0.50,
    max_total_added: int = 2000,
) -> Tuple[List[str], List[str]]:
    """Оверсэмплинг граничных примеров между семантически близкими классами.

    Для каждой пары классов с косинусным сходством центроидов ≥ pair_sim_threshold
    находит n_top_per_pair примеров каждого класса, наиболее похожих на центроид другого,
    и дублирует их в обучающей выборке. Это помогает классификатору лучше различать
    часто путаемые категории.
    """
    import scipy.sparse
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfIdf
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    _cnt = Counter(ytr)
    if len(_cnt) < 2 or len(Xtr) < 10:
        return Xtr, ytr

    try:
        _vec = _TfIdf(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=15_000, min_df=1,
        )
        _M = _vec.fit_transform(Xtr)
    except (ValueError, MemoryError, TypeError) as _tfidf_exc:
        _log.warning("oversample TF-IDF vectorizer failed: %s", _tfidf_exc)
        return Xtr, ytr

    _vecs = _M.toarray() if scipy.sparse.issparse(_M) else np.asarray(_M)
    _labels_arr = np.array(ytr)
    _unique = sorted(_cnt.keys())
    _centroids = {c: _vecs[_labels_arr == c].mean(axis=0) for c in _unique}

    _centroid_mat = np.array([_centroids[c] for c in _unique])
    _c_sims = _cos_sim(_centroid_mat)

    _Xaug: List[str] = list(Xtr)
    _yaug: List[str] = list(ytr)
    _rng = np.random.default_rng(int(random_state))
    _n_added = 0
    _pairs_processed = []

    for _i, _ci in enumerate(_unique):
        for _j, _cj in enumerate(_unique[_i + 1:], _i + 1):
            if _c_sims[_i, _j] < pair_sim_threshold:
                continue
            if _n_added >= max_total_added:
                break

            _remaining = max_total_added - _n_added
            _top_k = min(n_top_per_pair, _remaining // 2 + 1)

            # Примеры класса ci, ближайшие к центроиду cj
            _idx_ci = np.where(_labels_arr == _ci)[0]
            if len(_idx_ci) > 0:
                _sim_ci = _cos_sim(
                    _vecs[_idx_ci], _centroids[_cj].reshape(1, -1)
                )[:, 0]
                _hard_ci = _idx_ci[np.argsort(_sim_ci)[::-1][:_top_k]]
                for _h in _hard_ci:
                    _Xaug.append(Xtr[int(_h)])
                    _yaug.append(_ci)
                    _n_added += 1

            # Примеры класса cj, ближайшие к центроиду ci
            _idx_cj = np.where(_labels_arr == _cj)[0]
            if len(_idx_cj) > 0:
                _sim_cj = _cos_sim(
                    _vecs[_idx_cj], _centroids[_ci].reshape(1, -1)
                )[:, 0]
                _hard_cj = _idx_cj[np.argsort(_sim_cj)[::-1][:_top_k]]
                for _h in _hard_cj:
                    _Xaug.append(Xtr[int(_h)])
                    _yaug.append(_cj)
                    _n_added += 1

            _pairs_processed.append((_ci, _cj, float(_c_sims[_i, _j])))

        if _n_added >= max_total_added:
            break

    if _n_added == 0:
        return Xtr, ytr

    _perm = _rng.permutation(len(_Xaug))
    Xtr_out = [_Xaug[i] for i in _perm]
    ytr_out = [_yaug[i] for i in _perm]

    if log_cb:
        _top_pairs = sorted(_pairs_processed, key=lambda p: -p[2])[:5]
        _pair_str = ", ".join(f"«{a}»↔«{b}»({s:.2f})" for a, b, s in _top_pairs)
        log_cb(
            f"[Hard Negatives] +{_n_added} граничных примеров | "
            f"пар: {len(_pairs_processed)} | топ: {_pair_str}"
        )
    return Xtr_out, ytr_out


# Секции feature-text, которые можно случайно удалить при field dropout
_DROPOUT_FIELD_TAGS = frozenset(
    {"DESC", "CLIENT", "OPERATOR", "SUMMARY", "ANSWER_SHORT", "ANSWER_FULL"}
)


def _field_dropout_augment(
    Xtr: List[str],
    ytr: List[str],
    dropout_prob: float = 0.15,
    n_copies: int = 2,
    random_state: int = 42,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[str], List[str]]:
    """Аугментация для устойчивости к пропущенным полям.

    Для каждого обучающего примера создаёт n_copies копий, в которых
    каждая секция [FIELD] случайно удаляется с вероятностью dropout_prob.
    Это обучает модель классифицировать даже при заполненных 2-3 из 6 ячеек.
    Если все поля выпали — копия не добавляется.
    """
    _rng = np.random.default_rng(int(random_state))
    _Xaug: List[str] = list(Xtr)
    _yaug: List[str] = list(ytr)
    _n_added = 0

    for _text, _label in zip(Xtr, ytr):
        _lines = _text.split("\n")
        for _ in range(n_copies):
            # Парсим секции и случайно выбираем, какие оставить
            _out_lines: List[str] = []
            _skip = False
            for _line in _lines:
                _stripped = _line.strip()
                if _stripped.startswith("[") and _stripped.endswith("]"):
                    _tag = _stripped[1:-1]
                    if _tag in _DROPOUT_FIELD_TAGS:
                        # Удаляем поле с заданной вероятностью
                        _skip = (_rng.random() < dropout_prob)
                    else:
                        _skip = False
                if not _skip:
                    _out_lines.append(_line)

            _new_text = "\n".join(_out_lines).strip()
            # Добавляем только если хоть что-то осталось после дропаута
            if _new_text and any(
                f"[{t}]" in _new_text for t in _DROPOUT_FIELD_TAGS
            ):
                _Xaug.append(_new_text)
                _yaug.append(_label)
                _n_added += 1

    if log_cb and _n_added > 0:
        log_cb(
            f"[Field Dropout] +{_n_added} аугментированных копий "
            f"(p={dropout_prob:.0%}, n_copies={n_copies}) "
            f"→ итого {len(_Xaug)} примеров"
        )
    return _Xaug, _yaug


def detect_mislabeled_examples(
    pipe: Any,
    X: List[str],
    y: List[str],
    threshold: float = 0.30,
    max_results: int = 200,
    log_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """Находит обучающие примеры с подозрительно низкой уверенностью модели.

    После обучения прогоняет pipe.predict_proba на обучающих данных.
    Если вероятность истинного класса < threshold — пример подозрительный:
    возможная ошибка разметки, пограничный случай или аномалия.

    Returns список dict (sorted по prob_true asc):
        text, true_label, pred_label, prob_true, prob_pred
    """
    if not hasattr(pipe, "predict_proba"):
        return []

    try:
        proba = pipe.predict_proba(X)
    except (AttributeError, ValueError, RuntimeError) as _proba_exc:
        _log.warning("predict_proba failed in ROC-AUC section: %s", _proba_exc)
        return []

    classes = list(pipe.classes_)
    _cls_idx = {c: i for i, c in enumerate(classes)}

    results: List[Dict[str, Any]] = []
    for i, (xi, yi) in enumerate(zip(X, y)):
        ci = _cls_idx.get(str(yi), -1)
        if ci < 0:
            continue
        p_true = float(proba[i, ci])
        if p_true >= threshold:
            continue
        pi = int(proba[i].argmax())
        results.append({
            "prob_true":  round(p_true, 4),
            "prob_pred":  round(float(proba[i, pi]), 4),
            "true_label": str(yi),
            "pred_label": str(classes[pi]),
            "text":       str(xi)[:500],
        })
        if len(results) >= max_results:
            break

    results.sort(key=lambda r: r["prob_true"])
    if log_cb:
        log_cb(
            f"[Подозрительные метки] найдено {len(results)} примеров "
            f"(prob_true < {threshold:.0%})"
        )
    return results


def _format_classification_report(rep_dict: Dict[str, Any], digits: int = 3) -> str:
    """Формирует текстовый отчёт классификации из уже посчитанного dict'а.

    Позволяет избежать двойного вызова sklearn.classification_report (раз для
    текста, раз для dict'а). Формат совместим с sklearn, но строится за один
    проход по готовым метрикам.
    """
    fmt = f"{{:.{digits}f}}"
    rows: list[str] = []
    headers = ("precision", "recall", "f1-score", "support")
    w_label = max([10] + [len(str(k)) for k in rep_dict if k not in ("accuracy",)])
    rows.append(
        f"{'':>{w_label}}  "
        + "  ".join(f"{h:>9}" for h in headers)
    )
    rows.append("")
    # Классы — всё кроме специальных ключей
    special = {"accuracy", "macro avg", "weighted avg"}
    for label, metrics in rep_dict.items():
        if label in special or not isinstance(metrics, dict):
            continue
        rows.append(
            f"{str(label):>{w_label}}  "
            f"{fmt.format(metrics.get('precision', 0.0)):>9}  "
            f"{fmt.format(metrics.get('recall', 0.0)):>9}  "
            f"{fmt.format(metrics.get('f1-score', 0.0)):>9}  "
            f"{int(metrics.get('support', 0)):>9}"
        )
    rows.append("")
    if "accuracy" in rep_dict:
        # accuracy — скаляр; в dict поддержка — общая сумма примеров macro avg
        _acc = rep_dict["accuracy"]
        _support = int(rep_dict.get("macro avg", {}).get("support", 0)) if isinstance(
            rep_dict.get("macro avg"), dict) else 0
        rows.append(f"{'accuracy':>{w_label}}  {'':>9}  {'':>9}  {fmt.format(_acc):>9}  {_support:>9}")
    for agg in ("macro avg", "weighted avg"):
        m = rep_dict.get(agg)
        if isinstance(m, dict):
            rows.append(
                f"{agg:>{w_label}}  "
                f"{fmt.format(m.get('precision', 0.0)):>9}  "
                f"{fmt.format(m.get('recall', 0.0)):>9}  "
                f"{fmt.format(m.get('f1-score', 0.0)):>9}  "
                f"{int(m.get('support', 0)):>9}"
            )
    return "\n".join(rows)


def _compute_per_class_thresholds(
    proba: np.ndarray,
    classes: List[str],
    yva: List[str],
) -> Dict[str, float]:
    """Выбирает оптимальный порог на PR-кривой для каждого класса (максимизирует F0.5)."""
    from sklearn.metrics import precision_recall_curve

    def _f05(tpr):
        _, p, r = tpr
        denom = 0.25 * p + r
        return 1.25 * p * r / denom if denom > 0 else 0.0

    _yva_arr = np.array(yva)
    per_class_thresh: Dict[str, float] = {}
    for _i, _cls in enumerate(classes):
        _cls_proba = proba[:, _i]
        _cls_true = (_yva_arr == _cls).astype(int)
        if _cls_true.sum() < 2:
            per_class_thresh[str(_cls)] = 0.5
            continue
        _precisions, _recalls, _thresholds = precision_recall_curve(_cls_true, _cls_proba)
        _good = [
            (float(t), float(p), float(r))
            for p, r, t in zip(_precisions, _recalls, _thresholds)
            if p >= PR_MIN_PRECISION and r > 0
        ]
        if _good:
            per_class_thresh[str(_cls)] = max(_good, key=_f05)[0]
        elif len(_thresholds) > 0:
            per_class_thresh[str(_cls)] = float(np.median(_thresholds))
        else:
            per_class_thresh[str(_cls)] = 0.5
    return per_class_thresh


def _compute_validation_extras(
    pipe: Pipeline,
    Xva: List[str],
    yva: List[str],
    log_cb: Optional[Callable[[str], None]],
) -> Tuple[Dict[str, Any], List[str], List[str], Any, str, Dict]:
    """Вычисляет метрики валидации: предсказания, отчёт, пороги, ROC-AUC,
    температурное масштабирование и примеры путаниц.

    Возвращает: extras, pred, labels, cm, rep, rep_dict
    """
    extras: Dict[str, Any] = {}
    classes: List[str] = []

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(Xva)
        classes = list(pipe.classes_)
        pred = [classes[int(row.argmax())] for row in proba]
        val_confs = np.array([float(row.max()) for row in proba])
        if len(val_confs) > 0:
            extras["thresh_90"] = float(np.percentile(val_confs, CONF_THRESH_90_PERCENTILE))
            extras["thresh_75"] = float(np.percentile(val_confs, CONF_THRESH_75_PERCENTILE))
            extras["thresh_50"] = float(np.percentile(val_confs, CONF_THRESH_50_PERCENTILE))
    else:
        proba = None
        pred = list(pipe.predict(Xva))

    # Единый вызов classification_report — строим и dict, и текстовую форму
    # из одного прохода, избегая двойного пересчёта метрик.
    rep_dict = classification_report(yva, pred, digits=3, output_dict=True, zero_division=0)
    rep = _format_classification_report(rep_dict, digits=3)
    extras["report_dict"] = rep_dict

    if log_cb:
        _f1  = rep_dict.get("macro avg", {}).get("f1-score")
        _acc = rep_dict.get("accuracy")
        _f1_s  = f"{_f1:.3f}"  if _f1  is not None else "?"
        _acc_s = f"{_acc:.3f}" if _acc is not None else "?"
        log_cb(f"[Валидация] macro F1={_f1_s} | accuracy={_acc_s} | val_size={len(yva)}")

    labels = sorted(set(yva))
    cm = confusion_matrix(yva, pred, labels=labels)

    # Per-class пороги через Precision-Recall
    if proba is not None and classes:
        extras["per_class_thresholds"] = _compute_per_class_thresholds(proba, classes, yva)

        # ROC-AUC macro (OvR)
        try:
            from sklearn.metrics import roc_auc_score as _roc_auc
            _roc = _roc_auc(yva, proba, multi_class="ovr", average="macro", labels=classes)
            extras["roc_auc_macro"] = float(_roc)
            if log_cb:
                log_cb(f"[Валидация] ROC-AUC macro={float(_roc):.3f}")
        except (ValueError, AttributeError, ImportError) as _roc_exc:
            if log_cb:
                log_cb(f"[Валидация] ROC-AUC недоступен: {type(_roc_exc).__name__}: {_roc_exc}")

    # Предупреждение о классах с низким качеством предсказания
    if log_cb and rep_dict:
        _weak = [
            (cls, d["precision"], d["recall"], d["f1-score"])
            for cls, d in rep_dict.items()
            if isinstance(d, dict) and "f1-score" in d
               and cls not in ("macro avg", "weighted avg", "accuracy")
               and d.get("f1-score", 1.0) < 0.50
               and d.get("support", 0) > 0
        ]
        if _weak:
            _weak.sort(key=lambda x: x[3])
            _weak_lines = "\n".join(
                f"    «{cls}»  prec={prec:.2f}  rec={rec:.2f}  f1={f1:.2f}"
                for cls, prec, rec, f1 in _weak[:10]
            )
            log_cb(f"[⚠ Слабые классы] F1 < 0.50 ({len(_weak)} шт.):\n{_weak_lines}")
        else:
            _ok_f1 = rep_dict.get("macro avg", {}).get("f1-score", 0.0)
            if _ok_f1 >= 0.70:
                log_cb(f"[✓ Качество] Все классы имеют F1 ≥ 0.50 | macro F1={_ok_f1:.3f}")

    # ── Температурное масштабирование: калибровка вероятностей ──
    if proba is not None and classes:
        try:
            _T = compute_temperature_scaling(proba, yva, classes)
            extras["temperature"] = _T
            if log_cb and abs(_T - 1.0) > 0.05:
                _dir = "← вероятности занижены" if _T > 1.0 else "← вероятности завышены"
                log_cb(f"[Калибровка] температура T={_T:.3f} {_dir}")
            # ECE / MCE — качество калибровки вероятностей
            _ece, _mce = _compute_ece_mce(proba, yva, classes)
            extras["ece"] = _ece
            extras["mce"] = _mce
            _brier = _compute_brier_score(proba, yva, classes)
            extras["brier"] = _brier
            _pc_ece = _compute_per_class_ece(proba, yva, classes)
            extras["per_class_ece"] = _pc_ece
            if log_cb:
                _cal_q = "хорошо" if _ece < 0.05 else ("удовл." if _ece < 0.10 else "плохо")
                log_cb(
                    f"[Калибровка] ECE={_ece:.4f}  MCE={_mce:.4f}  "
                    f"Brier={_brier:.4f}  ({_cal_q})"
                )
                if _pc_ece:
                    _worst = sorted(_pc_ece.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    _lines = ", ".join(f"«{c}»={v:.3f}" for c, v in _worst)
                    log_cb(f"[Калибровка] Худшие per-class ECE: {_lines}")
        except Exception as _te:  # noqa: BLE001 — calibration is diagnostic-only; any failure must not block training
            _log.debug("temperature scaling failed: %s", _te)

    # ── Анализ путаниц: топ ошибочных пар с примерами текстов ──
    if cm is not None and labels and Xva:
        _conf_pairs: list = []
        for _ci, _ti in enumerate(labels):
            for _cj, _pj in enumerate(labels):
                if _ci != _cj and cm[_ci][_cj] > 0:
                    _conf_pairs.append((int(cm[_ci][_cj]), str(_ti), str(_pj)))
        _conf_pairs.sort(reverse=True)

        _conf_examples: Dict[str, Any] = {}
        for _cnt_v, _ti, _pj in _conf_pairs[:10]:
            _key = f"{_ti} → {_pj}"
            _exs = [
                str(Xva[_k])[:200]
                for _k, (_yt, _yp) in enumerate(zip(yva, pred))
                if str(_yt) == _ti and str(_yp) == _pj
            ][:3]
            _conf_examples[_key] = {"count": _cnt_v, "true": _ti, "pred": _pj, "examples": _exs}
        extras["confusion_examples"] = _conf_examples

        if log_cb and _conf_pairs:
            _top5 = _conf_pairs[:5]
            _lines = "\n".join(
                f"    {cnt_v}×  TRUE=«{ti}» → PRED=«{pj}»"
                for cnt_v, ti, pj in _top5
            )
            log_cb(f"[Топ путаниц] (TRUE→PRED):\n{_lines}")

    return extras, pred, labels, cm, rep, rep_dict


def _compute_ece_mce(
    proba: np.ndarray,
    yva: List[str],
    classes: List[str],
    n_bins: int = 10,
) -> tuple:
    """Expected Calibration Error (ECE) и Maximum Calibration Error (MCE).

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    MCE = max_b |acc(B_b) - conf(B_b)|

    Returns (ece, mce) как float. Возвращает (0.0, 0.0) при ошибке.
    """
    try:
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx.get(str(yi), 0) for yi in yva])
        # confidence = max predicted probability; correctness = argmax == true
        confidences = proba.max(axis=1)
        predictions = proba.argmax(axis=1)
        correct = (predictions == y_idx).astype(float)
        # Adaptive bin count: avoid empty bins on small val sets
        n_bins = max(3, min(n_bins, len(proba) // 20))
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_acc = 0.0
        mce_acc = 0.0
        n = len(confidences)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if not mask.any():
                continue
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            gap = abs(bin_acc - bin_conf)
            ece_acc += mask.sum() / n * gap
            mce_acc = max(mce_acc, gap)
        return float(ece_acc), float(mce_acc)
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as _ece_exc:
        _log.debug("ECE/MCE computation failed: %s", _ece_exc)
        return 0.0, 0.0


def _compute_brier_score(
    proba: np.ndarray,
    yva: List[str],
    classes: List[str],
) -> float:
    """Multi-class Brier Score (mean squared error против one-hot правды).

    Brier = (1/N) · Σ_i Σ_k (p_ik − y_ik)².
    Чем ниже, тем лучше. Для калиброванного бинарного классификатора идеал ≈ 0.
    """
    try:
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n, k = proba.shape
        one_hot = np.zeros_like(proba)
        for i, yi in enumerate(yva):
            j = class_to_idx.get(str(yi))
            if j is not None:
                one_hot[i, j] = 1.0
        diff = proba - one_hot
        return float(np.mean(np.sum(diff * diff, axis=1)))
    except (ValueError, KeyError, IndexError, TypeError) as _bs_exc:
        _log.debug("Brier computation failed: %s", _bs_exc)
        return 0.0


def _compute_per_class_ece(
    proba: np.ndarray,
    yva: List[str],
    classes: List[str],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Per-class ECE: один-против-всех binning вероятности класса c."""
    try:
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_to_idx.get(str(yi), -1) for yi in yva])
        n_bins = max(3, min(n_bins, len(proba) // 20))
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        out: Dict[str, float] = {}
        for j, cls in enumerate(classes):
            conf_j = proba[:, j]
            correct_j = (y_idx == j).astype(float)
            n = len(conf_j)
            if n == 0:
                out[str(cls)] = 0.0
                continue
            ece_j = 0.0
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (conf_j >= lo) & (conf_j < hi)
                if not mask.any():
                    continue
                bin_acc = correct_j[mask].mean()
                bin_conf = conf_j[mask].mean()
                ece_j += mask.sum() / n * abs(bin_acc - bin_conf)
            out[str(cls)] = float(ece_j)
        return out
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as _pc_exc:
        _log.debug("per-class ECE failed: %s", _pc_exc)
        return {}


def compute_temperature_scaling(
    proba: np.ndarray,
    yva: List[str],
    classes: List[str],
) -> float:
    """Подбирает температуру T, минимизирующую NLL на валидационной выборке.

    Использует степенное масштабирование (работает с вероятностями, без logits):
        p_cal_i = p_i^(1/T) / sum(p_j^(1/T))

    T > 1 — разглаживает распределение (модель была излишне уверена)
    T < 1 — заостряет распределение (модель была излишне неуверена)
    T = 1 — без изменений

    Returns:
        Оптимальная T (1.0 если эффект пренебрежимо мал).
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        return 1.0

    _class_to_idx = {c: i for i, c in enumerate(classes)}
    _y_idx = np.array([_class_to_idx.get(c, 0) for c in yva])

    def _nll(T: float) -> float:
        if T <= 0.01:
            return 1e10
        _scaled = np.power(np.clip(proba, 1e-10, 1.0), 1.0 / T)
        _sums = _scaled.sum(axis=1, keepdims=True)
        _sums[_sums == 0] = 1.0
        _p_cal = _scaled / _sums
        _p_correct = _p_cal[np.arange(len(_y_idx)), _y_idx]
        return float(-np.mean(np.log(np.clip(_p_correct, 1e-10, 1.0))))

    _res = minimize_scalar(_nll, bounds=(0.1, 5.0), method="bounded")
    _T = float(_res.x)
    return _T if abs(_T - 1.0) > 0.02 else 1.0


def cv_evaluate(
    X: List[str],
    y: List[str],
    pipe: Pipeline,
    n_splits: int = 5,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Стратифицированная K-fold кросс-валидация макро F1.

    Pipe должен быть НЕ обученным — cross_val_score клонирует его для каждого фолда.
    Подходит только для TF-IDF моделей (SBERT слишком медленный для CV).

    Returns:
        dict с ключами cv_f1_mean, cv_f1_std, cv_scores (list), cv_n_splits
        или пустой dict при ошибке.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    _cnt = Counter(y)
    if min(_cnt.values(), default=0) < n_splits:
        # Уменьшаем число фолдов если мало примеров на класс
        n_splits = max(2, min(_cnt.values(), default=2))

    try:
        if progress_cb:
            progress_cb(92.0, f"Кросс-валидация ({n_splits}-fold)…")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        _pipe_clone = clone(pipe)
        scores = cross_val_score(
            _pipe_clone, X, y,
            cv=skf,
            scoring="f1_macro",
            n_jobs=1,
            error_score="raise",
        )
        result: Dict[str, Any] = {
            "cv_f1_mean":  float(scores.mean()),
            "cv_f1_std":   float(scores.std()),
            "cv_scores":   [round(float(s), 4) for s in scores],
            "cv_n_splits": n_splits,
        }
        if log_cb:
            _fold_str = "  ".join(f"{s:.3f}" for s in scores)
            log_cb(
                f"[CV {n_splits}-fold] macro F1 = {scores.mean():.3f} ± {scores.std():.3f}"
                f"  (фолды: {_fold_str})"
            )
        return result
    except Exception as _cve:  # noqa: BLE001 — CV scorer can raise sklearn internals (BrokenProcessPool, FitFailedWarning-as-error); log + return
        if log_cb:
            log_cb(f"[CV] не удалось выполнить: {type(_cve).__name__}: {_cve}")
        return {}


def _log_training_start(
    X: List[str],
    y: List[str],
    clf_type: str,
    C: float,
    max_iter: int,
    balanced: bool,
    log_cb: Optional[Callable[[str], None]],
) -> None:
    """Логирует базовую статистику и, при необходимости, предупреждение о дисбалансе."""
    if log_cb is None:
        return
    n_classes = len(set(y))
    class_w = "balanced" if balanced else "None"
    log_cb(f"[Обучение] Выборка: {len(X)} примеров | классов: {n_classes}")
    log_cb(f"[Обучение] Классификатор: {clf_type} | C={C} | max_iter={max_iter} | class_weight={class_w}")
    cnt = Counter(y)
    if len(cnt) < 2:
        return
    max_c, min_c = max(cnt.values()), min(cnt.values())
    if min_c > 0 and max_c / min_c > 10:
        top_cls = max(cnt, key=cnt.get)
        bot_cls = min(cnt, key=cnt.get)
        log_cb(
            f"⚠️  [Дисбаланс] {top_cls!r}={max_c} vs {bot_cls!r}={min_c} "
            f"(ratio={max_c/min_c:.0f}x). Рекомендуется class_weight=balanced или SMOTE."
        )


def _apply_label_smoothing(
    ytr: List[str],
    *,
    eps: float,
    random_state: int,
    log_cb: Optional[Callable[[str], None]],
) -> List[str]:
    """Label smoothing для hinge/hard-label классификаторов: eps% примеров
    получают случайную метку (aka «label noise injection»). Эффект —
    регуляризация границ решений, что функционально близко к классическому
    label smoothing для soft-label loss.
    """
    if eps <= 0.0:
        return list(ytr)
    cls_all = sorted(set(ytr))
    if len(cls_all) < 2:
        return list(ytr)
    n_flip = int(len(ytr) * float(eps))
    if n_flip <= 0:
        return list(ytr)
    rng_ls = np.random.default_rng(int(random_state) + 7)
    idx_flip = rng_ls.choice(len(ytr), size=n_flip, replace=False)
    ytr_new = list(ytr)
    for i in idx_flip:
        others = [c for c in cls_all if c != ytr_new[i]]
        if others:
            ytr_new[i] = str(rng_ls.choice(others))
    if log_cb:
        log_cb(f"[Label-smoothing] перемаркировано {n_flip} примеров "
               f"(eps={eps:.2f})")
    return ytr_new


def _augment_training_data(
    Xtr: List[str],
    ytr: List[str],
    *,
    random_state: int,
    use_fuzzy_dedup: bool,
    fuzzy_dedup_threshold: int,
    use_smote: bool,
    oversample_strategy: str,
    max_dup_per_sample: int,
    use_hard_negatives: bool,
    use_field_dropout: bool,
    field_dropout_prob: float,
    field_dropout_copies: int,
    use_label_smoothing: bool,
    label_smoothing_eps: float,
    log_cb: Optional[Callable[[str], None]],
    progress_cb: Optional[Callable[[float, str], None]],
) -> Tuple[List[str], List[str]]:
    """Последовательность аугментаций обучающей выборки.

    Порядок применения важен: fuzzy_dedup → SMOTE → hard_negatives →
    field_dropout → label_smoothing. Каждая ступень опциональна и
    отключается соответствующим булевым флагом.
    """
    if use_fuzzy_dedup:
        Xtr, ytr, n_dedup = fuzzy_string_dedup(
            Xtr, ytr, threshold=int(fuzzy_dedup_threshold), log_cb=log_cb,
        )
        if n_dedup and log_cb:
            log_cb(f"[Обучение] После fuzzy-dedup: обучение={len(Xtr)}")

    if use_smote:
        Xtr, ytr = _oversample_rare_classes(
            Xtr, ytr, random_state, log_cb, progress_cb,
            max_dup_per_sample=max_dup_per_sample,
            strategy=oversample_strategy,
        )

    if use_hard_negatives:
        Xtr, ytr = _oversample_hard_negatives(
            Xtr, ytr, random_state=random_state, log_cb=log_cb,
        )

    if use_field_dropout:
        Xtr, ytr = _field_dropout_augment(
            Xtr, ytr,
            dropout_prob=float(field_dropout_prob),
            n_copies=int(field_dropout_copies),
            random_state=random_state,
            log_cb=log_cb,
        )

    if use_label_smoothing:
        ytr = _apply_label_smoothing(
            ytr,
            eps=float(label_smoothing_eps),
            random_state=random_state,
            log_cb=log_cb,
        )

    return Xtr, ytr


def _log_svd_explained_variance(
    pipe: Pipeline,
    log_cb: Optional[Callable[[str], None]],
) -> None:
    """Если в features-ступени есть TruncatedSVD, логирует explained variance."""
    if log_cb is None:
        return
    try:
        feat_step = pipe.named_steps.get("features")
        svd_step = None
        if feat_step is not None:
            if hasattr(feat_step, "named_steps"):
                svd_step = feat_step.named_steps.get("svd")
            elif hasattr(feat_step, "transformer_list"):
                for _tname, tr in feat_step.transformer_list:
                    if hasattr(tr, "named_steps"):
                        svd_step = tr.named_steps.get("svd")
                        if svd_step is not None:
                            break
        if svd_step is not None and hasattr(svd_step, "explained_variance_ratio_"):
            var_total = float(svd_step.explained_variance_ratio_.sum())
            n_comp = len(svd_step.explained_variance_ratio_)
            log_cb(f"[SVD] объяснённая дисперсия: {var_total:.3f} ({n_comp} компонент)")
    except Exception as svd_exc:  # noqa: BLE001 — diagnostic logging only; skip silently on any inspection failure
        log_cb(f"[SVD] диагностика недоступна: {type(svd_exc).__name__}: {svd_exc}")


def _estimate_model_size_bytes(
    pipe: Pipeline,
    log_cb: Optional[Callable[[str], None]],
) -> Optional[int]:
    """Быстрая аналитическая оценка размера модели (uncompressed).

    Обходит TfidfVectorizer / TruncatedSVD / linear классификаторы и
    суммирует `vocabulary_ bytes + idf_.nbytes + components_.nbytes +
    coef_.nbytes + intercept_.nbytes`. Это в ~50 раз быстрее чем
    `joblib.dump(..., compress=0)` (которому нужно сериализовать весь граф)
    и даёт близкую к реальности оценку без I/O.
    """
    try:
        import numpy as _np

        total = 0

        def _vocab_bytes(voc) -> int:
            # Упрощённая оценка: средняя длина ключа в UTF-8 + int index (8 байт) + dict overhead (~50).
            if not voc:
                return 0
            n = len(voc)
            avg_len = sum(len(k.encode("utf-8")) for k in list(voc.keys())[:200]) / max(1, min(200, n))
            return int(n * (avg_len + 8 + 50))

        def _arr_bytes(obj, attr) -> int:
            a = getattr(obj, attr, None)
            if a is None:
                return 0
            nb = getattr(a, "nbytes", None)
            if nb is not None:
                return int(nb)
            try:
                return int(_np.asarray(a).nbytes)
            except Exception:  # noqa: BLE001
                return 0

        def _walk(obj):
            nonlocal total
            # TF-IDF: vocab + idf
            if hasattr(obj, "vocabulary_"):
                total += _vocab_bytes(getattr(obj, "vocabulary_", None))
                total += _arr_bytes(obj, "idf_")
            # TruncatedSVD: components + explained_variance
            if hasattr(obj, "components_"):
                total += _arr_bytes(obj, "components_")
                total += _arr_bytes(obj, "explained_variance_")
            # Linear / calibrated classifiers
            for attr in ("coef_", "intercept_", "classes_"):
                if hasattr(obj, attr):
                    total += _arr_bytes(obj, attr)
            # CalibratedClassifierCV wraps sub-estimators
            for sub_attr in ("calibrated_classifiers_", "estimators_"):
                subs = getattr(obj, sub_attr, None)
                if subs:
                    for sub in subs:
                        _walk(sub)
                        # Calibrator arrays on each fold
                        for cal_attr in ("calibrators_", "calibrators"):
                            cals = getattr(sub, cal_attr, None)
                            if cals:
                                for cal in cals:
                                    for a in ("a_", "b_", "X_thresholds_", "y_thresholds_"):
                                        total += _arr_bytes(cal, a)
            # Recurse: Pipeline / FeatureUnion
            if hasattr(obj, "named_steps"):
                for step in obj.named_steps.values():
                    _walk(step)
            if hasattr(obj, "transformer_list"):
                for _name, tr in obj.transformer_list:
                    _walk(tr)

        _walk(pipe)
        # Fixed overhead for pickle framing / class refs. Empirically
        # observed 15-30 KB on small pipes.
        size = int(total) + 20_000
        if log_cb:
            log_cb(f"[Обучение] Размер модели (оценка): {size // 1024} КБ")
        return size
    except Exception as sz_exc:  # noqa: BLE001 — size estimate is telemetry; skip on exotic pipes
        _log.debug("model_size_bytes estimation failed: %s", sz_exc)
        return None


def train_model(
    X: List[str],
    y: List[str],
    features: Any,  # FeatureUnion (TF-IDF) или SBERTVectorizer
    C: float,
    max_iter: int,
    balanced: bool,
    test_size: float,
    random_state: int,
    calib_method: str = "sigmoid",
    progress_cb: Optional[Callable[[float, str], None]] = None,
    use_smote: bool = True,
    oversample_strategy: str = "cap",
    max_dup_per_sample: int = 5,
    log_cb: Optional[Callable[[str], None]] = None,
    run_cv: bool = False,
    use_hard_negatives: bool = False,
    use_field_dropout: bool = False,
    field_dropout_prob: float = 0.15,
    field_dropout_copies: int = 2,
    use_label_smoothing: bool = False,
    label_smoothing_eps: float = 0.05,
    use_fuzzy_dedup: bool = False,
    fuzzy_dedup_threshold: int = 92,
) -> Tuple[Pipeline, str, str, Optional[List[str]], Optional[Any], Dict]:
    """
    Обучает Pipeline(features → classifier).

    Функция оркестрирует стадии, каждая из которых вынесена в отдельную
    helper-функцию:
      _log_training_start       — диагностика и предупреждение о дисбалансе
      cv_evaluate               — опциональная K-fold кросс-валидация
      _maybe_skip_validation    — ранний выход при маленькой выборке
      _augment_training_data    — fuzzy_dedup → SMOTE → hard_negatives →
                                  field_dropout → label_smoothing
      _log_svd_explained_variance
      _compute_validation_extras — метрики holdout
      _estimate_model_size_bytes

    Returns:
        pipe        — обученный sklearn Pipeline
        clf_type    — строка с типом классификатора
        report      — текстовый classification_report (или сообщение о пропуске валидации)
        labels      — список классов (или None если валидация пропущена)
        cm          — confusion matrix (или None если валидация пропущена)
        extras      — доп. данные: {'thresh_90', 'thresh_75', 'report_dict', ...}
    """
    clf, clf_type = make_classifier(y, C=C, max_iter=max_iter, balanced=balanced,
                                    calib_method=calib_method)
    pipe = Pipeline([("features", features), ("clf", clf)])

    _log_training_start(X, y, clf_type, C, max_iter, balanced, log_cb)

    # ── Кросс-валидация ДО обучения (pipe ещё не обучен) ──────────────────
    _cv_results: Dict[str, Any] = {}
    if run_cv:
        _cv_results = cv_evaluate(X, y, pipe, n_splits=5, log_cb=log_cb,
                                  progress_cb=progress_cb)

    if progress_cb:
        progress_cb(60.0, "Разделение обучение/тест…")

    # Безопасный holdout: только если достаточно данных и классов.
    # stratify=y требует минимум 2 примера на класс — иначе ValueError.
    early = _maybe_skip_validation(X, y, test_size, pipe, clf_type, log_cb, progress_cb)
    if early is not None:
        # early = (pipe, clf_type, rep, None, None, extras_dict)
        early[-1].update(_cv_results)
        return early

    Xtr, Xva, ytr, yva = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )
    if log_cb:
        log_cb(f"[Обучение] Разбивка: обучение={len(Xtr)}, валидация={len(Xva)} (test_size={test_size})")

    Xtr, ytr = _augment_training_data(
        Xtr, ytr,
        random_state=random_state,
        use_fuzzy_dedup=use_fuzzy_dedup,
        fuzzy_dedup_threshold=fuzzy_dedup_threshold,
        use_smote=use_smote,
        oversample_strategy=oversample_strategy,
        max_dup_per_sample=max_dup_per_sample,
        use_hard_negatives=use_hard_negatives,
        use_field_dropout=use_field_dropout,
        field_dropout_prob=field_dropout_prob,
        field_dropout_copies=field_dropout_copies,
        use_label_smoothing=use_label_smoothing,
        label_smoothing_eps=label_smoothing_eps,
        log_cb=log_cb,
        progress_cb=progress_cb,
    )

    if progress_cb:
        progress_cb(80.0, "Обучение…")
    if log_cb:
        log_cb(f"[Обучение] Запуск обучения ({clf_type}) на {len(Xtr)} примерах…")
    import time as _time
    _t_fit_start = _time.monotonic()
    pipe.fit(Xtr, ytr)
    _training_duration_sec = _time.monotonic() - _t_fit_start
    if log_cb:
        log_cb(f"[Обучение] Время обучения: {_training_duration_sec:.1f}с")

    _log_svd_explained_variance(pipe, log_cb)

    if progress_cb:
        progress_cb(90.0, "Валидация…")

    extras, pred, labels, cm, rep, _ = _compute_validation_extras(pipe, Xva, yva, log_cb)
    extras["n_train"] = len(Xtr)
    extras["n_test"] = len(Xva)
    extras["training_duration_sec"] = round(_training_duration_sec, 3)
    _model_size = _estimate_model_size_bytes(pipe, log_cb)
    if _model_size is not None:
        extras["model_size_bytes"] = _model_size
    extras.update(_cv_results)

    return pipe, clf_type, rep, labels, cm, extras


def find_best_c(
    X: List[str],
    y: List[str],
    features: Any,
    balanced: bool,
    max_iter: int,
    candidates: Optional[List[float]] = None,
    cv: int = 5,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[float, Dict[float, float]]:
    """
    Подбирает оптимальный C для LinearSVC через стратифицированную кросс-валидацию.

    Метрика оптимизации: macro F1 (устойчива к дисбалансу классов).

    Returns:
        best_c   — лучшее найденное значение C
        scores   — {C: mean_macro_f1}
    """
    import time as _time
    import os as _os
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Адаптивный n_jobs: None → половина ядер (безопасный дефолт для shared/виртуальных сред).
    # Caller может передать hw_profile.n_jobs_cv для machine-aware управления.
    if n_jobs is None:
        n_jobs = max(1, (_os.cpu_count() or 2) // 2)

    if candidates is None:
        n = len(X)
        if n < 500:
            candidates = [0.1, 0.3, 0.7, 1.0, 3.0, 5.0, 10.0, 20.0, 30.0]
        elif n < 2000:
            candidates = [0.05, 0.1, 0.3, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0]
        else:
            candidates = [0.01, 0.05, 0.1, 0.3, 0.7, 1.0, 2.0, 3.0, 5.0]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores: Dict[float, float] = {}
    n = len(candidates)
    _t0 = _time.time()

    for i, c in enumerate(candidates):
        if cancel_event is not None and cancel_event.is_set():
            break
        if progress_cb:
            # ETA: среднее время одной итерации × оставшиеся итерации
            elapsed = _time.time() - _t0
            if i > 0:
                avg_sec = elapsed / i
                eta_sec = avg_sec * (n - i)
                eta_str = f" | ост. ~{int(eta_sec // 60)}м {int(eta_sec % 60):02d}с"
            else:
                eta_str = ""
            progress_cb(
                10.0 + 80.0 * i / n,
                f"GridSearch C={c}: {i+1}/{n}{eta_str}",
            )
        clf = LinearSVC(
            C=c,
            max_iter=max_iter,
            class_weight=("balanced" if balanced else None),
        )
        pipe = Pipeline([("features", features), ("clf", clf)])
        try:
            cv_scores = cross_val_score(
                pipe, X, y,
                cv=skf,
                scoring="f1_macro",
                n_jobs=n_jobs,
            )
            scores[c] = float(np.mean(cv_scores))
        except (MemoryError, KeyboardInterrupt) as _me:
            if isinstance(_me, MemoryError):
                import gc as _gc
                _gc.collect()   # освобождаем то, что можно, до выхода из функции
            raise
        except Exception as _e:  # noqa: BLE001 — sklearn CV for a single C can fail for pathological splits; mark C=0 and continue
            _log.warning("GridSearch: cross_val_score failed for C=%s: %s", c, _e)
            scores[c] = 0.0

    if not scores:
        # Отменено до завершения первой итерации — возвращаем первый кандидат по умолчанию
        return candidates[0], scores

    best_c = max(scores, key=lambda c: scores[c])
    if progress_cb:
        progress_cb(100.0, f"GridSearch завершён: лучший C={best_c} (F1={scores[best_c]:.4f})")
    return best_c, scores


def optuna_tune(
    X: List[str],
    y: List[str],
    features: Any,
    balanced: bool,
    max_iter: int,
    n_trials: int = 30,
    cv: int = 4,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Подбирает гиперпараметры через Optuna (TPE-сэмплер).

    Оптимизирует: C для LinearSVC (широкий диапазон 0.001–100).
    При наличии TfidfVectorizer в pipeline — также ngram_range и max_features.

    Returns:
        best_params — {param: value}
        best_score  — лучший macro F1
    """
    import os as _os
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna не установлен. Выполните: pip install optuna")

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import copy

    if n_jobs is None:
        n_jobs = max(1, (_os.cpu_count() or 2) // 2)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    _completed = [0]

    def _objective(trial: "optuna.Trial") -> float:
        if cancel_event is not None and cancel_event.is_set():
            raise optuna.exceptions.TrialPruned()

        c_val = trial.suggest_float("C", 0.001, 100.0, log=True)

        clf = LinearSVC(
            C=c_val,
            max_iter=max_iter,
            class_weight=("balanced" if balanced else None),
        )

        # Попытка настроить TF-IDF если он есть в пайплайне
        try:
            feat_clone = copy.deepcopy(features)
            if hasattr(feat_clone, "named_steps"):
                for _name, _step in feat_clone.named_steps.items():
                    if hasattr(_step, "ngram_range"):
                        _ngram_min = trial.suggest_int("ngram_min", 1, 2)
                        _ngram_max = trial.suggest_int("ngram_max", 3, 6)
                        if _ngram_min <= _ngram_max:
                            _step.ngram_range = (_ngram_min, _ngram_max)
                        break
        except Exception as _feat_exc:  # noqa: BLE001 — sklearn FeatureUnion clone can raise varied errors; fall back to original features
            _log.debug("optuna feature clone failed: %s", _feat_exc)
            feat_clone = features

        pipe = Pipeline([("features", feat_clone), ("clf", clf)])
        try:
            cv_scores = cross_val_score(
                pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=n_jobs,
            )
            score = float(np.mean(cv_scores))
        except Exception as _cv_exc:  # noqa: BLE001 — Optuna trial must never crash outer study; score=0 signals bad config
            _log.debug("optuna cross_val_score failed: %s", _cv_exc)
            score = 0.0

        _completed[0] += 1
        if progress_cb:
            progress_cb(
                10.0 + 80.0 * _completed[0] / n_trials,
                f"Optuna: trial {_completed[0]}/{n_trials} | C={c_val:.4f} | F1={score:.4f}",
            )
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_objective, n_trials=n_trials,
                   callbacks=[lambda study, trial: None])

    best_params = study.best_params
    best_score = study.best_value
    if progress_cb:
        progress_cb(100.0, f"Optuna завершён: {best_params} | F1={best_score:.4f}")
    return best_params, best_score


def confident_learning_detect(
    X: List[str],
    y: List[str],
    features: Any,
    balanced: bool,
    max_iter: int,
    cv: int = 5,
    threshold_factor: float = 1.0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    n_jobs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Confident Learning: обнаружение ошибок разметки через кросс-валидацию.

    Алгоритм:
    1. Обучаем K-fold — получаем out-of-fold вероятности для каждого примера.
    2. Строим матрицу совместной частоты: C[истинный_класс][предсказанный_класс].
    3. Для каждого класса вычисляем порог уверенности = mean(p[true_class]).
    4. Пример считается мислейбленным если p[true_class] < порог × threshold_factor.

    Returns:
        Список словарей, отсортированных по убыванию подозрительности:
        [{idx, text, given_label, likely_label, p_given, p_likely}]
    """
    import os as _os
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV

    if n_jobs is None:
        n_jobs = max(1, (_os.cpu_count() or 2) // 2)

    classes = sorted(set(y))
    n_classes = len(classes)
    if n_classes < 2 or len(X) < cv * 2:
        return []

    cls_idx = {c: i for i, c in enumerate(classes)}
    y_arr = np.array([cls_idx[lbl] for lbl in y])
    X_arr = np.array(X)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), n_classes))

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        if cancel_event is not None and cancel_event.is_set():
            break
        if progress_cb:
            progress_cb(10.0 + 70.0 * fold_i / cv,
                        f"Confident Learning: fold {fold_i + 1}/{cv}…")
        X_tr = X_arr[tr_idx].tolist()
        y_tr = [y[i] for i in tr_idx]
        X_val = X_arr[val_idx].tolist()

        try:
            from collections import Counter as _Counter
            _cl_min_cls = min(_Counter(y_tr).values()) if y_tr else 1
            _cl_cv = min(3, max(2, _cl_min_cls))
            clf = LinearSVC(
                C=1.0, max_iter=max_iter,
                class_weight=("balanced" if balanced else None),
            )
            cal = CalibratedClassifierCV(clf, cv=_cl_cv)
            pipe = Pipeline([("features", features), ("clf", cal)])
            pipe.fit(X_tr, y_tr)
            fold_classes = list(pipe.classes_)
            fold_proba = pipe.predict_proba(X_val)
            for local_i, global_i in enumerate(val_idx):
                for local_ci, cls_name in enumerate(fold_classes):
                    global_ci = cls_idx.get(cls_name, -1)
                    if global_ci >= 0:
                        oof_proba[global_i, global_ci] = fold_proba[local_i, local_ci]
        except Exception as _e:  # noqa: BLE001 — single fold failure must not abort CL pipeline; other folds compensate
            _log.warning("Confident Learning fold %d failed: %s", fold_i, _e)

    # Пороги уверенности по классу
    thresholds = {}
    for cls_name in classes:
        ci = cls_idx[cls_name]
        mask = (y_arr == ci)
        if mask.sum() == 0:
            thresholds[cls_name] = 0.0
        else:
            thresholds[cls_name] = float(oof_proba[mask, ci].mean()) * threshold_factor

    # Поиск подозрительных примеров
    suspicious = []
    for i, (text, given_label) in enumerate(zip(X, y)):
        ci = cls_idx[given_label]
        p_given = float(oof_proba[i, ci])
        thr = thresholds[given_label]
        if p_given < thr:
            likely_ci = int(oof_proba[i].argmax())
            likely_label = classes[likely_ci]
            if likely_label != given_label:
                suspicious.append({
                    "idx": i,
                    "text": str(text)[:300],
                    "given_label": given_label,
                    "likely_label": likely_label,
                    "p_given": round(p_given, 4),
                    "p_likely": round(float(oof_proba[i, likely_ci]), 4),
                })

    suspicious.sort(key=lambda r: r["p_given"])
    if progress_cb:
        progress_cb(100.0, f"Confident Learning: {len(suspicious)} подозрительных примеров")
    return suspicious


def train_kfold_ensemble(
    X: List[str],
    y: List[str],
    features_factory: Callable[[], Any],
    balanced: bool,
    C: float,
    max_iter: int,
    k: int = 5,
    calib_method: str = "sigmoid",
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[List[Any], List[str]]:
    """
    Обучает K независимых моделей на разных фолдах и возвращает их.

    Каждая модель обучается на (K-1)/K данных — полный K-fold ансамбль.
    При инференсе усредняются вероятности от K моделей.

    Args:
        features_factory: callable () -> Pipeline или Transformer (вызывается K раз для независимых копий)
        k: число фолдов (= число моделей)

    Returns:
        (models, classes) — список обученных Pipeline и список меток классов
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    import copy

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    models = []
    classes_ref: List[str] = []
    X_arr = np.array(X)
    y_arr = np.array(y)

    for fold_i, (tr_idx, _) in enumerate(skf.split(X_arr, y_arr)):
        if cancel_event is not None and cancel_event.is_set():
            break
        if progress_cb:
            progress_cb(
                10.0 + 80.0 * fold_i / k,
                f"K-fold ансамбль: фолд {fold_i + 1}/{k}…",
            )
        X_tr = X_arr[tr_idx].tolist()
        y_tr = y_arr[tr_idx].tolist()

        feat = features_factory()
        from collections import Counter as _Counter
        _min_cls = min(_Counter(y_tr).values()) if y_tr else 1
        if _min_cls >= 3:
            clf = LinearSVC(
                C=C, max_iter=max_iter,
                class_weight=("balanced" if balanced else None),
            )
            _cal_cv = max(2, min(3, _min_cls))
            cal = CalibratedClassifierCV(clf, method=calib_method, cv=_cal_cv)
            pipe = Pipeline([("features", feat), ("clf", cal)])
        else:
            # Fallback для малых фолдов: LogisticRegression с нативными вероятностями
            lr = LogisticRegression(
                C=C, max_iter=max(max_iter, 500),
                class_weight=("balanced" if balanced else None),
                solver="lbfgs",
            )
            pipe = Pipeline([("features", feat), ("clf", lr)])
        pipe.fit(X_tr, y_tr)
        models.append(pipe)
        if not classes_ref:
            classes_ref = list(pipe.classes_)

    if progress_cb:
        progress_cb(100.0, f"K-fold ансамбль: обучено {len(models)} моделей")
    return models, classes_ref
