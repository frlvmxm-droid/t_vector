"""
ml_diagnostics — диагностика данных и извлечение ключевых слов кластеров.

Содержит:
  extract_cluster_keywords            — ключевые слова по центроидам
  extract_cluster_keywords_from_labels — ключевые слова по меткам кластеров
  extract_cluster_keywords_ctfidf     — c-TF-IDF ключевые слова
  clean_training_data                 — дедупликация и фильтрация датасета
  dataset_health_checks               — проверки датасета на типичные проблемы
  find_cluster_representative_texts   — центроид-ближайшие тексты на кластер
  merge_similar_clusters              — слияние семантически близких кластеров
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from app_logger import get_logger
from ml_vectorizers import PerFieldVectorizer

_log = get_logger(__name__)

def extract_cluster_keywords(
    vec: Any,
    centers: Any,
    top_n: int = 12,
) -> list[str]:
    """
    Извлекает топ-N ключевых слов для каждого центроида кластера.
    Возвращает список строк (по одной на кластер).

    Поддерживает:
    - TfidfVectorizer — единый словарь get_feature_names_out()
    - PerFieldVectorizer — объединяет feature names из всех активных полей
    - sklearn Pipeline — ищет последний совместимый шаг
    - Прочие объекты — возвращает пустые строки вместо краша
    """
    # Извлекаем feature names из разных типов векторизаторов
    feat: np.ndarray | None = None

    if isinstance(vec, PerFieldVectorizer):
        # Объединяем имена признаков всех активных полей
        names: list[str] = []
        for _, tag, _ in vec._active:
            cv = vec._char_vecs.get(tag)
            wv = vec._word_vecs.get(tag)
            if cv is not None:
                names.extend(f"{tag.lower()}:char:{n}" for n in cv.get_feature_names_out())
            if wv is not None:
                names.extend(f"{tag.lower()}:word:{n}" for n in wv.get_feature_names_out())
        feat = np.array(names) if names else None
    elif hasattr(vec, "get_feature_names_out"):
        feat = vec.get_feature_names_out()
    elif hasattr(vec, "get_feature_names"):
        feat = np.array(vec.get_feature_names())

    kws: list[str] = []
    if feat is None or len(feat) == 0:
        return [""] * centers.shape[0]

    for i in range(centers.shape[0]):
        rowc = centers[i]
        # Центроиды могут иметь меньше элементов чем feat (напр. после SVD)
        if len(rowc) != len(feat):
            kws.append("")
            continue
        top_idx = rowc.argsort()[::-1][:top_n]
        words = [str(feat[j]) for j in top_idx if rowc[j] > 0]
        kws.append(", ".join(words[:top_n]))
    return kws


def extract_cluster_keywords_from_labels(
    vec: Any,
    Xv,
    labels,
    n_clusters: int,
    top_n: int = 12,
) -> list[str]:
    """
    Извлекает ключевые слова для каждого кластера по меткам принадлежности строк.

    Используется когда кластеризация выполнена на SBERT-эмбеддингах (плотные векторы),
    а TF-IDF матрица нужна только для интерпретации ключевых слов.

    vec        — TfidfVectorizer (уже обученный на тех же текстах)
    Xv         — TF-IDF матрица (n_samples × vocab), scipy sparse или np.ndarray
    labels     — массив меток кластеров длиной n_samples
    n_clusters — число кластеров K
    top_n      — сколько ключевых слов на кластер
    """
    feat = None
    if hasattr(vec, "get_feature_names_out"):
        feat = vec.get_feature_names_out()
    elif hasattr(vec, "get_feature_names"):
        feat = np.array(vec.get_feature_names())

    if feat is None or len(feat) == 0:
        return [""] * n_clusters

    labels_arr = np.asarray(labels)
    kws: list[str] = []
    for k in range(n_clusters):
        mask = (labels_arr == k)
        if not np.any(mask):
            kws.append("")
            continue
        cluster_rows = Xv[mask]
        # mean по строкам → (1, vocab) для sparse, (vocab,) для dense
        mean_vec = np.asarray(cluster_rows.mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[::-1][:top_n]
        words = [str(feat[j]) for j in top_idx if mean_vec[j] > 0]
        kws.append(", ".join(words[:top_n]))
    return kws


def extract_cluster_keywords_ctfidf(
    docs: list[str],
    labels,
    n_clusters: int,
    stop_words=None,
    top_n: int = 12,
    use_lemma: bool = True,
) -> list[str]:
    """
    c-TF-IDF для извлечения ключевых слов кластеров.

    Формула BERTopic: tf(t,c) * log(1 + |docs| / tf_all(t))
    где c — кластер, tf(t,c) — частота термина в кластере (нормализованная),
    tf_all(t) — суммарная частота по всему корпусу.

    Преимущество перед обычным TF-IDF mean:
      • Выделяет слова, специфичные для кластера vs. всего корпуса
      • Общие доменные слова («банк», «клиент») подавляются IDF-компонентой

    use_lemma=True — перед CountVectorizer пропускаем документы через
    pymorphy-лемматизатор. Склоняемые формы («снял», «снимаю», «снятие»)
    склеиваются в единую лемму «снять», что в русском в 2–3× уменьшает
    раздробленность словаря и качественно улучшает c-TF-IDF топ-слова.
    Если pymorphy3/2 не установлены — лемматизация прозрачно отключается.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    labels_arr = np.asarray(labels)
    n_docs = len(docs)
    docs_arr = np.asarray(docs, dtype=object)

    # Объединяем документы каждого кластера в один мега-документ
    cluster_docs: list[str] = []
    cluster_sizes = np.zeros(n_clusters, dtype=float)
    for k in range(n_clusters):
        mask = (labels_arr == k)
        cluster_sizes[k] = float(mask.sum())
        cluster_docs.append(" ".join(docs_arr[mask]))

    # Лемматизация склеивает словоформы до подачи в CountVectorizer.
    # Делаем здесь, а не на уровне отдельных documents — это дешевле:
    # n_clusters мега-документов вместо n_docs исходных.
    if use_lemma:
        try:
            from ml_vectorizers import Lemmatizer
            _lemm = Lemmatizer(include_pos=False).fit(cluster_docs)
            if getattr(_lemm, "is_active_", False):
                cluster_docs = list(_lemm.transform(cluster_docs))
        except (ImportError, RuntimeError, ValueError) as _lemma_exc:
            # ImportError: pymorphy2/3 absent. RuntimeError/ValueError:
            # dictionary load / UD-model quirks. Lemmatization is advisory,
            # so we fall back to raw tokens on any of these.
            _log.debug(
                "[c-TF-IDF] лемматизация недоступна (%s: %s), работаю без неё",
                type(_lemma_exc).__name__, _lemma_exc,
            )

    try:
        # ngram_range=(1,3): унаграммы + биграммы + триграммы — ключевые слова
        # вида "досрочное погашение" информативнее отдельных слов.
        # min_df=1: не фильтруем по числу кластеров, т.к. CountVectorizer видит
        # только n_clusters мега-документов; кластерно-специфичные термины
        # (встречающиеся в одном кластере) — это именно то, что нужно c-TF-IDF.
        cv = CountVectorizer(
            stop_words=stop_words,
            min_df=1,
            ngram_range=(1, 3),
            max_features=50_000,
        )
        X_counts = cv.fit_transform(cluster_docs)
    except ValueError as _ctfidf_exc:
        # sklearn CountVectorizer raises ValueError on empty-vocab / stop-words-only
        # inputs. Anything else (e.g. MemoryError on huge corpora) should propagate.
        _log.warning(
            "[c-TF-IDF] CountVectorizer не удался: %s: %s",
            type(_ctfidf_exc).__name__, _ctfidf_exc,
        )
        return [""] * n_clusters

    feat = cv.get_feature_names_out()
    X = X_counts.toarray().astype(float)

    # TF: нормализуем на размер кластера (число документов)
    sizes = np.maximum(cluster_sizes, 1.0)
    tf = X / sizes[:, np.newaxis]

    # IDF: log(1 + |all_docs| / total_term_count_across_clusters)
    tf_all = np.asarray(X.sum(axis=0)).ravel() + 1.0
    idf = np.log(1.0 + n_docs / tf_all)

    ctfidf = tf * idf[np.newaxis, :]

    kws: list[str] = []
    for k in range(n_clusters):
        top_idx = ctfidf[k].argsort()[::-1][:top_n]
        words = [str(feat[j]) for j in top_idx if ctfidf[k, j] > 0]
        kws.append(", ".join(words[:top_n]))
    return kws


def clean_training_data(
    X: list[str],
    y: list[str],
    min_samples_per_class: int = 2,
    drop_conflicts: bool = False,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Очищает обучающий датасет перед тренировкой.

    Шаг 1: Считает конфликты (один текст → разные метки).
            Если drop_conflicts=True — удаляет ВСЕ строки конфликтных текстов.
    Шаг 2: Удаляет точные дубли (x, label), сохраняя порядок первого вхождения.
    Шаг 3: Исключает классы с кол-вом примеров < min_samples_per_class.

    Returns:
        (X_clean, y_clean, report) где report содержит:
            n_before, n_duplicates, n_conflicts, n_conflict_rows_dropped,
            n_after_dedup, excluded_classes {label: count}, n_excluded_rows,
            n_final, min_samples
    """
    from collections import defaultdict as _dd

    # ── Конфликты: один текст → разные метки ──
    _x_to_labels: dict[str, set] = _dd(set)
    for _x, _lab in zip(X, y):
        _x_to_labels[_x].add(_lab)
    _conflict_texts = {_x for _x, _labs in _x_to_labels.items() if len(_labs) > 1}
    n_conflicts = len(_conflict_texts)

    # Удаляем конфликтные строки до дедупликации (если включено)
    if drop_conflicts and _conflict_texts:
        _pairs_in = list(zip(X, y))
        _pairs_in = [(_x, _lab) for _x, _lab in _pairs_in if _x not in _conflict_texts]
        X_in = [p[0] for p in _pairs_in]
        y_in = [p[1] for p in _pairs_in]
        n_conflict_rows_dropped = len(X) - len(X_in)
    else:
        X_in, y_in = list(X), list(y)
        n_conflict_rows_dropped = 0

    # ── Дедупликация точных (x, label) пар, сохраняем порядок ──
    _seen: set = set()
    X_dedup: list[str] = []
    y_dedup: list[str] = []
    n_dup = 0
    for _x, _lab in zip(X_in, y_in):
        _key = (_x, _lab)
        if _key in _seen:
            n_dup += 1
        else:
            _seen.add(_key)
            X_dedup.append(_x)
            y_dedup.append(_lab)

    # ── Исключение классов с недостаточным кол-вом примеров ──
    _cnt = Counter(y_dedup)
    _rare = {lab: c for lab, c in _cnt.items() if c < min_samples_per_class}
    if _rare:
        X_clean = [_x for _x, _lab in zip(X_dedup, y_dedup) if _lab not in _rare]
        y_clean = [_lab for _lab in y_dedup if _lab not in _rare]
    else:
        X_clean, y_clean = X_dedup, y_dedup

    return X_clean, y_clean, {
        "n_before":                 len(X),
        "n_duplicates":             n_dup,
        "n_conflicts":              n_conflicts,
        "n_conflict_rows_dropped":  n_conflict_rows_dropped,
        "n_after_dedup":            len(X_dedup),
        "excluded_classes":         dict(sorted(_rare.items(), key=lambda kv: kv[1])),
        "n_excluded_rows":          len(X_dedup) - len(X_clean),
        "n_final":                  len(X_clean),
        "min_samples":              min_samples_per_class,
    }


def detect_near_duplicate_conflicts(
    X: list[str],
    y: list[str],
    threshold: float = 0.92,
    max_pairs: int = 200,
    log_fn: Any | None = None,
) -> list[tuple[str, str, str, str, float]]:
    """Находит пары почти одинаковых текстов с разными метками.

    Использует TF-IDF (char 2-4 n-grams) + косинусное сходство.
    Для датасетов > 5 000 строк анализирует случайную выборку 5 000.

    Returns:
        Список кортежей (text1, text2, label1, label2, similarity),
        отсортированный по убыванию сходства.
    """
    if threshold <= 0 or len(X) < 2:
        return []

    import random as _random

    from sklearn.feature_extraction.text import TfidfVectorizer as _TfIdf
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    _max_analyze = 5000
    if len(X) > _max_analyze:
        if log_fn:
            log_fn(
                f"  [ND] Датасет большой ({len(X)} строк) — анализируем выборку {_max_analyze}"
            )
        _rng = _random.Random(42)
        _idx = sorted(_rng.sample(range(len(X)), _max_analyze))
        _X = [X[i] for i in _idx]
        _y = [y[i] for i in _idx]
    else:
        _X, _y = list(X), list(y)

    try:
        _vec = _TfIdf(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=20_000, min_df=1,
        )
        _M = _vec.fit_transform(_X)
    except ValueError:
        # sklearn raises ValueError on empty / stop-words-only corpora.
        return []

    _n = len(_X)
    _pairs: list[tuple[str, str, str, str, float]] = []
    _BLOCK = 500

    for _bi_start in range(0, _n, _BLOCK):
        _block = _M[_bi_start: _bi_start + _BLOCK]
        _sims = _cos_sim(_block, _M)
        for _bi in range(_sims.shape[0]):
            _gi = _bi_start + _bi
            for _j in range(_gi + 1, _n):
                _s = float(_sims[_bi, _j])
                if _s >= threshold and _y[_gi] != _y[_j]:
                    _pairs.append((_X[_gi], _X[_j], _y[_gi], _y[_j], _s))
                    if len(_pairs) >= max_pairs:
                        _pairs.sort(key=lambda _p: -_p[4])
                        return _pairs
        if len(_pairs) >= max_pairs:
            break

    _pairs.sort(key=lambda _p: -_p[4])
    return _pairs[:max_pairs]


def dataset_health_checks(
    stats: dict[str, Any],
    y: list[str],
) -> tuple[list[str], list[str]]:
    """
    Проверяет датасет на типичные проблемы.

    Returns:
        fatal — критические ошибки (обучение невозможно)
        warn  — предупреждения (обучение возможно, но качество под угрозой)
    """
    fatal: list[str] = []
    warn: list[str] = []

    if stats.get("rows_used", 0) <= 0:
        fatal.append("После фильтрации не осталось обучающих строк (пустые тексты и/или label).")

    n_classes = len(set(y))
    if n_classes < 2:
        fatal.append("Недостаточно разных классов (<2). Нужны минимум 2 причины.")

    cnt = Counter(y)
    if cnt:
        top_lab, top_c = cnt.most_common(1)[0]
        if top_c / max(1, len(y)) > 0.50:
            warn.append(
                f"Дисбаланс: '{top_lab}' = {top_c/len(y)*100:.1f}% строк. "
                f"Реком.: class_weight=balanced + больше примеров редких классов."
            )
        rare1 = sum(1 for c in cnt.values() if c == 1)
        if rare1:
            warn.append(f"Редких классов (1 пример): {rare1}. Реком.: 5–20 примеров на класс минимум.")

    roles_rate = stats.get("roles_found_rows", 0) / max(1, stats.get("rows_raw", 1))
    if stats.get("has_dialog_rows", 0) > 0 and roles_rate < 0.30:
        warn.append("Разметка ролей CLIENT/OPERATOR встречается редко (<30%). Качество по диалогам будет ниже.")

    return fatal, warn


def find_cluster_representative_texts(
    X_texts: list[str],
    labels: Any,
    vectors: Any,
    n_top: int = 5,
) -> dict[int, list[str]]:
    """Возвращает {cluster_id: [top_n_texts]} — тексты, ближайшие к центроиду кластера.

    Использует косинусное сходство. Работает со sparse (scipy) и dense (numpy) матрицами.
    Кластер -1 (шум HDBSCAN) пропускается.
    """
    import scipy.sparse
    from sklearn.metrics.pairwise import cosine_similarity

    labels_arr = np.asarray(labels)
    vecs = vectors.toarray() if scipy.sparse.issparse(vectors) else np.asarray(vectors)

    result: dict[int, list[str]] = {}
    for cid in sorted(set(labels_arr.tolist()) - {-1}):
        idx = np.where(labels_arr == cid)[0]
        if len(idx) == 0:
            result[cid] = []
            continue
        centroid = vecs[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, vecs[idx])[0]
        top = np.argsort(sims)[::-1][:n_top]
        result[cid] = [str(X_texts[idx[i]])[:300] for i in top if idx[i] < len(X_texts)]
    return result


def merge_similar_clusters(
    labels: Any,
    vectors: Any,
    threshold: float = 0.85,
    log_fn: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Сливает кластеры с косинусным сходством центроидов >= threshold.

    Алгоритм Union-Find: меньший кластер поглощается большим.
    Возвращает (new_labels, merge_info) где merge_info содержит:
        merges   — список (src_id, dst_id, cosine_sim)
        n_before — число кластеров до слияния
        n_after  — число кластеров после слияния
    Метки перенумеровываются последовательно (0, 1, 2, …).
    Кластер -1 (шум) не трогается.
    """
    import scipy.sparse
    from sklearn.metrics.pairwise import cosine_similarity

    labels_arr = np.asarray(labels).copy()
    unique = sorted(set(labels_arr.tolist()) - {-1})

    if len(unique) < 2:
        return labels_arr, {"merges": [], "n_before": len(unique), "n_after": len(unique)}

    vecs = vectors.toarray() if scipy.sparse.issparse(vectors) else np.asarray(vectors)
    centroids = np.array([vecs[labels_arr == c].mean(axis=0) for c in unique])
    sim = cosine_similarity(centroids)

    parent: dict[int, int] = {c: c for c in unique}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    merges: list[tuple[int, int, float]] = []
    for i, ci in enumerate(unique):
        for j, cj in enumerate(unique[i + 1:], i + 1):
            if sim[i, j] >= threshold:
                pi, pj = _find(ci), _find(cj)
                if pi != pj:
                    size_pi = int(np.sum(labels_arr == pi))
                    size_pj = int(np.sum(labels_arr == pj))
                    if size_pi >= size_pj:
                        parent[pj] = pi
                        merges.append((pj, pi, float(sim[i, j])))
                    else:
                        parent[pi] = pj
                        merges.append((pi, pj, float(sim[i, j])))

    if not merges:
        return labels_arr, {"merges": [], "n_before": len(unique), "n_after": len(unique)}

    new_labels = np.array([_find(int(lbl)) if lbl != -1 else -1 for lbl in labels_arr])
    unique_after = sorted(set(new_labels.tolist()) - {-1})
    remap: dict[int, int] = {old: new for new, old in enumerate(unique_after)}
    remap[-1] = -1
    final = np.array([remap[int(lbl)] for lbl in new_labels])

    info: dict[str, Any] = {
        "merges":   merges,
        "n_before": len(unique),
        "n_after":  len(unique_after),
    }
    if log_fn:
        for src, dst, s in merges:
            log_fn(f"  Объединяю кластеры {src}→{dst} (cosine={s:.3f})")
        log_fn(f"  Кластеров: {len(unique)} → {len(unique_after)}")
    return final, info


def rank_for_active_learning(
    X_texts: list[str],
    proba: Any,
    classes: list[str],
    *,
    top_n: int = 50,
    strategy: str = "entropy",
    per_class_quota: int | None = None,
) -> list[dict[str, Any]]:
    """Ранжирует примеры для активного обучения.

    Возвращает список словарей с полями:
      • idx         — индекс в X_texts
      • text        — исходный текст (до 300 символов)
      • best_label  — argmax-предсказание
      • best_prob   — вероятность лучшего класса
      • score       — метрика неуверенности (выше = полезнее для разметки)
      • strategy    — "entropy" | "margin" | "least_confident"

    Стратегии:
      entropy         — энтропия распределения вероятностей (Shannon H)
      margin          — разница между лучшим и вторым (меньше = сложнее)
      least_confident — 1 - P(best)

    per_class_quota — если задан, возвращает не более N примеров на класс
                      (предотвращает доминирование одного класса в очереди).
    """
    proba_arr = np.asarray(proba)
    n_samples = proba_arr.shape[0]
    eps = 1e-10

    if strategy == "entropy":
        p = np.clip(proba_arr, eps, 1.0)
        scores = -np.sum(p * np.log(p), axis=1)
    elif strategy == "margin":
        sorted_p = np.sort(proba_arr, axis=1)[:, ::-1]
        if sorted_p.shape[1] < 2:
            scores = 1.0 - sorted_p[:, 0]
        else:
            scores = 1.0 - (sorted_p[:, 0] - sorted_p[:, 1])
    else:  # least_confident
        scores = 1.0 - proba_arr.max(axis=1)

    best_idx = proba_arr.argmax(axis=1)
    best_probs = proba_arr[np.arange(n_samples), best_idx]

    order = np.argsort(scores)[::-1]
    results: list[dict[str, Any]] = []
    per_class_count: dict[str, int] = {}

    for i in order:
        if len(results) >= top_n:
            break
        label = str(classes[int(best_idx[i])]) if classes else str(int(best_idx[i]))
        if per_class_quota is not None:
            if per_class_count.get(label, 0) >= per_class_quota:
                continue
        per_class_count[label] = per_class_count.get(label, 0) + 1
        results.append({
            "idx":        int(i),
            "text":       str(X_texts[i])[:300] if i < len(X_texts) else "",
            "best_label": label,
            "best_prob":  float(best_probs[i]),
            "score":      float(scores[i]),
            "strategy":   strategy,
        })

    return results
