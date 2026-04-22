"""Top-N ключевые слова на кластер без зависимости от обученного vectorizer.

Порт десктопного closure `_cluster_step_keywords` (app_cluster.py ≈2534).
Для SBERT/combo/ensemble TF-IDF vectorizer недоступен в postprocess —
здесь мы строим лёгкий TF-IDF на лету и извлекаем центроидные топ-N
токены для каждого кластера.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def top_keywords_per_cluster(
    texts: Sequence[str],
    labels: Any,
    *,
    top_n: int = 10,
    vocab: Iterable[str] | None = None,
    min_df: int = 1,
    max_features: int = 20_000,
) -> dict[int, list[str]]:
    """Return ``{cluster_id: [kw1, kw2, ...]}`` using a fresh TF-IDF pass.

    Noise labels (``-1``) are skipped. Missing clusters (empty mask)
    get an empty list. Exceptions during vectorisation fall through to
    a plain-count fallback so callers always get a mapping.
    """
    import numpy as np

    labels_arr = np.asarray(labels)
    unique = [int(c) for c in np.unique(labels_arr) if c >= 0]
    if not texts or not unique:
        return {}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vocab_list = list(vocab) if vocab is not None else None
        tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features if vocab_list is None else None,
            vocabulary=vocab_list,
            sublinear_tf=True,
        )
        matrix = tfidf.fit_transform(list(texts))
        feature_names = tfidf.get_feature_names_out()
    except Exception:
        return {cid: [] for cid in unique}

    out: dict[int, list[str]] = {}
    for cid in unique:
        mask = labels_arr == cid
        if not mask.any():
            out[cid] = []
            continue
        row_mean = np.asarray(matrix[mask].mean(axis=0)).ravel()
        if row_mean.size == 0:
            out[cid] = []
            continue
        top_idx = np.argsort(row_mean)[::-1][:top_n]
        out[cid] = [str(feature_names[i]) for i in top_idx if row_mean[i] > 0.0]
    return out
