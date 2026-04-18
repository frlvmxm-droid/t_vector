# -*- coding: utf-8 -*-
"""
ml_core — обратно-совместимый шим.

Все публичные имена реэкспортируются из специализированных модулей:
  ml_compat      — ранний патч torch.__version__
  ml_vectorizers — _BuiltinsPatch, SBERTVectorizer, PerFieldSBERTVectorizer,
                   DeBERTaVectorizer, make_neural_vectorizer, find_sbert_in_pipeline,
                   find_setfit_classifier, PhraseRemover, Lemmatizer,
                   MetaFeatureExtractor, PerFieldVectorizer, make_hybrid_vectorizer,
                   SBERT_LOCAL_DIR, DEBERTA_MODEL_IDS
  ml_training    — make_classifier, train_model, find_best_c
  ml_diagnostics — extract_cluster_keywords, extract_cluster_keywords_from_labels,
                   extract_cluster_keywords_ctfidf, clean_training_data,
                   dataset_health_checks, find_cluster_representative_texts,
                   merge_similar_clusters, detect_near_duplicate_conflicts

Прямой импорт из специализированных модулей предпочтителен для нового кода.
"""
from __future__ import annotations

# ml_compat импортируется первым — выполняет ранний патч torch.__version__
import ml_compat  # noqa: F401

from ml_vectorizers import (  # noqa: F401
    SBERT_LOCAL_DIR,
    _BuiltinsPatch,
    SBERTVectorizer,
    PerFieldSBERTVectorizer,
    DeBERTaVectorizer,
    DEBERTA_MODEL_IDS,
    make_neural_vectorizer,
    find_sbert_in_pipeline,
    find_setfit_classifier,
    _import_train_model_setfit,
    PhraseRemover,
    Lemmatizer,
    MetaFeatureExtractor,
    PerFieldVectorizer,
    make_hybrid_vectorizer,
)

from ml_training import (  # noqa: F401
    make_classifier,
    _maybe_skip_validation,
    _oversample_rare_classes,
    _oversample_hard_negatives,
    _field_dropout_augment,
    detect_mislabeled_examples,
    _compute_validation_extras,
    compute_temperature_scaling,
    cv_evaluate,
    train_model,
    find_best_c,
    optuna_tune,
    confident_learning_detect,
    train_kfold_ensemble,
)

from ml_diagnostics import (  # noqa: F401
    extract_cluster_keywords,
    extract_cluster_keywords_from_labels,
    extract_cluster_keywords_ctfidf,
    clean_training_data,
    dataset_health_checks,
    find_cluster_representative_texts,
    merge_similar_clusters,
    detect_near_duplicate_conflicts,
)
