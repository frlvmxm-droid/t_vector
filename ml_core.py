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
from ml_diagnostics import (  # noqa: F401
    clean_training_data,
    dataset_health_checks,
    detect_near_duplicate_conflicts,
    extract_cluster_keywords,
    extract_cluster_keywords_ctfidf,
    extract_cluster_keywords_from_labels,
    find_cluster_representative_texts,
    merge_similar_clusters,
)
from ml_training import (  # noqa: F401
    _compute_validation_extras,
    _field_dropout_augment,
    _maybe_skip_validation,
    _oversample_hard_negatives,
    _oversample_rare_classes,
    compute_temperature_scaling,
    confident_learning_detect,
    cv_evaluate,
    detect_mislabeled_examples,
    find_best_c,
    make_classifier,
    optuna_tune,
    train_kfold_ensemble,
    train_model,
)
from ml_vectorizers import (  # noqa: F401
    DEBERTA_MODEL_IDS,
    SBERT_LOCAL_DIR,
    DeBERTaVectorizer,
    Lemmatizer,
    MetaFeatureExtractor,
    PerFieldSBERTVectorizer,
    PerFieldVectorizer,
    PhraseRemover,
    SBERTVectorizer,
    _BuiltinsPatch,
    _import_train_model_setfit,
    find_sbert_in_pipeline,
    find_setfit_classifier,
    make_hybrid_vectorizer,
    make_neural_vectorizer,
)
