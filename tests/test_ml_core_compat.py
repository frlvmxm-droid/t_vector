# -*- coding: utf-8 -*-
"""
Backward-compatibility shim tests for ml_core.py.

Verifies that all public names documented in the shim's docstring remain
importable from ml_core and have the expected type (callable / class).

No ML training is performed here — this is purely an import-level test.
"""
from __future__ import annotations

import sys
from pathlib import Path
import inspect

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import ml_core  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _attr(name: str):
    """Return attribute from ml_core, or None if absent."""
    return getattr(ml_core, name, None)


def _is_callable(name: str) -> bool:
    val = _attr(name)
    return callable(val)


def _is_class(name: str) -> bool:
    val = _attr(name)
    return inspect.isclass(val)


# ===========================================================================
# ml_training re-exports
# ===========================================================================

class TestMlTrainingReexports:

    def test_make_classifier_importable(self):
        from ml_core import make_classifier  # noqa: F401
        assert callable(make_classifier)

    def test_train_model_importable(self):
        from ml_core import train_model  # noqa: F401
        assert callable(train_model)

    def test_find_best_c_importable(self):
        from ml_core import find_best_c  # noqa: F401
        assert callable(find_best_c)

    def test_cv_evaluate_importable(self):
        from ml_core import cv_evaluate  # noqa: F401
        assert callable(cv_evaluate)

    def test_optuna_tune_importable(self):
        from ml_core import optuna_tune  # noqa: F401
        assert callable(optuna_tune)

    def test_detect_mislabeled_examples_importable(self):
        from ml_core import detect_mislabeled_examples  # noqa: F401
        assert callable(detect_mislabeled_examples)

    def test_confident_learning_detect_importable(self):
        from ml_core import confident_learning_detect  # noqa: F401
        assert callable(confident_learning_detect)

    def test_train_kfold_ensemble_importable(self):
        from ml_core import train_kfold_ensemble  # noqa: F401
        assert callable(train_kfold_ensemble)

    def test_make_classifier_is_function(self):
        assert _is_callable("make_classifier")
        assert not _is_class("make_classifier")

    def test_train_model_is_function(self):
        assert _is_callable("train_model")
        assert not _is_class("train_model")


# ===========================================================================
# ml_vectorizers re-exports
# ===========================================================================

class TestMlVectorizersReexports:

    def test_make_hybrid_vectorizer_importable(self):
        from ml_core import make_hybrid_vectorizer  # noqa: F401
        assert callable(make_hybrid_vectorizer)

    def test_lemmatizer_importable(self):
        from ml_core import Lemmatizer  # noqa: F401
        assert inspect.isclass(Lemmatizer)

    def test_meta_feature_extractor_importable(self):
        from ml_core import MetaFeatureExtractor  # noqa: F401
        assert inspect.isclass(MetaFeatureExtractor)

    def test_per_field_vectorizer_importable(self):
        from ml_core import PerFieldVectorizer  # noqa: F401
        assert inspect.isclass(PerFieldVectorizer)

    def test_sbert_vectorizer_importable(self):
        from ml_core import SBERTVectorizer  # noqa: F401
        assert inspect.isclass(SBERTVectorizer)

    def test_phrase_remover_importable(self):
        from ml_core import PhraseRemover  # noqa: F401
        assert inspect.isclass(PhraseRemover)

    def test_make_hybrid_vectorizer_is_function(self):
        assert _is_callable("make_hybrid_vectorizer")
        assert not _is_class("make_hybrid_vectorizer")

    def test_lemmatizer_is_class(self):
        assert _is_class("Lemmatizer")

    def test_meta_feature_extractor_is_class(self):
        assert _is_class("MetaFeatureExtractor")

    def test_per_field_vectorizer_is_class(self):
        assert _is_class("PerFieldVectorizer")


# ===========================================================================
# ml_diagnostics re-exports
# ===========================================================================

class TestMlDiagnosticsReexports:

    def test_extract_cluster_keywords_importable(self):
        from ml_core import extract_cluster_keywords  # noqa: F401
        assert callable(extract_cluster_keywords)

    def test_extract_cluster_keywords_from_labels_importable(self):
        from ml_core import extract_cluster_keywords_from_labels  # noqa: F401
        assert callable(extract_cluster_keywords_from_labels)

    def test_extract_cluster_keywords_ctfidf_importable(self):
        from ml_core import extract_cluster_keywords_ctfidf  # noqa: F401
        assert callable(extract_cluster_keywords_ctfidf)

    def test_clean_training_data_importable(self):
        from ml_core import clean_training_data  # noqa: F401
        assert callable(clean_training_data)

    def test_dataset_health_checks_importable(self):
        from ml_core import dataset_health_checks  # noqa: F401
        assert callable(dataset_health_checks)

    def test_find_cluster_representative_texts_importable(self):
        from ml_core import find_cluster_representative_texts  # noqa: F401
        assert callable(find_cluster_representative_texts)

    def test_merge_similar_clusters_importable(self):
        from ml_core import merge_similar_clusters  # noqa: F401
        assert callable(merge_similar_clusters)

    def test_detect_near_duplicate_conflicts_importable(self):
        from ml_core import detect_near_duplicate_conflicts  # noqa: F401
        assert callable(detect_near_duplicate_conflicts)


# ===========================================================================
# Module-level attribute checks (no explicit import statement)
# ===========================================================================

class TestModuleAttributes:

    @pytest.mark.parametrize("name", [
        "make_classifier",
        "train_model",
        "find_best_c",
        "make_hybrid_vectorizer",
        "extract_cluster_keywords",
        "clean_training_data",
        "merge_similar_clusters",
    ])
    def test_function_attribute_exists_and_callable(self, name: str):
        assert _is_callable(name), f"{name!r} is not callable in ml_core"

    @pytest.mark.parametrize("name", [
        "Lemmatizer",
        "MetaFeatureExtractor",
        "PerFieldVectorizer",
        "SBERTVectorizer",
        "PhraseRemover",
    ])
    def test_class_attribute_exists_and_is_type(self, name: str):
        assert _is_class(name), f"{name!r} is not a class in ml_core"

    def test_no_missing_names(self):
        """Spot-check that the most critical names are all present and non-None."""
        required = [
            "make_classifier", "train_model", "make_hybrid_vectorizer",
            "Lemmatizer", "extract_cluster_keywords",
        ]
        missing = [n for n in required if _attr(n) is None]
        assert missing == [], f"Missing from ml_core: {missing}"
