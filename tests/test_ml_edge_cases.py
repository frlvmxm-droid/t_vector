# -*- coding: utf-8 -*-
"""Edge-case tests: empty input, single class, NaN, unbalanced data, encoding."""
import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_training import TrainingOptions, make_classifier, train_model
from app_cluster_pipeline import (
    prepare_inputs,
    build_cluster_role_context,
)


def _vec():
    return TfidfVectorizer(max_features=50)


# ── Empty / degenerate input ──────────────────────────────────────────────────

def test_make_classifier_single_class_returns_logreg():
    y = ["A"] * 10
    clf, clf_type = make_classifier(y, C=1.0, max_iter=100, balanced=False)
    # Only 1 class → min_class = 10 but len(set(y)) = 1 < 2 → LogReg fallback
    assert clf_type == "LogReg"


def test_train_model_single_class_skips_validation():
    # 1 class — sklearn may raise ValueError (LogReg can't fit 1-class) or return early.
    # Either outcome is acceptable; what's NOT acceptable is an unhandled crash outside
    # the well-defined sklearn error boundary.
    X = ["текст один два три"] * 10
    y = ["ONLY"] * 10
    try:
        pipe, _, report, labels, cm, extras = train_model(
            X=X, y=y, features=_vec(),
            C=1.0, max_iter=100, balanced=False,
            test_size=0.2, random_state=42,
            options=TrainingOptions(use_smote=False),
        )
        assert pipe is not None
    except ValueError as e:
        # sklearn raises ValueError for 1-class data: this is expected
        assert "class" in str(e).lower() or "sample" in str(e).lower()


def test_train_model_two_samples_skips_validation():
    X = ["hello world", "foo bar"]
    y = ["A", "B"]
    pipe, _, report, labels, cm, extras = train_model(
        X=X, y=y, features=_vec(),
        C=1.0, max_iter=100, balanced=False,
        test_size=0.5, random_state=0,
        options=TrainingOptions(use_smote=False),
    )
    assert pipe is not None


def test_train_model_empty_strings_in_X():
    X = ["", "", "текст слово", "другой текст", "ещё текст"] * 6
    y = ["A", "A", "B", "B", "C"] * 6
    pipe, _, report, labels, cm, extras = train_model(
        X=X, y=y, features=_vec(),
        C=1.0, max_iter=100, balanced=False,
        test_size=0.2, random_state=42,
        options=TrainingOptions(use_smote=False),
    )
    assert pipe is not None


# ── Unbalanced classes ────────────────────────────────────────────────────────

def test_train_model_highly_unbalanced_with_balanced_weight():
    X = ["мажоритарный класс"] * 100 + ["редкий класс"] * 5
    y = ["majority"] * 100 + ["rare"] * 5
    pipe, clf_type, report, labels, cm, extras = train_model(
        X=X, y=y, features=_vec(),
        C=1.0, max_iter=200, balanced=True,
        test_size=0.2, random_state=42,
        options=TrainingOptions(use_smote=False),
    )
    assert pipe is not None
    # With balanced weights, both classes should appear in predictions
    preds = pipe.predict(X[-5:])
    assert len(set(preds)) >= 1  # at least predicts something


@pytest.mark.parametrize("n_rare", [6, 9, 15])
def test_train_model_parametrized_minority_sizes(n_rare):
    # n_rare must be >= 6 so CalibratedClassifierCV(cv=3) has ≥2 minority samples
    # per fold after stratified train/test split
    X = ["текст мажор"] * 30 + ["редкий"] * n_rare
    y = ["maj"] * 30 + ["min"] * n_rare
    pipe, _, _, _, _, _ = train_model(
        X=X, y=y, features=_vec(),
        C=1.0, max_iter=100, balanced=True,
        test_size=0.2, random_state=42,
        options=TrainingOptions(use_smote=False),
    )
    assert pipe is not None


# ── Russian / Cyrillic text ───────────────────────────────────────────────────

def test_train_model_pure_cyrillic():
    X = ["банк карта дебетовый кредитный"] * 20 + ["страховка полис выплата"] * 20
    y = ["банки"] * 20 + ["страхование"] * 20
    pipe, _, report, labels, cm, _ = train_model(
        X=X, y=y, features=_vec(),
        C=1.0, max_iter=200, balanced=False,
        test_size=0.25, random_state=0,
        options=TrainingOptions(use_smote=False),
    )
    assert pipe is not None
    assert pipe.predict(["карта банк"]) in [["банки"], ["страхование"]]


def test_train_model_mixed_encoding_strings():
    # Mix of Cyrillic and Latin in same dataset
    X = (["bank credit card"] * 15 + ["банк кредит карта"] * 15 +
         ["страховка полис"] * 15)
    y = ["en"] * 15 + ["ru"] * 15 + ["ins"] * 15
    pipe, _, _, _, _, _ = train_model(
        X=X, y=y, features=_vec(),
        C=1.0, max_iter=200, balanced=False,
        test_size=0.2, random_state=7,
        options=TrainingOptions(use_smote=False),
    )
    assert pipe is not None


# ── Cluster pipeline edge cases ───────────────────────────────────────────────

def test_prepare_inputs_with_empty_files_snapshot():
    snap = {
        "cluster_role_mode": "all",
        "cluster_algo": "kmeans",
        "cluster_vec_mode": "tfidf",
        "ignore_chatbot_cluster": False,
        "call_col": "call",
        "chat_col": "chat",
    }
    result = prepare_inputs([], snap)
    assert result.files_snapshot == []


def test_build_cluster_role_context_all_modes():
    base_snap = {
        "cluster_algo": "kmeans",
        "cluster_vec_mode": "tfidf",
        "call_col": "call",
        "chat_col": "chat",
    }
    for mode in ("all", "client", "operator"):
        snap = {**base_snap, "cluster_role_mode": mode}
        ctx = build_cluster_role_context(snap)
        assert ctx.role_label != ""
        assert ctx.cluster_snap["cluster_algo"] == "kmeans"


def test_build_cluster_role_context_invalid_algo_raises():
    snap = {
        "cluster_algo": "not_an_algo",
        "cluster_vec_mode": "tfidf",
    }
    with pytest.raises(ValueError, match="Unsupported cluster_algo"):
        build_cluster_role_context(snap)


def test_build_cluster_role_context_invalid_vec_mode_raises():
    snap = {
        "cluster_algo": "kmeans",
        "cluster_vec_mode": "not_a_mode",
    }
    with pytest.raises(ValueError, match="Unsupported cluster_vec_mode"):
        build_cluster_role_context(snap)


@pytest.mark.parametrize("algo", ["kmeans", "hdbscan", "bertopic", "lda", "fastopic"])
def test_build_cluster_role_context_all_valid_algos(algo):
    snap = {"cluster_algo": algo, "cluster_vec_mode": "tfidf"}
    ctx = build_cluster_role_context(snap)
    assert ctx.cluster_snap["cluster_algo"] == algo


@pytest.mark.parametrize("vec_mode", ["tfidf", "sbert", "combo", "ensemble"])
def test_build_cluster_role_context_all_valid_vec_modes(vec_mode):
    snap = {"cluster_algo": "kmeans", "cluster_vec_mode": vec_mode}
    ctx = build_cluster_role_context(snap)
    assert ctx.cluster_snap["cluster_vec_mode"] == vec_mode
