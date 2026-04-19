"""Direct coverage for the manual-validation fallback in workflow_contracts.

The Pydantic schemas are tested via TestTrainConfig/TestApplyConfig/
TestClusterConfig in ``test_workflow_contracts.py``. This file targets the
**fallback** branch (`_manual_validate_payload`) plus the small numeric
helpers that back it — both are uncovered when Pydantic is installed
because the public ``validate_*_payload`` entry points short-circuit to
the schema. They still ship in the module because Pydantic is an optional
runtime dep on Windows installers, so a regression here would silently
admit invalid snap-files on user machines without Pydantic.
"""

from __future__ import annotations

import pytest

from workflow_contracts import (
    _as_float,
    _as_int,
    _as_str,
    _manual_validate_payload,
    _require,
)


# --------------------------------------------------------------------------
# _require / _as_str
# --------------------------------------------------------------------------

def test_require_raises_on_missing_key() -> None:
    with pytest.raises(ValueError, match="Отсутствует обязательный параметр: foo"):
        _require({}, "foo")


def test_as_str_strips_whitespace() -> None:
    assert _as_str({"k": "  hello  "}, "k") == "hello"


def test_as_str_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="должен быть строкой"):
        _as_str({"k": 42}, "k")


def test_as_str_rejects_empty_after_strip() -> None:
    with pytest.raises(ValueError, match="не должен быть пустым"):
        _as_str({"k": "   "}, "k")


# --------------------------------------------------------------------------
# _as_float
# --------------------------------------------------------------------------

def test_as_float_rejects_non_numeric_string() -> None:
    with pytest.raises(ValueError, match="должен быть числом"):
        _as_float({"k": "not a number"}, "k")


def test_as_float_rejects_none() -> None:
    with pytest.raises(ValueError, match="должен быть числом"):
        _as_float({"k": None}, "k")


def test_as_float_rejects_inf() -> None:
    with pytest.raises(ValueError, match="конечным числом"):
        _as_float({"k": float("inf")}, "k")


def test_as_float_rejects_nan() -> None:
    with pytest.raises(ValueError, match="конечным числом"):
        _as_float({"k": float("nan")}, "k")


def test_as_float_min_max_bounds() -> None:
    with pytest.raises(ValueError, match=">= 0.0"):
        _as_float({"k": -0.5}, "k", min_value=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        _as_float({"k": 1.5}, "k", max_value=1.0)
    assert _as_float({"k": 0.5}, "k", min_value=0.0, max_value=1.0) == 0.5


# --------------------------------------------------------------------------
# _as_int
# --------------------------------------------------------------------------

def test_as_int_rejects_non_integer_string() -> None:
    with pytest.raises(ValueError, match="должен быть целым числом"):
        _as_int({"k": "abc"}, "k")


def test_as_int_rejects_none() -> None:
    with pytest.raises(ValueError, match="должен быть целым числом"):
        _as_int({"k": None}, "k")


def test_as_int_min_max_bounds() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _as_int({"k": 0}, "k", min_value=1)
    with pytest.raises(ValueError, match="<= 100"):
        _as_int({"k": 101}, "k", max_value=100)
    assert _as_int({"k": 50}, "k", min_value=1, max_value=100) == 50


# --------------------------------------------------------------------------
# _manual_validate_payload — train
# --------------------------------------------------------------------------

_MIN_TRAIN = {"train_mode": "tfidf", "C": 1.0, "max_iter": 1000, "test_size": 0.2}


def test_manual_train_minimal_fills_defaults() -> None:
    out = _manual_validate_payload("train", dict(_MIN_TRAIN))
    assert out["use_smote"] is True
    assert out["oversample_strategy"] == "augment_light"
    assert out["diagnostic_mode"] is False
    # Optional TrainingOptions-mirror fields stay absent unless provided.
    assert "field_dropout_prob" not in out


def test_manual_train_validates_optional_field_dropout_prob() -> None:
    with pytest.raises(ValueError, match="<= 1.0"):
        _manual_validate_payload(
            "train", {**_MIN_TRAIN, "field_dropout_prob": 1.5},
        )
    out = _manual_validate_payload(
        "train", {**_MIN_TRAIN, "field_dropout_prob": 0.25},
    )
    assert out["field_dropout_prob"] == 0.25


def test_manual_train_validates_label_smoothing_eps() -> None:
    with pytest.raises(ValueError, match="<= 0.5"):
        _manual_validate_payload(
            "train", {**_MIN_TRAIN, "label_smoothing_eps": 0.6},
        )


def test_manual_train_validates_fuzzy_dedup_threshold() -> None:
    with pytest.raises(ValueError, match="<= 100"):
        _manual_validate_payload(
            "train", {**_MIN_TRAIN, "fuzzy_dedup_threshold": 200},
        )


def test_manual_train_validates_field_dropout_copies() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _manual_validate_payload(
            "train", {**_MIN_TRAIN, "field_dropout_copies": 0},
        )


def test_manual_train_rejects_zero_test_size() -> None:
    with pytest.raises(ValueError, match="test_size"):
        _manual_validate_payload(
            "train", {**_MIN_TRAIN, "test_size": 0.0},
        )


def test_manual_train_rejects_zero_C() -> None:
    # Manual path's _as_float(min_value=0.0) admits 0.0; the explicit
    # `c_val <= 0.0` re-check catches it. Guards against a future
    # refactor that silently widens C to non-positive.
    with pytest.raises(ValueError, match="C"):
        _manual_validate_payload(
            "train", {**_MIN_TRAIN, "C": 0.0},
        )


# --------------------------------------------------------------------------
# _manual_validate_payload — apply / cluster
# --------------------------------------------------------------------------

def test_manual_apply_minimal_fills_defaults() -> None:
    out = _manual_validate_payload(
        "apply", {"model_file": "m.joblib", "apply_file": "i.csv", "pred_col": "y"},
    )
    assert out["use_ensemble"] is False
    assert out["diagnostic_mode"] is False


def test_manual_cluster_minimal_fills_defaults() -> None:
    out = _manual_validate_payload(
        "cluster",
        {
            "cluster_algo": "kmeans", "cluster_vec_mode": "tfidf",
            "k_clusters": 5, "n_init_cluster": 10, "cluster_min_df": 1,
            "use_umap": True,
        },
    )
    assert out["merge_similar_clusters"] is False
    assert out["merge_threshold"] == 0.85
    assert out["n_repr_examples"] == 5


def test_manual_cluster_rejects_k_below_min() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        _manual_validate_payload(
            "cluster",
            {
                "cluster_algo": "kmeans", "cluster_vec_mode": "tfidf",
                "k_clusters": 1, "n_init_cluster": 10, "cluster_min_df": 1,
                "use_umap": False,
            },
        )
