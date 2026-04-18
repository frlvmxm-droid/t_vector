import numpy as np
import pytest

from apply_prediction_service import predict_with_thresholds, validate_apply_bundle
from exceptions import SchemaError


def test_validate_apply_bundle_rejects_wrong_type():
    with pytest.raises(SchemaError, match="ARTIFACT_TYPE_MISMATCH"):
        validate_apply_bundle({"artifact_type": "cluster_model_bundle", "schema_version": 1})


def test_predict_with_thresholds_returns_fallback_when_below_threshold():
    proba = np.array([[0.2, 0.8], [0.55, 0.45]], dtype=float)
    out = predict_with_thresholds(proba, ["A", "B"], default_threshold=0.6, threshold_mode="strict")
    assert out.labels == ["B", "other_label"]
    assert out.confidences == [0.8, 0.55]
    assert out.needs_review == [0, 1]


def test_predict_with_thresholds_custom_fallback_label():
    proba = np.array([[0.51, 0.49]], dtype=float)
    out = predict_with_thresholds(
        proba,
        ["A", "B"],
        default_threshold=0.8,
        fallback_label="REVIEW",
        threshold_mode="strict",
    )
    assert out.labels == ["REVIEW"]
    assert out.confidences == [0.51]
    assert out.needs_review == [1]


def test_predict_with_thresholds_review_only_mode_when_fallback_none():
    proba = np.array([[0.51, 0.49]], dtype=float)
    out = predict_with_thresholds(
        proba,
        ["A", "B"],
        default_threshold=0.8,
        fallback_label=None,
        threshold_mode="strict",
    )
    assert out.labels == ["A"]
    assert out.confidences == [0.51]
    assert out.needs_review == [1]


def test_predict_with_thresholds_legacy_mode_when_strict_disabled():
    proba = np.array([[0.51, 0.49]], dtype=float)
    out = predict_with_thresholds(
        proba,
        ["A", "B"],
        default_threshold=0.8,
        fallback_label="REVIEW",
        strict_threshold=False,
    )
    assert out.labels == ["A"]
    assert out.confidences == [0.51]
    assert out.needs_review == [1]


def test_predict_with_thresholds_default_is_legacy_compatible():
    proba = np.array([[0.51, 0.49]], dtype=float)
    out = predict_with_thresholds(
        proba,
        ["A", "B"],
        default_threshold=0.8,
        fallback_label="REVIEW",
    )
    assert out.labels == ["A"]
    assert out.needs_review == [1]


def test_predict_with_thresholds_review_only_mode():
    proba = np.array([[0.51, 0.49]], dtype=float)
    out = predict_with_thresholds(
        proba,
        ["A", "B"],
        default_threshold=0.8,
        fallback_label="REVIEW",
        threshold_mode="review_only",
    )
    assert out.labels == ["A"]
    assert out.needs_review == [1]


def test_predict_with_thresholds_migration_compat_argmax_with_low_threshold():
    proba = np.array([[0.9, 0.1]], dtype=float)
    out = predict_with_thresholds(
        proba,
        ["A", "B"],
        default_threshold=0.2,
        strict_threshold=False,
    )
    assert out.labels == ["A"]
    assert out.needs_review == [0]


def test_predict_with_thresholds_threshold_mode_precedence():
    proba = np.array([[0.51, 0.49]], dtype=float)
    out = predict_with_thresholds(
        proba,
        ["A", "B"],
        default_threshold=0.8,
        threshold_mode="strict",
        strict_threshold=False,  # ignored
    )
    assert out.labels == ["other_label"]


def test_predict_with_thresholds_legacy_review_only_flag_compat():
    proba = np.array([[0.51, 0.49]], dtype=float)
    out = predict_with_thresholds(
        proba,
        ["A", "B"],
        default_threshold=0.8,
        fallback_label="REVIEW",
        review_only=True,
        strict_threshold=True,  # legacy conflict: review_only should win
    )
    assert out.labels == ["A"]
    assert out.needs_review == [1]


def test_predict_with_thresholds_invalid_mode_raises():
    proba = np.array([[0.6, 0.4]], dtype=float)
    with pytest.raises(ValueError):
        predict_with_thresholds(
            proba,
            ["A", "B"],
            threshold_mode="legacy",  # type: ignore[arg-type]
        )


def test_predict_with_thresholds_chunk_integration_flow():
    classes = ["A", "B"]
    chunks = [
        np.array([[0.9, 0.1], [0.55, 0.45]], dtype=float),
        np.array([[0.4, 0.6], [0.51, 0.49]], dtype=float),
    ]
    labels = []
    reviews = []
    for ch in chunks:
        out = predict_with_thresholds(
            ch,
            classes,
            default_threshold=0.6,
            threshold_mode="strict",
            fallback_label="other_label",
        )
        labels.extend(out.labels)
        reviews.extend(out.needs_review)
    assert labels == ["A", "other_label", "B", "other_label"]
    assert reviews == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Ambiguity detection tests
# ---------------------------------------------------------------------------

def test_is_ambiguous_field_present():
    proba = np.array([[0.55, 0.45], [0.9, 0.1]], dtype=float)
    out = predict_with_thresholds(proba, ["A", "B"])
    assert hasattr(out, "is_ambiguous")
    assert len(out.is_ambiguous) == 2


def test_is_ambiguous_flagged_when_margin_below_threshold():
    # margin = 0.51 - 0.49 = 0.02, ambiguity_threshold=0.10 → ambiguous
    proba = np.array([[0.51, 0.49]], dtype=float)
    out = predict_with_thresholds(proba, ["A", "B"], ambiguity_threshold=0.10)
    assert out.is_ambiguous[0] is True


def test_is_ambiguous_not_flagged_when_margin_above_threshold():
    # margin = 0.9 - 0.1 = 0.8, ambiguity_threshold=0.10 → not ambiguous
    proba = np.array([[0.9, 0.1]], dtype=float)
    out = predict_with_thresholds(proba, ["A", "B"], ambiguity_threshold=0.10)
    assert out.is_ambiguous[0] is False


def test_is_ambiguous_single_class_no_crash():
    # Only one class → no second score → should not raise
    proba = np.array([[1.0]], dtype=float)
    out = predict_with_thresholds(proba, ["A"], ambiguity_threshold=0.10)
    assert len(out.is_ambiguous) == 1
    assert out.is_ambiguous[0] is False
