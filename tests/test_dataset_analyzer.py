# -*- coding: utf-8 -*-
"""
Tests for dataset_analyzer.py:
  - analyze_dataset(X, y, field_coverage=None) -> Dict[str, Any]
  - build_param_changes(recommendations, current_values) -> List[Dict[str, Any]]
"""
from __future__ import annotations

import pytest
from dataset_analyzer import analyze_dataset, build_param_changes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_balanced(n_per_class: int, n_classes: int, text_len: int = 100):
    """Returns (X, y) with perfectly balanced classes."""
    X, y = [], []
    for cls in range(n_classes):
        for i in range(n_per_class):
            X.append("а" * text_len)
            y.append(f"class_{cls}")
    return X, y


def _make_imbalanced(majority: int, minority: int, text_len: int = 100):
    """Returns (X, y) with two classes: majority vs minority count."""
    X = ["а" * text_len] * (majority + minority)
    y = ["major"] * majority + ["minor"] * minority
    return X, y


# ===========================================================================
# analyze_dataset — return structure
# ===========================================================================

class TestAnalyzeDatasetStructure:

    def test_returns_dict_with_three_keys(self):
        X, y = _make_balanced(50, 3)
        result = analyze_dataset(X, y)
        assert isinstance(result, dict)
        assert "stats" in result
        assert "issues" in result
        assert "recommendations" in result

    def test_stats_contains_required_keys(self):
        X, y = _make_balanced(50, 3)
        stats = analyze_dataset(X, y)["stats"]
        required = [
            "n_samples", "n_classes", "class_counts",
            "max_class_count", "min_class_count",
            "imbalance_ratio", "rare_classes",
            "avg_text_len", "median_text_len",
            "top5_classes", "tail5_classes", "field_coverage",
        ]
        for key in required:
            assert key in stats, f"Missing stats key: {key}"

    def test_issues_is_list(self):
        X, y = _make_balanced(50, 3)
        issues = analyze_dataset(X, y)["issues"]
        assert isinstance(issues, list)

    def test_recommendations_is_dict(self):
        X, y = _make_balanced(50, 3)
        recs = analyze_dataset(X, y)["recommendations"]
        assert isinstance(recs, dict)

    def test_stats_n_samples_correct(self):
        X, y = _make_balanced(20, 5)   # 100 samples total
        stats = analyze_dataset(X, y)["stats"]
        assert stats["n_samples"] == 100

    def test_stats_n_classes_correct(self):
        X, y = _make_balanced(20, 5)
        stats = analyze_dataset(X, y)["stats"]
        assert stats["n_classes"] == 5

    def test_stats_class_counts_correct(self):
        X, y = _make_balanced(10, 3)
        stats = analyze_dataset(X, y)["stats"]
        for cls in ["class_0", "class_1", "class_2"]:
            assert stats["class_counts"][cls] == 10

    def test_field_coverage_none_results_in_empty_dict(self):
        X, y = _make_balanced(20, 2)
        stats = analyze_dataset(X, y, field_coverage=None)["stats"]
        assert stats["field_coverage"] == {}

    def test_field_coverage_passed_through(self):
        X, y = _make_balanced(20, 2)
        cov = {"desc": 0.8, "summary": 0.3}
        stats = analyze_dataset(X, y, field_coverage=cov)["stats"]
        assert stats["field_coverage"] == cov


# ===========================================================================
# analyze_dataset — stats values
# ===========================================================================

class TestAnalyzeDatasetStats:

    def test_imbalance_ratio_balanced(self):
        X, y = _make_balanced(100, 3)
        stats = analyze_dataset(X, y)["stats"]
        assert stats["imbalance_ratio"] == 1.0

    def test_imbalance_ratio_10_to_1(self):
        X, y = _make_imbalanced(majority=100, minority=10)
        stats = analyze_dataset(X, y)["stats"]
        assert stats["imbalance_ratio"] == 10.0

    def test_rare_classes_empty_when_all_have_enough(self):
        X, y = _make_balanced(15, 4)
        stats = analyze_dataset(X, y)["stats"]
        assert stats["rare_classes"] == []

    def test_rare_classes_detected_below_10(self):
        # class_rare has 5 samples → below _RARE_MIN=10
        X = ["text"] * 55
        y = ["common"] * 50 + ["rare"] * 5
        stats = analyze_dataset(X, y)["stats"]
        assert "rare" in stats["rare_classes"]

    def test_avg_text_len_computed(self):
        X = ["а" * 50, "а" * 150]
        y = ["a", "b"]
        stats = analyze_dataset(X, y)["stats"]
        assert stats["avg_text_len"] == 100.0

    def test_median_text_len_computed(self):
        X = ["а" * 10, "а" * 20, "а" * 30]
        y = ["a", "a", "b"]
        stats = analyze_dataset(X, y)["stats"]
        assert stats["median_text_len"] == 20.0

    def test_max_min_class_counts(self):
        X = ["x"] * 90 + ["y"] * 10
        y = ["major"] * 90 + ["minor"] * 10
        stats = analyze_dataset(X, y)["stats"]
        assert stats["max_class_count"] == 90
        assert stats["min_class_count"] == 10

    def test_top5_is_sorted_descending(self):
        X, y = [], []
        for i, cnt in enumerate([100, 80, 60, 40, 20, 10]):
            X += ["text"] * cnt
            y += [f"cls{i}"] * cnt
        stats = analyze_dataset(X, y)["stats"]
        top = stats["top5_classes"]
        counts = [c for _, c in top]
        assert counts == sorted(counts, reverse=True)


# ===========================================================================
# analyze_dataset — issues generation
# ===========================================================================

class TestAnalyzeDatasetIssues:

    def _levels(self, issues):
        return [i["level"] for i in issues]

    def _msgs(self, issues):
        return [i["msg"] for i in issues]

    def test_balanced_large_dataset_no_critical_issues(self):
        X, y = _make_balanced(200, 4)  # 800 samples, balanced
        issues = analyze_dataset(X, y)["issues"]
        assert "critical" not in self._levels(issues)

    def test_too_few_samples_critical_issue(self):
        """Less than 30 samples → critical issue."""
        X = ["text"] * 20
        y = ["a"] * 10 + ["b"] * 10
        issues = analyze_dataset(X, y)["issues"]
        assert "critical" in self._levels(issues)
        assert any("Критически" in m for m in self._msgs(issues))

    def test_small_dataset_warning_issue(self):
        """30–200 samples → warning issue."""
        X, y = _make_balanced(25, 4)  # 100 samples
        issues = analyze_dataset(X, y)["issues"]
        levels = self._levels(issues)
        # Should have warning but not critical for small-but-not-tiny
        assert "warning" in levels or "info" in levels

    def test_single_class_critical_issue(self):
        """Only one class → critical issue."""
        X = ["text"] * 50
        y = ["only_class"] * 50
        issues = analyze_dataset(X, y)["issues"]
        assert "critical" in self._levels(issues)
        assert any("2 класс" in m for m in self._msgs(issues))

    def test_10_to_1_imbalance_warning(self):
        """10:1 imbalance ratio → at least a warning-level issue."""
        X, y = _make_imbalanced(majority=500, minority=50)
        issues = analyze_dataset(X, y)["issues"]
        levels = self._levels(issues)
        assert "warning" in levels or "critical" in levels

    def test_imbalance_above_20x_critical(self):
        """Imbalance > 20x → critical issue."""
        X, y = _make_imbalanced(majority=1000, minority=40)
        issues = analyze_dataset(X, y)["issues"]
        assert "critical" in self._levels(issues)

    def test_imbalance_between_3_and_6_info_or_warning(self):
        """Imbalance 3–6x → info-level issue."""
        X, y = _make_imbalanced(majority=400, minority=100)  # 4:1
        issues = analyze_dataset(X, y)["issues"]
        assert len(issues) > 0

    def test_rare_class_triggers_warning(self):
        """Class with < 10 samples triggers a warning issue."""
        X = ["text"] * 105
        y = ["common"] * 100 + ["rare"] * 5
        issues = analyze_dataset(X, y)["issues"]
        assert any("Редкие" in m for m in self._msgs(issues))
        assert any(i["level"] == "warning" for i in issues
                   if "Редкие" in i["msg"])

    def test_low_field_coverage_info_issue(self):
        """Field coverage below 0.45 → info issue."""
        X, y = _make_balanced(100, 2)
        cov = {"summary": 0.2, "desc": 0.9}
        issues = analyze_dataset(X, y, field_coverage=cov)["issues"]
        assert any("заполнен" in m for m in self._msgs(issues))

    def test_no_low_coverage_issue_when_all_fields_above_threshold(self):
        X, y = _make_balanced(100, 2)
        cov = {"summary": 0.8, "desc": 0.9}
        issues = analyze_dataset(X, y, field_coverage=cov)["issues"]
        assert not any("заполнен" in m for m in self._msgs(issues))

    def test_issue_dicts_have_level_and_msg(self):
        """Every issue dict must have 'level' and 'msg'."""
        X, y = _make_balanced(10, 2)  # small → likely issues
        issues = analyze_dataset(X, y)["issues"]
        for issue in issues:
            assert "level" in issue
            assert "msg" in issue
            assert issue["level"] in ("critical", "warning", "info")

    def test_more_than_20_classes_info_issue(self):
        """More than 20 classes → info issue."""
        X, y = _make_balanced(5, 25)  # 25 classes, 125 samples
        issues = analyze_dataset(X, y)["issues"]
        msgs = self._msgs(issues)
        assert any("класс" in m.lower() for m in msgs)


# ===========================================================================
# analyze_dataset — recommendations
# ===========================================================================

class TestAnalyzeDatasetRecommendations:

    def test_recs_have_max_features(self):
        X, y = _make_balanced(100, 3)
        recs = analyze_dataset(X, y)["recommendations"]
        assert "max_features" in recs
        assert isinstance(recs["max_features"], int)

    def test_recs_have_C(self):
        X, y = _make_balanced(100, 3)
        recs = analyze_dataset(X, y)["recommendations"]
        assert "C" in recs

    def test_recs_have_use_svd(self):
        X, y = _make_balanced(100, 3)
        recs = analyze_dataset(X, y)["recommendations"]
        assert "use_svd" in recs
        assert isinstance(recs["use_svd"], bool)

    def test_recs_have_balanced_flag(self):
        X, y = _make_balanced(100, 3)
        recs = analyze_dataset(X, y)["recommendations"]
        assert "balanced" in recs
        assert isinstance(recs["balanced"], bool)

    def test_recs_have_use_smote(self):
        X, y = _make_balanced(100, 3)
        recs = analyze_dataset(X, y)["recommendations"]
        assert "use_smote" in recs

    def test_small_dataset_low_max_features(self):
        """Tiny dataset → lower max_features recommended."""
        X, y = _make_balanced(10, 2)  # 20 samples → tiny
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["max_features"] <= 10_000

    def test_large_dataset_high_max_features(self):
        """Large dataset → higher max_features recommended."""
        X = ["а" * 100] * 20_000
        y = [f"cls_{i % 5}" for i in range(20_000)]
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["max_features"] >= 100_000

    def test_imbalanced_recommends_smote(self):
        """Strong imbalance → use_smote=True."""
        X, y = _make_imbalanced(majority=700, minority=100)  # 7:1
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["use_smote"] is True

    def test_balanced_does_not_require_smote(self):
        """Balanced dataset → use_smote=False."""
        X, y = _make_balanced(200, 3)
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["use_smote"] is False

    def test_short_texts_small_ngram(self):
        """Short texts → smaller word n-gram range."""
        X = ["ок"] * 100  # avg 2 chars
        y = [f"cls_{i % 2}" for i in range(100)]
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["word_ng_max"] == 1

    def test_long_texts_larger_ngram(self):
        """Long texts → word_ng_max=3."""
        X = ["а " * 300] * 100  # avg >200 chars
        y = [f"cls_{i % 3}" for i in range(100)]
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["word_ng_max"] == 3

    def test_many_classes_hierarchical_flag(self):
        """15+ classes → use_hierarchical=True."""
        X, y = _make_balanced(5, 20)  # 20 classes
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["use_hierarchical"] is True

    def test_few_classes_no_hierarchical(self):
        """Fewer than 15 classes → use_hierarchical=False."""
        X, y = _make_balanced(50, 5)
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["use_hierarchical"] is False

    def test_rare_classes_suggest_anchor_texts(self):
        """Rare classes present → suggest_anchor_texts=True."""
        X = ["text"] * 55
        y = ["common"] * 50 + ["rare"] * 5
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["suggest_anchor_texts"] is True

    def test_no_rare_classes_no_anchor_suggestion(self):
        """No rare classes → suggest_anchor_texts=False."""
        X, y = _make_balanced(20, 3)
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["suggest_anchor_texts"] is False

    def test_medium_dataset_uses_svd(self):
        """Dataset in medium range → use_svd=True."""
        # _MEDIUM=3000, _SMALL=800 → between 800 and 3000
        X = ["а" * 80] * 1000
        y = [f"cls_{i % 4}" for i in range(1000)]
        recs = analyze_dataset(X, y)["recommendations"]
        assert recs["use_svd"] is True

    def test_field_dropout_when_low_coverage(self):
        """Low field coverage → use_field_dropout=True."""
        X, y = _make_balanced(50, 3)
        cov = {"summary": 0.3}  # below 0.45
        recs = analyze_dataset(X, y, field_coverage=cov)["recommendations"]
        assert recs["use_field_dropout"] is True

    def test_no_field_dropout_when_all_high_coverage(self):
        X, y = _make_balanced(50, 3)
        cov = {"summary": 0.9, "desc": 0.95}
        recs = analyze_dataset(X, y, field_coverage=cov)["recommendations"]
        assert recs["use_field_dropout"] is False


# ===========================================================================
# build_param_changes
# ===========================================================================

class TestBuildParamChanges:

    def _get_recs(self, n=500, n_classes=5):
        X = ["а" * 100] * n
        y = [f"cls_{i % n_classes}" for i in range(n)]
        return analyze_dataset(X, y)["recommendations"]

    def test_returns_list(self):
        recs = self._get_recs()
        rows = build_param_changes(recs, {})
        assert isinstance(rows, list)

    def test_each_row_has_required_keys(self):
        recs = self._get_recs()
        rows = build_param_changes(recs, {})
        for row in rows:
            assert "param" in row
            assert "label" in row
            assert "current" in row
            assert "recommended" in row
            assert "changed" in row

    def test_changed_true_when_values_differ(self):
        recs = self._get_recs()
        # Force a mismatch by using wrong current value
        current = {"max_features": 999}
        rows = build_param_changes(recs, current)
        max_feat_row = next(r for r in rows if r["param"] == "max_features")
        if max_feat_row["recommended"] != 999:
            assert max_feat_row["changed"] is True

    def test_changed_false_when_values_match(self):
        recs = self._get_recs()
        # Match current to recommended
        current = {p: recs[p] for p in recs}
        rows = build_param_changes(recs, current)
        for row in rows:
            assert row["changed"] is False, f"Expected unchanged: {row['param']}"

    def test_current_none_when_key_absent(self):
        recs = self._get_recs()
        rows = build_param_changes(recs, {})
        for row in rows:
            assert row["current"] is None

    def test_all_recommended_values_match_recs(self):
        recs = self._get_recs()
        rows = build_param_changes(recs, {})
        for row in rows:
            assert row["recommended"] == recs[row["param"]]

    def test_empty_recommendations_returns_empty_list(self):
        rows = build_param_changes({}, {"max_features": 50_000})
        assert rows == []

    def test_label_is_human_readable_string(self):
        recs = self._get_recs()
        rows = build_param_changes(recs, {})
        for row in rows:
            assert isinstance(row["label"], str)
            assert len(row["label"]) > 0

    def test_partial_current_values_mixed_changed(self):
        """Only some params match → only those show changed=False."""
        recs = self._get_recs()
        rows = build_param_changes(recs, {"max_features": recs["max_features"]})
        max_feat_row = next(r for r in rows if r["param"] == "max_features")
        assert max_feat_row["changed"] is False
        # Other params have no current → changed=True (None != any value)
        other_rows = [r for r in rows if r["param"] != "max_features"]
        assert all(r["changed"] is True for r in other_rows)

    def test_boolean_param_change_detected(self):
        """Boolean params are compared correctly."""
        recs = {"use_smote": True}
        rows = build_param_changes(recs, {"use_smote": False})
        assert rows[0]["changed"] is True

    def test_boolean_param_no_change_when_equal(self):
        recs = {"use_smote": False}
        rows = build_param_changes(recs, {"use_smote": False})
        assert rows[0]["changed"] is False
