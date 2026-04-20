"""Unit tests for cluster_keywords_service.top_keywords_per_cluster."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from cluster_keywords_service import top_keywords_per_cluster  # noqa: E402


def test_returns_dict_keyed_by_cluster_id() -> None:
    texts = [
        "кредит карта платёж",
        "кредит банк платёж",
        "карта кредит платёж",
        "вклад депозит процент",
        "вклад депозит процент",
        "депозит ставка вклад",
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])
    out = top_keywords_per_cluster(texts, labels, top_n=3)
    assert set(out.keys()) == {0, 1}
    assert all(isinstance(v, list) for v in out.values())


def test_keywords_reflect_cluster_topic() -> None:
    texts = [
        "кредит карта платёж банк",
        "кредит кредит банк",
        "кредит карта кредит",
        "вклад депозит процент",
        "депозит вклад ставка",
        "вклад процент процент",
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])
    out = top_keywords_per_cluster(texts, labels, top_n=5)
    # Cluster 0 should surface the credit vocabulary.
    assert any("кредит" in kw for kw in out[0])
    assert any("вклад" in kw or "депозит" in kw for kw in out[1])


def test_noise_labels_are_skipped() -> None:
    texts = ["alpha beta", "alpha gamma", "delta epsilon"]
    labels = np.array([0, 0, -1])
    out = top_keywords_per_cluster(texts, labels, top_n=3)
    assert set(out.keys()) == {0}


def test_empty_texts_returns_empty_dict() -> None:
    assert top_keywords_per_cluster([], np.array([]), top_n=5) == {}


def test_all_noise_returns_empty_dict() -> None:
    texts = ["a b", "c d"]
    labels = np.array([-1, -1])
    assert top_keywords_per_cluster(texts, labels, top_n=3) == {}


def test_top_n_limits_output_length() -> None:
    texts = [
        " ".join(f"word{i}" for i in range(20)),
        " ".join(f"word{i}" for i in range(5, 25)),
    ]
    labels = np.array([0, 0])
    out = top_keywords_per_cluster(texts, labels, top_n=3)
    assert 0 in out
    assert len(out[0]) <= 3
