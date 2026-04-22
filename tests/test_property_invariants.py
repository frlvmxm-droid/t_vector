"""Property-based invariants via hypothesis.

Покрывает три ядерных алгоритма, для которых обычные example-based тесты
не дают уверенности по всему пространству входов:

  * ``ml_vectorizers._dhondt_allocate`` — целочисленное распределение
    бюджета. Инварианты: sum == budget, монотонность по весу,
    нулевые веса дают нули.
  * temperature softmax из ``ml_distillation`` — численная устойчивость
    при произвольной температуре T > 0; sum по строкам == 1 ± 1e-9, нет
    NaN/Inf даже при T → ∞.
  * ``ml_training._apply_label_smoothing`` — детерминизм при том же
    seed, точное число перемаркированных позиций, ни одна метка не
    «перевернулась» в саму себя.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_distillation import DISTILL_EPS  # noqa: E402
from ml_training import _apply_label_smoothing  # noqa: E402
from ml_vectorizers import _dhondt_allocate  # noqa: E402


# ---------------------------------------------------------------------------
# d'Hondt allocator
# ---------------------------------------------------------------------------

_KEY_ALPHABET = st.text(
    alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
    min_size=1,
    max_size=3,
)

_weights_strategy = st.dictionaries(
    keys=_KEY_ALPHABET,
    values=st.integers(min_value=0, max_value=1000),
    min_size=1,
    max_size=8,
)


@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(weights=_weights_strategy,
       budget=st.integers(min_value=0, max_value=10_000),
       floor=st.integers(min_value=0, max_value=20))
def test_dhondt_sum_equals_budget(weights, budget, floor):
    """Сумма аллокаций == budget, если есть хотя бы один положительный вес."""
    alloc = _dhondt_allocate(budget, weights, floor=floor)

    assert set(alloc.keys()) == set(weights.keys())
    has_active = any(w > 0 for w in weights.values())
    if has_active and budget > 0:
        assert sum(alloc.values()) == budget
    else:
        # Нет активных ключей или budget=0 → все нули.
        assert sum(alloc.values()) == 0


@settings(max_examples=200, deadline=None)
@given(weights=_weights_strategy,
       budget=st.integers(min_value=0, max_value=5_000),
       floor=st.integers(min_value=0, max_value=10))
def test_dhondt_zero_weight_keys_get_zero(weights, budget, floor):
    """Ключи с нулевым весом всегда получают нулевую аллокацию."""
    alloc = _dhondt_allocate(budget, weights, floor=floor)
    for key, w in weights.items():
        if w == 0:
            assert alloc[key] == 0


@settings(max_examples=150, deadline=None)
@given(base_weight=st.integers(min_value=1, max_value=100),
       bump=st.integers(min_value=1, max_value=500),
       budget=st.integers(min_value=10, max_value=2_000))
def test_dhondt_monotonic_in_weight(base_weight, bump, budget):
    """Увеличение веса ключа не уменьшает его долю (при равных прочих)."""
    weights_a = {"X": base_weight, "Y": base_weight, "Z": base_weight}
    weights_b = {"X": base_weight + bump, "Y": base_weight, "Z": base_weight}
    alloc_a = _dhondt_allocate(budget, weights_a, floor=0)
    alloc_b = _dhondt_allocate(budget, weights_b, floor=0)
    assert alloc_b["X"] >= alloc_a["X"]


# ---------------------------------------------------------------------------
# Temperature softmax (mirror of ml_distillation logic)
# ---------------------------------------------------------------------------


def _temperature_softmax(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Локальное зеркало логики из ml_distillation для prop-тестов.

    Изолированный воспроизводимый кусок: ``log(clip) / T`` →
    logsumexp-нормализация. Тот же код в проде; здесь дублируем, чтобы
    тестировать функцию без зависимостей на teacher-API.
    """
    log_p = np.log(np.clip(proba, DISTILL_EPS, 1.0)) / float(temperature)
    log_p -= log_p.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(log_p).sum(axis=1, keepdims=True))
    return np.exp(log_p - log_z)


@settings(max_examples=200, deadline=None)
@given(
    n_classes=st.integers(min_value=2, max_value=12),
    n_samples=st.integers(min_value=1, max_value=20),
    temperature=st.floats(min_value=0.01, max_value=1e6,
                          allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_softmax_rows_sum_to_one(n_classes, n_samples, temperature, seed):
    """sum(row) == 1 ± 1e-9 при любой T > 0; ни одного NaN/Inf."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(n_classes), size=n_samples)
    out = _temperature_softmax(raw, temperature)
    assert out.shape == raw.shape
    assert np.all(np.isfinite(out)), "softmax produced NaN/Inf"
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(out >= 0.0)


@settings(max_examples=50, deadline=None)
@given(
    n_classes=st.integers(min_value=2, max_value=8),
    n_samples=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_softmax_high_temperature_approaches_uniform(n_classes, n_samples, seed):
    """T → ∞ → распределение стремится к uniform 1/n_classes."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(n_classes), size=n_samples)
    out = _temperature_softmax(raw, temperature=1e5)
    expected = np.full_like(out, 1.0 / n_classes)
    # При T=1e5 отклонение от uniform — порядка 1e-4 (зависит от entropy
    # исходного распределения через clip(eps)). Берём щедрый запас.
    np.testing.assert_allclose(out, expected, atol=1e-2)


# ---------------------------------------------------------------------------
# Label smoothing (random label injection)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=10, max_value=200),
    n_classes=st.integers(min_value=2, max_value=6),
    eps=st.floats(min_value=0.01, max_value=0.5,
                  allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_label_smoothing_deterministic_with_seed(n_samples, n_classes, eps, seed):
    """Тот же seed → бит-в-бит идентичный результат."""
    classes = [f"c{i}" for i in range(n_classes)]
    rng = np.random.default_rng(seed ^ 0xA5A5)
    labels = [classes[int(rng.integers(0, n_classes))] for _ in range(n_samples)]
    assume(len(set(labels)) >= 2)  # smoothing no-op для одноклассового y

    a = _apply_label_smoothing(labels, eps=eps, random_state=seed, log_cb=None)
    b = _apply_label_smoothing(labels, eps=eps, random_state=seed, log_cb=None)
    assert a == b


@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=20, max_value=200),
    n_classes=st.integers(min_value=2, max_value=6),
    eps=st.floats(min_value=0.05, max_value=0.5,
                  allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_label_smoothing_exact_flip_count(n_samples, n_classes, eps, seed):
    """Ровно ``int(n*eps)`` позиций изменены; ни одна метка → сама в себя."""
    classes = [f"c{i}" for i in range(n_classes)]
    rng = np.random.default_rng(seed ^ 0xDEADBEEF)
    labels = [classes[int(rng.integers(0, n_classes))] for _ in range(n_samples)]
    assume(len(set(labels)) >= 2)

    expected_flips = int(n_samples * eps)
    out = _apply_label_smoothing(labels, eps=eps, random_state=seed, log_cb=None)

    diffs = sum(1 for a, b in zip(labels, out) if a != b)
    assert diffs == expected_flips, (
        f"expected {expected_flips} flips, got {diffs}"
    )
    # Ни одна метка не «перевернулась» в саму себя — это гарантирует
    # выбор `others = [c for c in cls_all if c != current]`.
    for a, b in zip(labels, out):
        if a != b:
            assert b in classes
            assert b != a


@settings(max_examples=50, deadline=None)
@given(
    labels=st.lists(st.sampled_from(["x"]), min_size=5, max_size=50),
    eps=st.floats(min_value=0.0, max_value=0.5,
                  allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_label_smoothing_single_class_is_noop(labels, eps, seed):
    """Одноклассовый y → возврат копии (нечего перемешивать)."""
    out = _apply_label_smoothing(labels, eps=eps, random_state=seed, log_cb=None)
    assert out == labels
    assert out is not labels  # должна быть копия, не тот же объект
