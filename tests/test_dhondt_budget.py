# -*- coding: utf-8 -*-
"""Unit-тесты для ml_vectorizers._dhondt_allocate.

Покрывают ключевые свойства пропорционального распределения бюджета
признаков методом д'Ондта, заменившего связку
``max(floor, proportional) + global scale-down``.
"""
from __future__ import annotations

import pytest

from ml_vectorizers import _dhondt_allocate


class TestDhondtAllocate:
    def test_sum_equals_budget(self):
        weights = {"A": 3, "B": 2, "C": 2, "D": 2, "E": 1, "F": 1}
        budget = 150_000
        alloc = _dhondt_allocate(budget, weights, floor=200)
        assert sum(alloc.values()) == budget, (
            f"Сумма {sum(alloc.values())} != бюджет {budget}"
        )
        # Все активные получили ≥ floor.
        for k in weights:
            assert alloc[k] >= 200

    def test_equal_weights_are_symmetric(self):
        weights = {"A": 1, "B": 1, "C": 1, "D": 1}
        budget = 1000
        alloc = _dhondt_allocate(budget, weights, floor=10)
        values = list(alloc.values())
        assert sum(values) == budget
        # Симметрия: при равных весах максимум и минимум отличаются не более чем на 1.
        assert max(values) - min(values) <= 1

    def test_floor_respected_when_budget_tight(self):
        weights = {"A": 5, "B": 1}
        budget = 1000
        alloc = _dhondt_allocate(budget, weights, floor=100)
        assert alloc["A"] >= 100
        assert alloc["B"] >= 100
        assert sum(alloc.values()) == budget

    def test_zero_weight_gets_zero(self):
        weights = {"A": 3, "B": 0, "C": 2}
        budget = 500
        alloc = _dhondt_allocate(budget, weights, floor=50)
        assert alloc["B"] == 0
        assert alloc["A"] >= 50
        assert alloc["C"] >= 50
        assert sum(alloc.values()) == budget

    def test_higher_weight_gets_more(self):
        weights = {"A": 5, "B": 1}
        alloc = _dhondt_allocate(1200, weights, floor=100)
        assert alloc["A"] > alloc["B"]
        # Проверяем пропорциональность остатка после floor.
        # Остаток = 1200 - 200 = 1000; д’Ондт: 5/1, 1/1, 5/2, 5/3 … → A доминирует.
        assert alloc["A"] >= alloc["B"] * 3

    def test_floor_clamped_when_budget_smaller_than_n_floor(self):
        weights = {"A": 1, "B": 1, "C": 1}
        # Бюджет < 3 × floor=50 → floor клампится до budget//3.
        alloc = _dhondt_allocate(60, weights, floor=50)
        assert sum(alloc.values()) == 60
        for v in alloc.values():
            assert v <= 50

    def test_zero_budget_returns_zeros(self):
        weights = {"A": 3, "B": 2}
        alloc = _dhondt_allocate(0, weights, floor=10)
        assert alloc == {"A": 0, "B": 0}

    def test_empty_active_set(self):
        weights = {"A": 0, "B": 0}
        alloc = _dhondt_allocate(100, weights, floor=10)
        assert alloc == {"A": 0, "B": 0}
