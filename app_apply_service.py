# -*- coding: utf-8 -*-
"""Сервисные утилиты вкладки классификации (без UI)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class EnsemblePredictor:
    """Помощник для выравнивания и смешивания вероятностей ансамбля."""

    @staticmethod
    def align_probabilities(
        proba2: np.ndarray,
        classes1: List[str],
        classes2: List[str],
    ) -> np.ndarray:
        class_to_idx_2: Dict[str, int] = {c: i for i, c in enumerate(classes2)}
        aligned = np.zeros((proba2.shape[0], len(classes1)), dtype=proba2.dtype)
        for j1, c in enumerate(classes1):
            j2 = class_to_idx_2.get(c)
            if j2 is not None:
                aligned[:, j1] = proba2[:, j2]
        return aligned

    @staticmethod
    def blend(proba1: np.ndarray, proba2_aligned: np.ndarray, w1: float) -> np.ndarray:
        w1f = float(w1)
        w2f = 1.0 - w1f
        mixed = w1f * proba1 + w2f * proba2_aligned
        row_sums = mixed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return mixed / row_sums

