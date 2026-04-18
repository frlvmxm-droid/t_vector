# -*- coding: utf-8 -*-
"""Выбор числа кластеров по inertia-кривой."""
from __future__ import annotations

from typing import List

from app_logger import get_logger

_log = get_logger(__name__)


class ClusterElbowSelector:
    """Сервис выбора K по inertia-кривой (метод локтя)."""

    @staticmethod
    def pick_elbow_k(inertias: List[float], ks: List[int]) -> int:
        if len(ks) < 3:
            return ks[0]
        _ctx = (
            f"len(ks)={len(ks)}, "
            f"inertia_min={min(inertias) if inertias else 'n/a'}, "
            f"inertia_max={max(inertias) if inertias else 'n/a'}"
        )
        try:
            from kneed import KneeLocator

            kn = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
            if kn.knee is not None:
                return int(kn.knee)
        except ImportError as _e:
            _log.debug(
                "ClusterElbowSelector: kneed unavailable (%s), fallback heuristic used | %s",
                _e,
                _ctx,
            )
        except Exception as _e:
            _log.debug(
                "ClusterElbowSelector: KneeLocator failed (%s), fallback heuristic used | %s",
                _e,
                _ctx,
            )

        sec = []
        for i in range(1, len(inertias) - 1):
            d2 = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
            sec.append((d2, ks[i]))
        sec.sort(reverse=True)
        return int(sec[0][1]) if sec else int(ks[0])
