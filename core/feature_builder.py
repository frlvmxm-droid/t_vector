# -*- coding: utf-8 -*-
"""core/feature_builder.py — шим для обратной совместимости. Код находится в feature_builder.py.

.. deprecated::
    Импортируйте напрямую из ``feature_builder``.
    Этот шим будет удалён в следующем мажорном релизе.
"""
import warnings as _warnings
_warnings.warn(
    "core.feature_builder устарел — импортируйте напрямую из feature_builder. "
    "Этот шим будет удалён в следующем мажорном релизе.",
    DeprecationWarning,
    stacklevel=2,
)

from feature_builder import *  # noqa: F401, F403
from feature_builder import build_feature_text, choose_row_profile_weights  # noqa: F401
