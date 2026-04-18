# -*- coding: utf-8 -*-
"""core/hw_profile.py — шим для обратной совместимости. Код находится в hw_profile.py.

.. deprecated::
    Импортируйте напрямую из ``hw_profile``.
    Этот шим будет удалён в следующем мажорном релизе.
"""
import warnings as _warnings
_warnings.warn(
    "core.hw_profile устарел — импортируйте напрямую из hw_profile. "
    "Этот шим будет удалён в следующем мажорном релизе.",
    DeprecationWarning,
    stacklevel=2,
)

from hw_profile import *  # noqa: F401, F403
from hw_profile import detect  # noqa: F401
