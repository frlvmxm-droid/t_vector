# -*- coding: utf-8 -*-
"""core/text_utils.py — шим для обратной совместимости. Код находится в text_utils.py.

.. deprecated::
    Импортируйте напрямую из ``text_utils``.
    Этот шим будет удалён в следующем мажорном релизе.
"""
import warnings as _warnings
_warnings.warn(
    "core.text_utils устарел — импортируйте напрямую из text_utils. "
    "Этот шим будет удалён в следующем мажорном релизе.",
    DeprecationWarning,
    stacklevel=2,
)

from text_utils import *  # noqa: F401, F403
from text_utils import normalize_text, parse_dialog_roles, clean_answer_text  # noqa: F401
