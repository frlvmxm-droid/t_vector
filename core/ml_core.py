# -*- coding: utf-8 -*-
"""core/ml_core.py — шим для обратной совместимости. Код находится в ml_core.py.

.. deprecated::
    Импортируйте напрямую из ``ml_core`` (или из специализированных модулей
    ``ml_vectorizers``, ``ml_training``, ``ml_diagnostics``).
    Этот шим будет удалён в следующем мажорном релизе.
"""
import warnings as _warnings
_warnings.warn(
    "core.ml_core устарел — импортируйте напрямую из ml_core "
    "(или из ml_vectorizers / ml_training / ml_diagnostics). "
    "Этот шим будет удалён в следующем мажорном релизе.",
    DeprecationWarning,
    stacklevel=2,
)

from ml_core import (  # noqa: F401
    SBERTVectorizer, DeBERTaVectorizer, make_neural_vectorizer,
    make_hybrid_vectorizer, train_model, find_best_c,
    dataset_health_checks, find_sbert_in_pipeline,
    SBERT_LOCAL_DIR, DEBERTA_MODEL_IDS,
)
