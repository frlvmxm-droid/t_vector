# -*- coding: utf-8 -*-
"""
Unit tests for ml_mlm_pretrain.

Heavy-dep tests (transformers/datasets/accelerate) are guarded:
  * is_available() — always testable
  * estimate_mlm_time_minutes() — pure arithmetic
  * pretrain_mlm() — only smoke-tested for ImportError path when deps missing
"""
from __future__ import annotations

import importlib.util

import pytest

from ml_mlm_pretrain import estimate_mlm_time_minutes, is_available, pretrain_mlm


HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_DATASETS = importlib.util.find_spec("datasets") is not None


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_returns_bool(self):
        assert isinstance(is_available(), bool)

    def test_matches_spec_lookup(self):
        assert is_available() == (HAS_TRANSFORMERS and HAS_DATASETS)


# ---------------------------------------------------------------------------
# estimate_mlm_time_minutes
# ---------------------------------------------------------------------------

class TestEstimateMlmTime:
    def test_returns_tuple(self):
        low, high = estimate_mlm_time_minutes(1000, epochs=3, has_gpu=True)
        assert isinstance(low, float)
        assert isinstance(high, float)

    def test_gpu_faster_than_cpu(self):
        gpu_low, _ = estimate_mlm_time_minutes(1000, epochs=3, has_gpu=True)
        cpu_low, _ = estimate_mlm_time_minutes(1000, epochs=3, has_gpu=False)
        assert cpu_low > gpu_low

    def test_more_texts_takes_longer(self):
        low_small, _ = estimate_mlm_time_minutes(1000, epochs=3, has_gpu=True)
        low_big, _ = estimate_mlm_time_minutes(10_000, epochs=3, has_gpu=True)
        assert low_big > low_small

    def test_more_epochs_takes_longer(self):
        low_3, _ = estimate_mlm_time_minutes(1000, epochs=3, has_gpu=True)
        low_10, _ = estimate_mlm_time_minutes(1000, epochs=10, has_gpu=True)
        assert low_10 > low_3

    def test_low_less_than_high(self):
        low, high = estimate_mlm_time_minutes(500, epochs=2)
        assert low <= high


# ---------------------------------------------------------------------------
# pretrain_mlm — ImportError guard
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    HAS_TRANSFORMERS and HAS_DATASETS,
    reason="deps present — ImportError path not testable",
)
class TestPretrainMlmMissingDeps:
    def test_raises_import_error_without_deps(self):
        with pytest.raises(ImportError, match="transformers|datasets"):
            pretrain_mlm(["hello"], output_dir="/tmp/mlm_test_nonexistent")


# ---------------------------------------------------------------------------
# pretrain_mlm — guard via monkey-patched is_available
# ---------------------------------------------------------------------------

class TestPretrainMlmGuardedByIsAvailable:
    def test_respects_is_available_false(self, monkeypatch):
        monkeypatch.setattr("ml_mlm_pretrain.is_available", lambda: False)
        with pytest.raises(ImportError):
            pretrain_mlm(["a", "b"], output_dir="/tmp/mlm_guard_test")
