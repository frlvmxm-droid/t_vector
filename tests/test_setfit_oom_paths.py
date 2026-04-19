"""SetFit graceful-degradation paths.

Wave 6.5 Block 2. Покрывает:
  * `_pick_vram_profile` для всех уровней VRAM (A100 → CPU).
  * env-overrides ``BRT_SETFIT_MAX_TRAIN_OVERRIDE`` /
    ``BRT_SETFIT_MAX_PAIRS_OVERRIDE`` (главный «аварийный тормоз» при
    OOM на новой машине).
  * `_cuda_cleanup` no-op при отсутствии torch / при torch.cuda.is_available()
    == False.
  * `_is_setfit_config_missing_error` ловит EntryNotFoundError,
    "Entry Not Found", "config_setfit", "404" — пути fallback на ST.

Полный fit() путь не покрываем — он требует sentence-transformers + GPU
(тяжёлая интеграционная зона; покрывается nightly UI smoke-тестом, а не
unit-тестами).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_setfit import (  # noqa: E402
    VRAM_PROFILES,
    _cuda_cleanup,
    _is_setfit_config_missing_error,
    _pick_vram_profile,
)


# ---------------------------------------------------------------------------
# _pick_vram_profile — VRAM-band selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vram_gb,expected_band_idx", [
    (80.0, 0),  # H100 → серверный профиль
    (40.0, 0),  # A100
    (24.0, 1),  # RTX 4090
    (16.0, 2),  # RTX 4070Ti
    (12.0, 2),  # boundary
    (8.0, 3),   # RTX 4060Ti
    (4.0, 3),   # CPU/iGPU
    (0.0, 3),   # no GPU
])
def test_pick_vram_profile_band_selection(vram_gb: float, expected_band_idx: int,
                                          monkeypatch) -> None:
    """Каждый VRAM-уровень мапится на правильный профиль."""
    monkeypatch.delenv("BRT_SETFIT_MAX_TRAIN_OVERRIDE", raising=False)
    monkeypatch.delenv("BRT_SETFIT_MAX_PAIRS_OVERRIDE", raising=False)
    max_train, max_pairs = _pick_vram_profile(vram_gb)
    expected_train = VRAM_PROFILES[expected_band_idx][1]
    expected_pairs = VRAM_PROFILES[expected_band_idx][2]
    assert max_train == expected_train
    assert max_pairs == expected_pairs


def test_pick_vram_profile_env_override_train(monkeypatch) -> None:
    """ENV-override на max_train заменяет профильное значение."""
    monkeypatch.setenv("BRT_SETFIT_MAX_TRAIN_OVERRIDE", "1234")
    monkeypatch.delenv("BRT_SETFIT_MAX_PAIRS_OVERRIDE", raising=False)
    max_train, _max_pairs = _pick_vram_profile(80.0)
    assert max_train == 1234


def test_pick_vram_profile_env_override_pairs(monkeypatch) -> None:
    """ENV-override на max_pairs заменяет профильное значение."""
    monkeypatch.delenv("BRT_SETFIT_MAX_TRAIN_OVERRIDE", raising=False)
    monkeypatch.setenv("BRT_SETFIT_MAX_PAIRS_OVERRIDE", "5678")
    _max_train, max_pairs = _pick_vram_profile(80.0)
    assert max_pairs == 5678


def test_pick_vram_profile_both_overrides(monkeypatch) -> None:
    """Оба override-а одновременно работают."""
    monkeypatch.setenv("BRT_SETFIT_MAX_TRAIN_OVERRIDE", "100")
    monkeypatch.setenv("BRT_SETFIT_MAX_PAIRS_OVERRIDE", "200")
    max_train, max_pairs = _pick_vram_profile(0.0)
    assert max_train == 100
    assert max_pairs == 200


# ---------------------------------------------------------------------------
# _cuda_cleanup — best-effort, never raises
# ---------------------------------------------------------------------------


def test_cuda_cleanup_no_torch_is_silent() -> None:
    """ImportError torch → silent no-op, не raise."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        _cuda_cleanup()  # should not raise


def test_cuda_cleanup_no_cuda_is_silent() -> None:
    """torch без cuda → no-op, не raise."""
    fake_torch = type("fake_torch", (), {})()
    fake_torch.cuda = type("fake_cuda", (), {
        "is_available": staticmethod(lambda: False),
    })()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        _cuda_cleanup(synchronize=True)  # should not raise
        _cuda_cleanup(synchronize=False)


# ---------------------------------------------------------------------------
# _is_setfit_config_missing_error — fallback ST-trigger
# ---------------------------------------------------------------------------


def test_is_setfit_config_missing_error_entry_not_found_class_name() -> None:
    """EntryNotFoundError по имени класса → True."""
    class EntryNotFoundError(Exception):
        pass
    assert _is_setfit_config_missing_error(EntryNotFoundError("any")) is True


def test_is_setfit_config_missing_error_repository_not_found_class_name() -> None:
    class RepositoryNotFoundError(Exception):
        pass
    assert _is_setfit_config_missing_error(RepositoryNotFoundError("any")) is True


def test_is_setfit_config_missing_error_substring_entry_not_found() -> None:
    """Generic OSError с substring 'Entry Not Found' → True."""
    assert _is_setfit_config_missing_error(OSError("Entry Not Found at hub")) is True


def test_is_setfit_config_missing_error_substring_config_setfit() -> None:
    assert _is_setfit_config_missing_error(OSError("config_setfit.json missing")) is True


def test_is_setfit_config_missing_error_substring_404() -> None:
    assert _is_setfit_config_missing_error(OSError("HTTP 404 from huggingface")) is True


def test_is_setfit_config_missing_error_unrelated_returns_false() -> None:
    """Неподходящее исключение → False (нет fallback)."""
    assert _is_setfit_config_missing_error(ValueError("bad input")) is False
    assert _is_setfit_config_missing_error(RuntimeError("CUDA OOM")) is False
