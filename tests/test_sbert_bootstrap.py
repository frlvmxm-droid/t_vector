# -*- coding: utf-8 -*-
"""Unit-тесты для ml_sbert_bootstrap.

Проверяют инварианты bootstrap-патчей, не требуя установленного
sentence-transformers:

  1. SBERTBuiltinsPatch восстанавливает builtins после __exit__.
  2. SBERTBuiltinsPatch не падает если имя не найдено (инжектирует
     SimpleNamespace-stub).
  3. SBERTBuiltinsPatch — no-op если имя уже есть в builtins.
  4. patch_torch_and_packaging идемпотентен (двойной вызов безопасен).
  5. ml_vectorizers._BuiltinsPatch — алиас SBERTBuiltinsPatch (backward compat).
"""
from __future__ import annotations

import builtins

import pytest

from ml_sbert_bootstrap import (
    SBERTBuiltinsPatch,
    patch_torch_and_packaging,
)


class TestSBERTBuiltinsPatch:
    def test_cleans_up_injected_names_on_exit(self):
        assert not hasattr(builtins, "_brt_test_marker_xyz")
        with SBERTBuiltinsPatch() as bp:
            # Инжектируем через обход — прямое имя, гарантированно отсутствующее
            bp._injected.append("_brt_test_marker_xyz")
            builtins._brt_test_marker_xyz = object()
            assert hasattr(builtins, "_brt_test_marker_xyz")
        assert not hasattr(builtins, "_brt_test_marker_xyz"), \
            "SBERTBuiltinsPatch должен снимать инжектированные имена в __exit__"

    def test_cleanup_runs_even_on_exception(self):
        assert not hasattr(builtins, "_brt_test_marker_ex")
        with pytest.raises(RuntimeError):
            with SBERTBuiltinsPatch() as bp:
                bp._injected.append("_brt_test_marker_ex")
                builtins._brt_test_marker_ex = 1
                raise RuntimeError("boom")
        assert not hasattr(builtins, "_brt_test_marker_ex")

    def test_inject_existing_builtin_is_noop(self):
        # `len` уже есть в builtins — inject() должен просто вернуть True.
        bp = SBERTBuiltinsPatch()
        assert bp.inject("len") is True
        assert bp._injected == []  # не запомнили — удалять нечего

    def test_inject_unknown_falls_back_to_stub(self):
        bp = SBERTBuiltinsPatch()
        name = "_brt_absolutely_unknown_name_qwerty"
        try:
            assert bp.inject(name) is True
            assert hasattr(builtins, name)
        finally:
            if hasattr(builtins, name):
                delattr(builtins, name)


class TestPatchTorchAndPackaging:
    def test_idempotent_double_call(self):
        # Должен не падать при любом кол-ве вызовов.
        patch_torch_and_packaging()
        patch_torch_and_packaging()
        # Если установлен packaging — проверим что флаг стоит.
        try:
            from packaging import version as _pv
            assert getattr(_pv, "_none_patched", False) is True
        except ImportError:
            pytest.skip("packaging не установлен")


class TestBackwardCompatAlias:
    def test_ml_vectorizers_reexports_builtins_patch(self):
        from ml_vectorizers import _BuiltinsPatch
        assert _BuiltinsPatch is SBERTBuiltinsPatch

    def test_ml_vectorizers_reexports_sbert_builtins_patch(self):
        # Проверяем, что внутри ml_vectorizers доступны обе формы имени.
        import ml_vectorizers as mv
        assert mv._BuiltinsPatch is SBERTBuiltinsPatch
        assert mv.SBERTBuiltinsPatch is SBERTBuiltinsPatch
