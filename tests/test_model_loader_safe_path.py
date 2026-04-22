# -*- coding: utf-8 -*-
"""
Tests for model_loader.is_safe_path and is_safe_path_strict.

is_safe_path_strict must refuse any path containing a symlink component
(mitigates TOCTOU between resolve() and the subsequent file open).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_loader import is_safe_path, is_safe_path_strict


class TestIsSafePath:
    def test_file_inside_base_accepted(self, tmp_path):
        target = tmp_path / "model.joblib"
        target.write_bytes(b"x")
        assert is_safe_path(tmp_path, target) is True

    def test_file_outside_base_rejected(self, tmp_path):
        other = tmp_path.parent / "outside.joblib"
        assert is_safe_path(tmp_path, other) is False

    def test_parent_traversal_rejected(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        traversal = sub / ".." / ".." / "etc"
        assert is_safe_path(tmp_path, traversal) is False


class TestIsSafePathStrict:
    def test_plain_file_accepted(self, tmp_path):
        target = tmp_path / "model.joblib"
        target.write_bytes(b"x")
        assert is_safe_path_strict(tmp_path, target) is True

    @pytest.mark.skipif(
        os.name == "nt", reason="symlinks require admin on Windows"
    )
    def test_symlink_to_inside_rejected(self, tmp_path):
        real = tmp_path / "real.joblib"
        real.write_bytes(b"x")
        link = tmp_path / "link.joblib"
        link.symlink_to(real)
        assert is_safe_path_strict(tmp_path, link) is False

    @pytest.mark.skipif(
        os.name == "nt", reason="symlinks require admin on Windows"
    )
    def test_symlink_to_outside_rejected(self, tmp_path):
        outside = tmp_path.parent / "other.joblib"
        outside.write_bytes(b"x")
        link = tmp_path / "link.joblib"
        link.symlink_to(outside)
        assert is_safe_path_strict(tmp_path, link) is False

    @pytest.mark.skipif(
        os.name == "nt", reason="symlinks require admin on Windows"
    )
    def test_symlinked_parent_directory_rejected(self, tmp_path):
        """Even if target file itself isn't a symlink, a symlinked parent
        directory on the path is a TOCTOU vector."""
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        target = real_dir / "model.joblib"
        target.write_bytes(b"x")
        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(real_dir)
        target_via_link = link_dir / "model.joblib"
        assert is_safe_path_strict(tmp_path, target_via_link) is False

    def test_outside_base_rejected(self, tmp_path):
        other = tmp_path.parent / "outside.joblib"
        assert is_safe_path_strict(tmp_path, other) is False

    def test_nonexistent_path_still_checked(self, tmp_path):
        # A file that doesn't yet exist should still pass if its parent
        # chain is clean (no symlinks).
        target = tmp_path / "future.joblib"
        assert is_safe_path_strict(tmp_path, target) is True
