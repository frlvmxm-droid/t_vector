# -*- coding: utf-8 -*-
"""Tests for the headless CLI entrypoint (`python -m bank_reason_trainer`).

Verifies:
  • Argparse dispatch routes to the right subcommand.
  • `cluster` subcommand calls `ClusteringWorkflow.run` with files +
    snap + log_cb, and prints a JSON summary on stdout.
  • `train`/`apply` skeletons refuse to run without `--allow-skeleton`
    and return exit code 2.
  • Invalid snap file produces exit code 1 (not a crash).
  • `--debug` re-raises instead of swallowing.
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from bank_reason_trainer.cli import build_parser, main


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_cluster_parses_files_and_snap():
    parser = build_parser()
    args = parser.parse_args(["cluster", "--files", "a.xlsx", "b.xlsx", "--snap", "s.json"])
    assert args.command == "cluster"
    assert args.files == ["a.xlsx", "b.xlsx"]
    assert args.snap == "s.json"


def test_parser_train_requires_allow_skeleton():
    parser = build_parser()
    args = parser.parse_args(["train", "--data", "d.xlsx", "--out", "m.joblib"])
    assert args.allow_skeleton is False


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

def test_cluster_calls_workflow_and_prints_summary(tmp_path, capsys):
    from types import SimpleNamespace

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps({"cluster_algo": "kmeans", "k_clusters": 3}), encoding="utf-8")

    fake_export = SimpleNamespace(written_paths=["/tmp/out.xlsx"])
    fake_result = SimpleNamespace(n_clusters=4, n_noise=1, export=fake_export)

    with patch("cluster_workflow_service.ClusteringWorkflow.run", return_value=fake_result) as mock_run:
        rc = main(["cluster", "--files", "f1.xlsx", "f2.xlsx", "--snap", str(snap_path)])

    assert rc == 0
    mock_run.assert_called_once()
    # files_snapshot must be a list (not generator / tuple)
    kwargs = mock_run.call_args.kwargs
    args_ = mock_run.call_args.args
    # function was called with keyword args
    assert kwargs.get("files_snapshot") == ["f1.xlsx", "f2.xlsx"] or (args_ and args_[0] == ["f1.xlsx", "f2.xlsx"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["n_clusters"] == 4
    assert parsed["n_noise"] == 1


def test_cluster_missing_snap_returns_error(capsys):
    rc = main(["cluster", "--files", "a.xlsx", "--snap", "/does/not/exist.json"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "snap file not found" in err


def test_cluster_without_snap_uses_empty_dict(capsys):
    """Omitting --snap is legal: an empty dict forwards to the workflow."""
    from types import SimpleNamespace

    fake_result = SimpleNamespace(n_clusters=0, n_noise=0, export=SimpleNamespace())
    with patch("cluster_workflow_service.ClusteringWorkflow.run", return_value=fake_result) as mock_run:
        rc = main(["cluster", "--files", "a.xlsx"])
    assert rc == 0
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("snap") == {} or (mock_run.call_args.args and mock_run.call_args.args[1] == {})


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_train_refuses_without_allow_skeleton(capsys):
    rc = main(["train", "--data", "d.xlsx", "--out", "m.joblib"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "skeleton" in err.lower()


def test_train_allow_skeleton_returns_ok(capsys):
    rc = main(["train", "--data", "d.xlsx", "--out", "m.joblib", "--allow-skeleton"])
    assert rc == 0


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def test_apply_refuses_without_allow_skeleton(capsys):
    rc = main(["apply", "--model", "m.joblib", "--data", "d.xlsx", "--out", "o.xlsx"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "skeleton" in err.lower()


# ---------------------------------------------------------------------------
# --debug re-raises
# ---------------------------------------------------------------------------

def test_debug_flag_reraises():
    with patch(
        "cluster_workflow_service.ClusteringWorkflow.run",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            main(["--debug", "cluster", "--files", "a.xlsx"])


# ---------------------------------------------------------------------------
# package-level smoke: `python -m bank_reason_trainer` importable
# ---------------------------------------------------------------------------

def test_package_importable():
    import bank_reason_trainer
    assert callable(bank_reason_trainer.main)
    assert callable(bank_reason_trainer.build_parser)
