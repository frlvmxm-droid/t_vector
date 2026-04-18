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
# cluster — skeleton until Wave 3a pipeline port
# ---------------------------------------------------------------------------

def test_cluster_refuses_without_allow_skeleton(capsys):
    """Stages 2-5 raise NotImplementedError, so the CLI gates behind the
    same --allow-skeleton opt-in as `train`/`apply` to avoid advertising
    a full pipeline it cannot run."""
    rc = main(["cluster", "--files", "a.xlsx"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "skeleton" in err.lower()


def test_cluster_allow_skeleton_runs_prepare_only(tmp_path, capsys):
    """With --allow-skeleton, the CLI runs the one real stage
    (prepare_inputs) and emits a JSON summary marking the port gap."""
    from types import SimpleNamespace

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps({"cluster_role_mode": "all", "ignore_chatbot_cluster": True}),
        encoding="utf-8",
    )

    fake_prepared = SimpleNamespace(
        files_snapshot=["f1.xlsx", "f2.xlsx"],
        role_context=SimpleNamespace(role_label="Весь диалог"),
    )
    with patch(
        "cluster_workflow_service.ClusteringWorkflow.prepare_only",
        return_value=fake_prepared,
    ) as mock_prep:
        rc = main([
            "cluster", "--allow-skeleton",
            "--files", "f1.xlsx", "f2.xlsx",
            "--snap", str(snap_path),
        ])

    assert rc == 0
    mock_prep.assert_called_once()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stage"] == "prepare_inputs"
    assert parsed["files"] == ["f1.xlsx", "f2.xlsx"]
    assert "Wave 3a" in parsed["note"]


def test_cluster_missing_snap_returns_error(capsys):
    """Snap-file errors surface as exit code 1 even with --allow-skeleton."""
    rc = main([
        "cluster", "--allow-skeleton",
        "--files", "a.xlsx",
        "--snap", "/does/not/exist.json",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "snap file not found" in err


def test_cluster_without_snap_uses_empty_dict(capsys):
    """Omitting --snap forwards an empty dict to ClusteringWorkflow.prepare_only."""
    from types import SimpleNamespace

    fake_prepared = SimpleNamespace(
        files_snapshot=["a.xlsx"],
        role_context=SimpleNamespace(role_label="Весь диалог"),
    )
    with patch(
        "cluster_workflow_service.ClusteringWorkflow.prepare_only",
        return_value=fake_prepared,
    ) as mock_prep:
        rc = main(["cluster", "--allow-skeleton", "--files", "a.xlsx"])
    assert rc == 0
    kwargs = mock_prep.call_args.kwargs
    args_ = mock_prep.call_args.args
    passed_snap = kwargs.get("snap") if "snap" in kwargs else (args_[1] if len(args_) > 1 else None)
    assert passed_snap == {}


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
        "cluster_workflow_service.ClusteringWorkflow.prepare_only",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            main(["--debug", "cluster", "--allow-skeleton", "--files", "a.xlsx"])


# ---------------------------------------------------------------------------
# package-level smoke: `python -m bank_reason_trainer` importable
# ---------------------------------------------------------------------------

def test_package_importable():
    import bank_reason_trainer
    assert callable(bank_reason_trainer.main)
    assert callable(bank_reason_trainer.build_parser)
