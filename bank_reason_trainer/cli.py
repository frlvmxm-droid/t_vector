# -*- coding: utf-8 -*-
"""Argparse-based CLI dispatcher for the three domain operations.

Design goals:
  • Zero tkinter imports on the happy path — so CI coverage runs headless.
  • Each subcommand delegates to the service layer; it does not copy
    business logic. When the service layer grows a new param, wire it
    here with a sensible default rather than shadowing state.
  • JSON snap files are the canonical parameter surface (same schema
    the UI serialises via `app._snap_params`). Missing keys fall back
    to service-layer defaults; unknown keys are preserved (forwarded
    verbatim).
  • Errors return a non-zero exit code and a one-line stderr message.
    Stack traces are only printed when the caller passes `--debug`.

This module intentionally does not auto-run: the `python -m …` entry
point is in `__main__.py`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"snap file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"snap file must be a JSON object, got {type(data).__name__}")
    return data


def _stderr_logger(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand: cluster
# ---------------------------------------------------------------------------

def cmd_cluster(args: argparse.Namespace) -> int:
    """Run the full 4-stage clustering pipeline via `ClusteringWorkflow`.

    Heavy deps (scikit-learn, SBERT) load lazily only when this command
    is dispatched, so `python -m bank_reason_trainer --help` stays fast
    and `train`/`apply` don't pay the import cost.
    """
    from cluster_workflow_service import ClusteringWorkflow   # local import: heavy

    snap = _load_json(args.snap) if args.snap else {}
    result = ClusteringWorkflow.run(
        files_snapshot=list(args.files or []),
        snap=snap,
        log_cb=_stderr_logger if args.verbose else None,
    )
    # Emit a compact summary to stdout so scripts can parse it.
    summary = {
        "n_clusters": int(result.n_clusters),
        "n_noise": int(result.n_noise),
        "export": getattr(result.export, "__dict__", {}),
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    """Thin skeleton — wires to `TrainingWorkflow.fit_and_evaluate`.

    Full implementation requires a tabular reader (reads X/y columns
    from `--data`, fetches the feature-builder config from `--snap`).
    That wiring is Wave 3c work (run_training split); this command
    currently refuses to proceed without an explicit `--allow-skeleton`
    opt-in so it cannot silently pretend to train.
    """
    if not args.allow_skeleton:
        _stderr_logger(
            "train: skeleton only. Wire up data loader (Wave 3c) or pass "
            "--allow-skeleton to acknowledge. See ADR-0002."
        )
        return 2
    # Importing here confirms service layer is wired correctly.
    from app_train_service import TrainingWorkflow   # noqa: F401  local import

    _stderr_logger(
        "train: skeleton acknowledged. Pipeline will be wired in Wave 3c; "
        "no model written."
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: apply
# ---------------------------------------------------------------------------

def cmd_apply(args: argparse.Namespace) -> int:
    """Thin skeleton — wires to `apply_prediction_service.predict_with_thresholds`."""
    if not args.allow_skeleton:
        _stderr_logger(
            "apply: skeleton only. Wire up batch predictor (Wave 5) or pass "
            "--allow-skeleton to acknowledge."
        )
        return 2
    from apply_prediction_service import predict_with_thresholds   # noqa: F401  local import

    _stderr_logger(
        "apply: skeleton acknowledged. Batch predictor to land in Wave 5; "
        "no predictions written."
    )
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bank_reason_trainer",
        description="Headless CLI for BankReasonTrainer (cluster / train / apply).",
    )
    p.add_argument("--debug", action="store_true", help="Show full stack traces.")

    sub = p.add_subparsers(dest="command", required=True)

    # cluster
    pc = sub.add_parser("cluster", help="Run the full clustering pipeline.")
    pc.add_argument("--files", nargs="+", required=True, help="Input .xlsx/.csv file paths.")
    pc.add_argument("--snap", default=None, help="JSON file with snap parameters.")
    pc.add_argument("--verbose", action="store_true", help="Log each stage to stderr.")
    pc.set_defaults(func=cmd_cluster)

    # train
    pt = sub.add_parser("train", help="Skeleton — train a classifier (Wave 3c).")
    pt.add_argument("--data", required=True, help="Input training data.")
    pt.add_argument("--out", required=True, help="Output model bundle path.")
    pt.add_argument("--snap", default=None, help="JSON file with snap parameters.")
    pt.add_argument(
        "--allow-skeleton", action="store_true",
        help="Acknowledge that the full pipeline is not yet wired here.",
    )
    pt.set_defaults(func=cmd_train)

    # apply
    pa = sub.add_parser("apply", help="Skeleton — batch prediction (Wave 5).")
    pa.add_argument("--model", required=True, help="Path to a .joblib bundle.")
    pa.add_argument("--data", required=True, help="Input data path.")
    pa.add_argument("--out", required=True, help="Output predictions path.")
    pa.add_argument(
        "--allow-skeleton", action="store_true",
        help="Acknowledge that the full pipeline is not yet wired here.",
    )
    pa.set_defaults(func=cmd_apply)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    try:
        return func(args)
    except Exception as exc:  # noqa: BLE001 — CLI top-level catch-all by design
        if getattr(args, "debug", False):
            raise
        _stderr_logger(f"error: {exc}")
        return 1
