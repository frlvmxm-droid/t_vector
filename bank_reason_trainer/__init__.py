# -*- coding: utf-8 -*-
"""BankReasonTrainer — headless CLI entrypoint package.

Provides a Tk-free surface for the three domain operations
(cluster / train / apply) so they can be driven from CI, batch
scripts, and coverage runs without importing `tkinter`.

The heavy ML logic lives in the existing service layer
(`cluster_workflow_service`, `app_train_service`, `apply_prediction_service`);
this package is a thin argparse dispatcher that translates CLI
arguments into service-layer calls.

Usage:
    python -m bank_reason_trainer cluster --files a.xlsx b.xlsx --snap snap.json
    python -m bank_reason_trainer train   --data train.xlsx --out model.joblib
    python -m bank_reason_trainer apply   --model model.joblib --data in.xlsx --out out.xlsx

See ADR-0004 for the offline LLM replay that pairs with this CLI in CI.
"""
from .cli import build_parser, main

__all__ = ["build_parser", "main"]
