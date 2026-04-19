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
import csv
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


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


def _read_tabular_columns(
    path: str,
    *,
    text_col: str,
    label_col: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Reads (text, label) columns from XLSX/CSV via the shared tabular reader.

    If ``label_col`` is None (apply path), returns ``(texts, [])`` and only
    requires the text column. Empty cells are skipped with their counterparts.
    """
    from excel_utils import idx_of, open_tabular  # local import: openpyxl optional

    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"data file not found: {path}")

    texts: List[str] = []
    labels: List[str] = []
    with open_tabular(p) as rows:
        header_raw = next(rows)
        header = ["" if h is None else str(h).strip() for h in header_raw]
        i_text = idx_of(header, text_col)
        if i_text is None:
            raise ValueError(
                f"text column {text_col!r} not found in {path}; "
                f"available headers: {header}"
            )
        i_label: Optional[int] = None
        if label_col is not None:
            i_label = idx_of(header, label_col)
            if i_label is None:
                raise ValueError(
                    f"label column {label_col!r} not found in {path}; "
                    f"available headers: {header}"
                )
        for row in rows:
            t = row[i_text] if i_text < len(row) else None
            t_str = "" if t is None else str(t).strip()
            if not t_str:
                continue
            if i_label is not None:
                lbl = row[i_label] if i_label < len(row) else None
                lbl_str = "" if lbl is None else str(lbl).strip()
                if not lbl_str:
                    continue
                labels.append(lbl_str)
            texts.append(t_str)
    return texts, labels


def _build_default_features() -> Any:
    """A minimal TF-IDF FeatureUnion suitable for the CLI happy path.

    Word (1-2) + char (3-5) n-grams. Mirrors the conservative defaults
    `make_hybrid_vectorizer` falls back to when SBERT/DeBERTa are not
    requested. Keeps the CLI dependency surface to scikit-learn only —
    no torch / sentence-transformers required.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=2, max_df=0.95, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, max_df=0.95, sublinear_tf=True,
        )),
    ])


def _write_apply_output(
    out_path: str,
    *,
    texts: Sequence[str],
    labels: Sequence[str],
    confidences: Sequence[float],
    needs_review: Sequence[int],
    text_col: str,
) -> None:
    """Writes CSV (default) or XLSX based on extension."""
    p = pathlib.Path(out_path)
    suffix = p.suffix.lower()
    rows = list(zip(texts, labels, confidences, needs_review))
    header = [text_col, "predicted_label", "confidence", "needs_review"]

    if suffix == ".csv":
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for t, l, c, r in rows:
                w.writerow([t, l, f"{c:.4f}", r])
        return

    if suffix in (".xlsx", ".xlsm"):
        from openpyxl import Workbook  # local import: openpyxl optional
        wb = Workbook(write_only=True)
        ws = wb.create_sheet("predictions")
        ws.append(header)
        for t, l, c, r in rows:
            ws.append([t, l, float(c), int(r)])
        wb.save(p)
        return

    raise ValueError(f"unsupported output extension {suffix!r}; use .csv or .xlsx")


# ---------------------------------------------------------------------------
# Subcommand: cluster
# ---------------------------------------------------------------------------

_CLUSTER_SUPPORTED_VEC_MODE = "tfidf"
_CLUSTER_SUPPORTED_ALGOS = ("kmeans", "agglo", "lda", "hdbscan")


def cmd_cluster(args: argparse.Namespace) -> int:
    """Run the headless clustering pipeline.

    Wave 3a slice ships ``cluster_vec_mode='tfidf'`` + ``cluster_algo``
    ∈ {'kmeans', 'agglo'} end-to-end. For a supported combo the full
    4-stage pipeline runs without ``--allow-skeleton``: it reads
    ``--text-col`` from each input file, fits TF-IDF + the chosen
    clusterer, and writes ``(text, cluster_id, top_keywords)`` rows to
    ``--out``. For any other combo the command falls back to the
    prepare-inputs skeleton path and requires ``--allow-skeleton``.
    """
    from cluster_workflow_service import ClusteringWorkflow   # local import: heavy

    snap = _load_json(args.snap) if args.snap else {}
    snap.setdefault("cluster_vec_mode", _CLUSTER_SUPPORTED_VEC_MODE)
    snap.setdefault("cluster_algo", _CLUSTER_SUPPORTED_ALGOS[0])

    is_supported = (
        snap.get("cluster_vec_mode") == _CLUSTER_SUPPORTED_VEC_MODE
        and snap.get("cluster_algo") in _CLUSTER_SUPPORTED_ALGOS
    )

    if not is_supported:
        if not args.allow_skeleton:
            _stderr_logger(
                f"cluster: combo {snap.get('cluster_vec_mode')!r}+"
                f"{snap.get('cluster_algo')!r} not ported in Wave 3a slice. "
                "Pass --allow-skeleton to run the prepare-inputs stage and exit, "
                "or pick cluster_vec_mode='tfidf' + cluster_algo in "
                f"{list(_CLUSTER_SUPPORTED_ALGOS)}."
            )
            return 2
        prepared = ClusteringWorkflow.prepare_only(
            files_snapshot=list(args.files or []), snap=snap,
        )
        summary = {
            "stage": "prepare_inputs",
            "files": list(prepared.files_snapshot),
            "role": prepared.role_context.role_label,
            "note": "skeleton — combo not in Wave 3a slice; see ADR-0007",
        }
        json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    # Supported combo — wire the slice-port required fields from CLI args.
    if not args.out:
        _stderr_logger("cluster: --out is required for the supported combo")
        return 2
    snap["text_col"] = args.text_col
    snap["k_clusters"] = int(args.k_clusters)
    snap["output_path"] = args.out

    log_cb = _stderr_logger if args.verbose else None
    res = ClusteringWorkflow.run(
        files_snapshot=list(args.files or []), snap=snap, log_cb=log_cb,
    )
    summary = {
        "stage": "export_cluster_outputs",
        "k_clusters_requested": int(args.k_clusters),
        "k_clusters_found": res.n_clusters,
        "n_noise": res.n_noise,
        "out": args.out,
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    """Trains a TF-IDF + LinearSVC(Calibrated) classifier from XLSX/CSV.

    Reads ``--text-col`` and ``--label-col`` from ``--data``, builds a
    word(1-2)+char(3-5) TF-IDF FeatureUnion, calls
    ``TrainingWorkflow.fit_and_evaluate``, persists a v1 train_model
    bundle to ``--out``. ``--snap`` is an optional JSON file with
    ``TrainingOptions``-shaped overrides (calib_method, use_smote, …).
    """
    from app_train_service import TrainingWorkflow
    from ml_training import TrainingOptions

    snap = _load_json(args.snap) if args.snap else {}

    texts, labels = _read_tabular_columns(
        args.data, text_col=args.text_col, label_col=args.label_col,
    )
    if len(set(labels)) < 2:
        _stderr_logger(
            f"train: need at least 2 distinct labels, got {len(set(labels))}"
        )
        return 1
    if len(texts) < 4:
        _stderr_logger(
            f"train: need at least 4 labeled rows, got {len(texts)}"
        )
        return 1

    features = _build_default_features()
    opts_kwargs = {
        k: v for k, v in snap.items()
        if k in {
            "calib_method", "use_smote", "oversample_strategy",
            "max_dup_per_sample", "run_cv", "use_hard_negatives",
            "use_field_dropout", "field_dropout_prob", "field_dropout_copies",
            "use_label_smoothing", "label_smoothing_eps",
            "use_fuzzy_dedup", "fuzzy_dedup_threshold",
        }
    }
    options = TrainingOptions(**opts_kwargs)

    workflow = TrainingWorkflow()
    pipe, clf_type, _report, _classes, _cm, extras = workflow.fit_and_evaluate(
        X=texts,
        y=labels,
        features=features,
        C=float(snap.get("C", 1.0)),
        max_iter=int(snap.get("max_iter", 2000)),
        balanced=bool(snap.get("balanced", False)),
        test_size=float(snap.get("test_size", 0.2)),
        random_state=int(snap.get("random_state", 42)),
        options=options,
        log_cb=_stderr_logger if args.verbose else None,
    )

    bundle: Dict[str, Any] = {
        "artifact_type": "train_model_bundle",
        "schema_version": 1,
        "pipeline": pipe,
        "config": {
            "clf_type": clf_type,
            "text_col": args.text_col,
            "label_col": args.label_col,
            "snap": snap,
        },
        "per_class_thresholds": dict(extras.get("per_class_thresholds", {})),
        "eval_metrics": {
            "macro_f1": float(extras.get("macro_f1", 0.0)),
            "accuracy": float(extras.get("accuracy", 0.0)),
            "n_train": int(extras.get("n_train", 0)),
            "n_test": int(extras.get("n_test", 0)),
        },
    }
    workflow.persist_artifact(bundle, args.out)

    summary = {
        "model_path": str(args.out),
        "clf_type": clf_type,
        "n_train_rows": len(texts),
        "n_classes": len(set(labels)),
        "macro_f1": bundle["eval_metrics"]["macro_f1"],
        "accuracy": bundle["eval_metrics"]["accuracy"],
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: apply
# ---------------------------------------------------------------------------

def cmd_apply(args: argparse.Namespace) -> int:
    """Predicts labels for ``--data`` rows using the bundle at ``--model``.

    Loads via ``model_loader.load_model_artifact`` (SHA-256 +
    artifact-identity validated), runs ``pipeline.predict_proba`` and
    ``predict_with_thresholds`` (per-class thresholds from the bundle
    + a global ``--threshold`` floor), and writes CSV or XLSX based
    on ``--out`` extension.
    """
    import numpy as np

    from apply_prediction_service import predict_with_thresholds, validate_apply_bundle
    from artifact_contracts import TRAIN_MODEL_ARTIFACT_TYPE
    from model_loader import load_model_artifact

    bundle = load_model_artifact(
        args.model,
        expected_artifact_types=(TRAIN_MODEL_ARTIFACT_TYPE,),
        log_fn=_stderr_logger if args.verbose else None,
    )
    validate_apply_bundle(bundle)
    pipe = bundle.get("pipeline")
    if pipe is None or not hasattr(pipe, "predict_proba"):
        _stderr_logger("apply: bundle pipeline missing predict_proba")
        return 1

    texts, _ = _read_tabular_columns(args.data, text_col=args.text_col, label_col=None)
    if not texts:
        _stderr_logger(f"apply: no non-empty rows in column {args.text_col!r}")
        return 1

    proba = pipe.predict_proba(texts)
    classes = list(pipe.classes_)
    per_cls_thr = dict(bundle.get("per_class_thresholds") or {})
    result = predict_with_thresholds(
        np.asarray(proba),
        classes=classes,
        default_threshold=float(args.threshold),
        per_class_thresholds=per_cls_thr,
        threshold_mode="review_only",
    )

    _write_apply_output(
        args.out,
        texts=texts,
        labels=result.labels,
        confidences=result.confidences,
        needs_review=result.needs_review,
        text_col=args.text_col,
    )

    summary = {
        "out_path": str(args.out),
        "n_rows": len(texts),
        "n_review": int(sum(result.needs_review)),
        "n_classes": len(classes),
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
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
    pc = sub.add_parser(
        "cluster",
        help="TF-IDF + KMeans clustering (Wave 3a slice; other combos still skeleton).",
    )
    pc.add_argument("--files", nargs="+", required=True, help="Input .xlsx/.csv file paths.")
    pc.add_argument("--snap", default=None, help="JSON file with snap parameters.")
    pc.add_argument(
        "--out", default=None,
        help="Output CSV path (required for the supported tfidf+kmeans combo).",
    )
    pc.add_argument(
        "--text-col", default="text",
        help="Text column name to read from each input file (default: text).",
    )
    pc.add_argument(
        "--k-clusters", type=int, default=8,
        help="Number of KMeans clusters (default: 8).",
    )
    pc.add_argument("--verbose", action="store_true", help="Log each stage to stderr.")
    pc.add_argument(
        "--allow-skeleton", action="store_true",
        help=(
            "Acknowledge skeleton fallback for combos outside the Wave 3a "
            "slice (anything other than tfidf+kmeans)."
        ),
    )
    pc.set_defaults(func=cmd_cluster)

    # train
    pt = sub.add_parser("train", help="Train a TF-IDF + LinearSVC classifier.")
    pt.add_argument("--data", required=True, help="Input training data (.xlsx/.csv).")
    pt.add_argument("--out", required=True, help="Output model bundle path (.joblib).")
    pt.add_argument("--text-col", default="text", help="Text column name (default: text).")
    pt.add_argument("--label-col", default="label", help="Label column name (default: label).")
    pt.add_argument("--snap", default=None, help="Optional JSON with TrainingOptions overrides.")
    pt.add_argument("--verbose", action="store_true", help="Log training progress to stderr.")
    pt.set_defaults(func=cmd_train)

    # apply
    pa = sub.add_parser("apply", help="Batch-predict labels with a trained bundle.")
    pa.add_argument("--model", required=True, help="Path to a .joblib bundle.")
    pa.add_argument("--data", required=True, help="Input data path (.xlsx/.csv).")
    pa.add_argument("--out", required=True, help="Output predictions path (.csv/.xlsx).")
    pa.add_argument("--text-col", default="text", help="Text column name (default: text).")
    pa.add_argument(
        "--threshold", type=float, default=0.5,
        help="Global confidence floor for review flag (default: 0.5).",
    )
    pa.add_argument("--verbose", action="store_true", help="Log to stderr.")
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
