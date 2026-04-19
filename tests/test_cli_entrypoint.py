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


def test_parser_train_accepts_data_and_out():
    """Train no longer gates behind --allow-skeleton (Wave 8.3 — real impl)."""
    parser = build_parser()
    args = parser.parse_args(["train", "--data", "d.xlsx", "--out", "m.joblib"])
    assert args.data == "d.xlsx"
    assert args.out == "m.joblib"
    assert args.text_col == "text"
    assert args.label_col == "label"
    assert not hasattr(args, "allow_skeleton")


# ---------------------------------------------------------------------------
# cluster — skeleton until Wave 3a pipeline port
# ---------------------------------------------------------------------------

def test_cluster_supported_combo_requires_out(capsys):
    """Wave 3a slice: default snap (tfidf+kmeans) is the supported combo;
    it runs the real pipeline and therefore needs --out."""
    rc = main(["cluster", "--files", "a.xlsx"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--out" in err


def test_cluster_unsupported_combo_refuses_without_allow_skeleton(tmp_path, capsys):
    """For combos outside the slice (e.g. sbert+hdbscan), --allow-skeleton
    is still required to acknowledge the prepare-only fallback."""
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps({"cluster_vec_mode": "sbert", "cluster_algo": "hdbscan"}),
        encoding="utf-8",
    )
    rc = main(["cluster", "--files", "a.xlsx", "--snap", str(snap_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Wave 3a slice" in err
    assert "sbert" in err and "hdbscan" in err


def test_cluster_allow_skeleton_runs_prepare_only_for_unsupported_combo(
    tmp_path, capsys,
):
    """With --allow-skeleton on an unsupported combo, prepare_inputs runs
    and the JSON summary names the port gap."""
    from types import SimpleNamespace

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps({
            "cluster_vec_mode": "sbert", "cluster_algo": "hdbscan",
            "cluster_role_mode": "all", "ignore_chatbot_cluster": True,
        }),
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
    assert "skeleton" in parsed["note"].lower()


def test_cluster_missing_snap_returns_error(capsys):
    """Snap-file errors surface as exit code 1."""
    rc = main([
        "cluster", "--files", "a.xlsx",
        "--snap", "/does/not/exist.json",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "snap file not found" in err


def test_cluster_supported_combo_runs_full_pipeline(tmp_path, capsys):
    """End-to-end: tiny CSV → 4-stage pipeline → output CSV with cluster ids."""
    import csv as _csv

    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    rows = [
        "перевод денег срочно", "блокировка карты сегодня",
        "перевод на счёт", "карта заблокирована вчера",
        "перевести деньги", "разблокировать карту",
        "отправить перевод", "карта блок",
        "перевод", "блок",
    ]
    with inp.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["text"])
        for r in rows:
            w.writerow([r])

    rc = main([
        "cluster", "--files", str(inp), "--out", str(out),
        "--text-col", "text", "--k-clusters", "2",
    ])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stage"] == "export_cluster_outputs"
    assert parsed["k_clusters_requested"] == 2

    with out.open() as f:
        reader = list(_csv.reader(f))
    assert reader[0] == ["text", "cluster_id", "top_keywords"]
    assert len(reader) == 1 + len(rows)
    cluster_ids = {row[1] for row in reader[1:]}
    assert cluster_ids <= {"0", "1"}


def test_cluster_lda_combo_runs_full_pipeline(tmp_path, capsys):
    """Slice extension: tfidf keyword matrix + LDA (CountVectorizer fit,
    TF-IDF keywords) round-trips through all 4 stages."""
    import csv as _csv

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps({
            "cluster_vec_mode": "tfidf", "cluster_algo": "lda",
            "lda_max_iter": 20,
        }),
        encoding="utf-8",
    )

    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    rows = [
        "перевод денег срочно", "блокировка карты сегодня",
        "перевод на счёт", "карта заблокирована вчера",
        "перевести деньги", "разблокировать карту",
        "отправить перевод", "карта блок",
        "перевод денег", "карту заблокировали",
    ]
    with inp.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["text"])
        for r in rows:
            w.writerow([r])

    rc = main([
        "cluster", "--files", str(inp), "--out", str(out),
        "--snap", str(snap_path),
        "--text-col", "text", "--k-clusters", "2",
    ])
    assert rc == 0, capsys.readouterr().err
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stage"] == "export_cluster_outputs"

    with out.open() as f:
        reader = list(_csv.reader(f))
    assert reader[0] == ["text", "cluster_id", "top_keywords"]
    assert len(reader) == 1 + len(rows)
    cluster_ids = {row[1] for row in reader[1:]}
    assert cluster_ids <= {"0", "1"}


def test_cluster_agglo_combo_runs_full_pipeline(tmp_path, capsys):
    """Slice extension: tfidf + agglomerative (Ward linkage) round-trips."""
    import csv as _csv

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps({"cluster_vec_mode": "tfidf", "cluster_algo": "agglo"}),
        encoding="utf-8",
    )

    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    rows = [
        "перевод денег срочно", "блокировка карты сегодня",
        "перевод на счёт", "карта заблокирована вчера",
        "перевести деньги", "разблокировать карту",
        "отправить перевод", "карта блок",
    ]
    with inp.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["text"])
        for r in rows:
            w.writerow([r])

    rc = main([
        "cluster", "--files", str(inp), "--out", str(out),
        "--snap", str(snap_path),
        "--text-col", "text", "--k-clusters", "2",
    ])
    assert rc == 0, capsys.readouterr().err
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stage"] == "export_cluster_outputs"

    with out.open() as f:
        reader = list(_csv.reader(f))
    assert reader[0] == ["text", "cluster_id", "top_keywords"]
    assert len(reader) == 1 + len(rows)
    cluster_ids = {row[1] for row in reader[1:]}
    assert cluster_ids <= {"0", "1"}


def test_cluster_sbert_kmeans_combo_runs_full_pipeline(tmp_path, capsys, monkeypatch):
    """Slice extension: sbert + kmeans. SBERTVectorizer is stubbed to return
    deterministic embeddings so the test stays offline (real model download
    happens via ml_sbert_bootstrap which requires HF Hub / cache)."""
    import csv as _csv
    import numpy as _np

    # Stub SBERTVectorizer.fit_transform to return a deterministic 8-dim
    # embedding derived from text hashing. This sidesteps the sentence-
    # transformers download + torch wiring and is faithful to how the
    # pipeline uses the vectorizer (fit_transform → dense matrix, KMeans).
    class _StubSBERT:
        def __init__(self, *args, **kwargs):
            pass

        def fit_transform(self, texts):
            rng = _np.random.default_rng(42)
            # Two distinct centroids; bucket by a keyword match so clustering
            # actually recovers the two topics.
            out = []
            for t in texts:
                base = rng.standard_normal(8) * 0.1
                if "перевод" in t.lower():
                    base += _np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
                else:
                    base += _np.array([0, 0, 0, 0, 1.0, 0, 0, 0])
                out.append(base)
            return _np.asarray(out, dtype=_np.float32)

    monkeypatch.setattr("ml_vectorizers.SBERTVectorizer", _StubSBERT)

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps({
            "cluster_vec_mode": "sbert", "cluster_algo": "kmeans",
            "sbert_model": "cointegrated/rubert-tiny2",
            "sbert_batch": 8, "sbert_device": "cpu",
        }),
        encoding="utf-8",
    )

    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    rows = [
        "перевод денег срочно", "блокировка карты сегодня",
        "перевод на счёт", "карта заблокирована вчера",
        "перевести деньги", "разблокировать карту",
        "отправить перевод", "карта блок",
    ]
    with inp.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["text"])
        for r in rows:
            w.writerow([r])

    rc = main([
        "cluster", "--files", str(inp), "--out", str(out),
        "--snap", str(snap_path),
        "--text-col", "text", "--k-clusters", "2",
    ])
    assert rc == 0, capsys.readouterr().err
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stage"] == "export_cluster_outputs"
    # K=2 requested; with the stubbed embeddings it should be exactly 2.
    assert parsed["k_clusters_found"] == 2

    with out.open() as f:
        reader = list(_csv.reader(f))
    assert reader[0] == ["text", "cluster_id", "top_keywords"]
    assert len(reader) == 1 + len(rows)
    cluster_ids = {row[1] for row in reader[1:]}
    assert cluster_ids <= {"0", "1"}


def test_cluster_hdbscan_combo_runs_full_pipeline(tmp_path, capsys):
    """Slice extension: tfidf + HDBSCAN (density-based; K is discovered,
    not requested). Noise label -1 is permitted; the XLSX writer still
    serialises it as a valid cluster_id string."""
    import csv as _csv

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps({
            "cluster_vec_mode": "tfidf", "cluster_algo": "hdbscan",
            # Small min_cluster_size so the 12-row fixture actually forms
            # clusters rather than being labelled all-noise.
            "hdbscan_min_cluster_size": 3,
        }),
        encoding="utf-8",
    )

    inp = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    rows = [
        "перевод денег срочно", "перевод денег быстро", "перевод денег сейчас",
        "блокировка карты сегодня", "блокировка карты утром", "блокировка карты вчера",
        "перевод на счёт", "перевод на карту", "перевод на дебет",
        "карта заблокирована", "карту заблокировали", "карта блок",
    ]
    with inp.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["text"])
        for r in rows:
            w.writerow([r])

    rc = main([
        "cluster", "--files", str(inp), "--out", str(out),
        "--snap", str(snap_path),
        # --k-clusters is ignored by HDBSCAN but the CLI still demands it.
        "--text-col", "text", "--k-clusters", "2",
    ])
    assert rc == 0, capsys.readouterr().err
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["stage"] == "export_cluster_outputs"
    # HDBSCAN discovers K; accept any non-negative value (including 0 if
    # the whole dataset is flagged noise for a pathological fixture —
    # the test fixture is structured so this should not happen, but we
    # assert on the schema contract rather than the K value).
    assert isinstance(parsed["k_clusters_found"], int)
    assert parsed["k_clusters_found"] >= 0
    assert parsed["n_noise"] >= 0

    with out.open() as f:
        reader = list(_csv.reader(f))
    assert reader[0] == ["text", "cluster_id", "top_keywords"]
    assert len(reader) == 1 + len(rows)


# ---------------------------------------------------------------------------
# train (Wave 8.3 — real implementation, no more --allow-skeleton)
# ---------------------------------------------------------------------------

def _write_training_csv(path: pathlib.Path) -> None:
    """Writes a tiny balanced 2-class CSV (8 rows per class)."""
    import csv
    a_rows = [f"перевод не прошёл сегодня в {h}" for h in range(8)]
    b_rows = [f"карта заблокирована номер {n}" for n in range(8)]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for t in a_rows:
            w.writerow([t, "перевод"])
        for t in b_rows:
            w.writerow([t, "блокировка"])


def test_train_missing_data_returns_error(tmp_path, capsys):
    rc = main([
        "train", "--data", str(tmp_path / "absent.csv"),
        "--out", str(tmp_path / "m.joblib"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "data file not found" in err


def test_train_writes_joblib_and_summary(tmp_path, capsys):
    """Real train E2E: CSV in, joblib out, JSON summary on stdout."""
    data = tmp_path / "train.csv"
    out = tmp_path / "model.joblib"
    _write_training_csv(data)

    rc = main(["train", "--data", str(data), "--out", str(out)])
    assert rc == 0, capsys.readouterr().err
    assert out.is_file()
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_train_rows"] == 16
    assert summary["n_classes"] == 2
    assert "LinearSVC" in summary["clf_type"] or "LogReg" in summary["clf_type"]


def test_train_too_few_classes_returns_error(tmp_path, capsys):
    """Single-class data → exit 1 with informative message (not crash)."""
    import csv
    data = tmp_path / "single.csv"
    with data.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        for i in range(8):
            w.writerow([f"text {i}", "only_class"])

    rc = main(["train", "--data", str(data), "--out", str(tmp_path / "m.joblib")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "2 distinct labels" in err


# ---------------------------------------------------------------------------
# apply (Wave 8.3 — real implementation)
# ---------------------------------------------------------------------------

def test_apply_round_trip_train_then_predict(tmp_path, capsys):
    """End-to-end: train a model, then apply it to fresh CSV."""
    train_data = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    apply_data = tmp_path / "apply.csv"
    out_path = tmp_path / "out.csv"

    _write_training_csv(train_data)

    rc = main(["train", "--data", str(train_data), "--out", str(model_path)])
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()  # drain summary

    import csv
    with apply_data.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text"])
        w.writerow(["перевод не дошёл"])
        w.writerow(["заблокировали карту"])

    rc = main([
        "apply", "--model", str(model_path),
        "--data", str(apply_data), "--out", str(out_path),
    ])
    assert rc == 0, capsys.readouterr().err
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_rows"] == 2
    assert out_path.is_file()

    with out_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["text", "predicted_label", "confidence", "needs_review"]
    assert len(rows) == 3  # header + 2 predictions
    # Both predicted labels must come from the training set
    assert rows[1][1] in {"перевод", "блокировка"}
    assert rows[2][1] in {"перевод", "блокировка"}


def test_apply_missing_model_returns_error(tmp_path, capsys):
    rc = main([
        "apply", "--model", str(tmp_path / "absent.joblib"),
        "--data", str(tmp_path / "d.csv"), "--out", str(tmp_path / "o.csv"),
    ])
    assert rc == 1
    # error string varies by missing-file vs unsupported-extension; just check non-empty
    assert capsys.readouterr().err.strip() != ""


# ---------------------------------------------------------------------------
# --debug re-raises
# ---------------------------------------------------------------------------

def test_debug_flag_reraises(tmp_path):
    """--debug re-raises rather than swallowing. Uses the unsupported-combo
    fallback (prepare_only) because that's the path patched here."""
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps({"cluster_vec_mode": "sbert", "cluster_algo": "hdbscan"}),
        encoding="utf-8",
    )
    with patch(
        "cluster_workflow_service.ClusteringWorkflow.prepare_only",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            main([
                "--debug", "cluster", "--allow-skeleton",
                "--files", "a.xlsx", "--snap", str(snap_path),
            ])


# ---------------------------------------------------------------------------
# package-level smoke: `python -m bank_reason_trainer` importable
# ---------------------------------------------------------------------------

def test_package_importable():
    import bank_reason_trainer
    assert callable(bank_reason_trainer.main)
    assert callable(bank_reason_trainer.build_parser)
