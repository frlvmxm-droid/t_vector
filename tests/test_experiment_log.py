# -*- coding: utf-8 -*-
"""Unit tests for experiment_log.py."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from experiment_log import log_experiment, read_experiments, EXPERIMENT_LOG


# ---------------------------------------------------------------------------
# log_experiment
# ---------------------------------------------------------------------------

def test_log_creates_file(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", log_path)

    log_experiment("model.joblib", {"train_mode": "tfidf", "C": 1.0}, {"macro_f1": 0.85, "accuracy": 0.87})

    assert log_path.exists()


def test_log_writes_valid_json(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", log_path)

    log_experiment("m.joblib", {"train_mode": "tfidf"}, {"macro_f1": 0.9, "accuracy": 0.91, "n_train": 100, "n_test": 25})
    line = log_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)

    assert record["model_file"] == "m.joblib"
    assert record["macro_f1"] == 0.9
    assert record["accuracy"] == 0.91
    assert record["n_train"] == 100
    assert record["n_test"] == 25
    assert "timestamp" in record
    assert "params" in record


def test_log_appends_multiple_entries(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", log_path)

    for i in range(3):
        log_experiment(f"model_{i}.joblib", {"C": float(i)}, {"macro_f1": 0.8 + i * 0.05})

    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    records = [json.loads(l) for l in lines]
    assert records[0]["model_file"] == "model_0.joblib"
    assert records[2]["model_file"] == "model_2.joblib"


def test_log_records_tracked_params(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", log_path)

    snap = {
        "train_mode": "tfidf",
        "C": 0.5,
        "max_iter": 1000,
        "use_smote": True,
        "use_lemma": False,
        "irrelevant_key": "ignored",
    }
    log_experiment("m.joblib", snap, {"macro_f1": 0.8})
    record = json.loads(log_path.read_text(encoding="utf-8").strip())

    params = record["params"]
    assert params["train_mode"] == "tfidf"
    assert params["C"] == 0.5
    assert params["use_smote"] is True
    assert "irrelevant_key" not in params  # only tracked params recorded


def test_log_silent_on_error(tmp_path, monkeypatch):
    """log_experiment must never raise — errors are swallowed silently."""
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", pathlib.Path("/nonexistent/path/that/cannot/be/created/ever/experiments.jsonl"))
    # Should not raise
    log_experiment("m.joblib", {}, {})


# ---------------------------------------------------------------------------
# read_experiments
# ---------------------------------------------------------------------------

def test_read_returns_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", tmp_path / "nonexistent.jsonl")
    assert read_experiments() == []


def test_read_returns_last_n(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", log_path)

    for i in range(10):
        log_experiment(f"m{i}.joblib", {}, {"macro_f1": float(i) / 10})

    records = read_experiments(last_n=3)
    assert len(records) == 3
    # Last 3 should be m7, m8, m9
    assert records[-1]["model_file"] == "m9.joblib"


def test_read_skips_invalid_json_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", log_path)

    log_path.write_text(
        '{"model_file": "ok.joblib", "macro_f1": 0.9}\n'
        'THIS IS NOT JSON\n'
        '{"model_file": "ok2.joblib", "macro_f1": 0.8}\n',
        encoding="utf-8",
    )

    records = read_experiments()
    assert len(records) == 2
    assert records[0]["model_file"] == "ok.joblib"


def test_read_returns_full_records(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("experiment_log.EXPERIMENT_LOG", log_path)

    log_experiment("model.joblib", {"train_mode": "sbert", "C": 2.0}, {"macro_f1": 0.92, "n_train": 500})
    records = read_experiments()

    assert len(records) == 1
    assert records[0]["macro_f1"] == 0.92
    assert records[0]["params"]["train_mode"] == "sbert"
