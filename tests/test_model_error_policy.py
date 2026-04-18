import logging

from exceptions import ModelLoadError, SchemaError
from model_error_policy import classify_error, log_structured_event


def test_classify_error_schema_hard_fail():
    d = classify_error(SchemaError("bad schema"))
    assert d.category == "schema_error"
    assert d.recoverable is False
    assert d.incident_marker == "MODEL_SCHEMA_HARD_FAIL"


def test_classify_error_model_load_recoverable():
    d = classify_error(ModelLoadError("broken artifact"))
    assert d.category == "model_load_error"
    assert d.recoverable is True
    assert d.incident_marker == "MODEL_LOAD_RECOVERABLE"


def test_classify_error_unexpected_incident():
    d = classify_error(RuntimeError("boom"))
    assert d.category == "unexpected_error"
    assert d.recoverable is False
    assert d.incident_marker == "UNEXPECTED_INCIDENT"


def test_log_structured_event_emits_key_values(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_structured")
    log_structured_event(
        logger,
        event="cluster.run",
        stage="load_model",
        file="x.joblib",
        rows=100,
        duration_sec=0.45,
        error_class=None,
        correlation_id="cid-123",
    )
    assert "event=cluster.run" in caplog.text
    assert "correlation_id=cid-123" in caplog.text
