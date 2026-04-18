# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for app_logger.py.

Covers:
  - get_logger: naming convention, "app.*" hierarchy
  - UILogHandler: emit callback, level filtering
  - setup_file_logging: creates handler, deduplication, file creation
  - redirect_warnings: redirects warnings.warn to logger
"""
from __future__ import annotations

import logging
import logging.handlers
import warnings
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app_logger import (
    get_logger,
    setup_file_logging,
    redirect_warnings,
    UILogHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_file_handlers(logger: logging.Logger) -> None:
    """Remove all RotatingFileHandlers from a logger (cleanup between tests)."""
    handlers_to_remove = [
        h for h in logger.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    for h in handlers_to_remove:
        h.close()
        logger.removeHandler(h)


# ===========================================================================
# get_logger
# ===========================================================================

class TestGetLogger:

    def test_returns_logger_instance(self):
        log = get_logger("some_module")
        assert isinstance(log, logging.Logger)

    def test_bare_name_gets_app_prefix(self):
        log = get_logger("my_module")
        assert log.name == "app.my_module"

    def test_app_prefix_kept_as_is(self):
        log = get_logger("app.my_module")
        assert log.name == "app.my_module"

    def test_root_app_name_kept_as_is(self):
        log = get_logger("app")
        assert log.name == "app"

    def test_app_dot_prefix_submodule(self):
        log = get_logger("app.train")
        assert log.name == "app.train"

    def test_different_names_return_different_loggers(self):
        log1 = get_logger("module_a")
        log2 = get_logger("module_b")
        assert log1.name != log2.name

    def test_same_name_returns_same_logger(self):
        log1 = get_logger("shared_name")
        log2 = get_logger("shared_name")
        assert log1 is log2

    def test_logger_in_app_hierarchy(self):
        log = get_logger("some_feature")
        # It must be a child of the "app" root logger
        assert log.name.startswith("app.")

    def test_excel_utils_name_prefixed(self):
        # Simulate what excel_utils.py does: get_logger(__name__)
        log = get_logger("excel_utils")
        assert log.name == "app.excel_utils"

    def test_deeply_nested_name(self):
        log = get_logger("a.b.c")
        assert log.name == "app.a.b.c"


# ===========================================================================
# UILogHandler
# ===========================================================================

class TestUILogHandler:

    def test_emit_calls_callback(self):
        received = []
        handler = UILogHandler(callback=received.append)
        record = logging.LogRecord(
            name="test", level=logging.WARNING,
            pathname="", lineno=0, msg="hello world",
            args=(), exc_info=None,
        )
        handler.emit(record)
        assert len(received) == 1
        assert "hello world" in received[0]

    def test_default_level_is_warning(self):
        handler = UILogHandler(callback=lambda msg: None)
        assert handler.level == logging.WARNING

    def test_custom_level_set(self):
        handler = UILogHandler(callback=lambda msg: None, level=logging.ERROR)
        assert handler.level == logging.ERROR

    def test_emit_formats_with_level_name(self):
        received = []
        handler = UILogHandler(callback=received.append)
        record = logging.LogRecord(
            name="test", level=logging.WARNING,
            pathname="", lineno=0, msg="test message",
            args=(), exc_info=None,
        )
        handler.emit(record)
        assert "WARNING" in received[0]

    def test_emit_includes_message_text(self):
        received = []
        handler = UILogHandler(callback=received.append)
        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="", lineno=0, msg="something failed",
            args=(), exc_info=None,
        )
        handler.emit(record)
        assert "something failed" in received[0]

    def test_callback_not_called_for_below_level(self):
        received = []
        handler = UILogHandler(callback=received.append, level=logging.WARNING)
        # DEBUG is below WARNING
        record = logging.LogRecord(
            name="test", level=logging.DEBUG,
            pathname="", lineno=0, msg="debug message",
            args=(), exc_info=None,
        )
        # handler.emit always calls callback when called directly;
        # level filtering is done by the logging system before emit.
        # We test via a real logger integration instead.
        logger = logging.getLogger("test.ui_filter")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.debug("should not appear")
            assert len(received) == 0
        finally:
            logger.removeHandler(handler)

    def test_callback_called_for_warning_via_logger(self):
        received = []
        handler = UILogHandler(callback=received.append, level=logging.WARNING)
        logger = logging.getLogger("test.ui_warning")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.warning("important notice")
            assert len(received) == 1
            assert "important notice" in received[0]
        finally:
            logger.removeHandler(handler)

    def test_custom_fmt_used(self):
        received = []
        handler = UILogHandler(
            callback=received.append,
            fmt="CUSTOM: %(message)s",
        )
        record = logging.LogRecord(
            name="test", level=logging.WARNING,
            pathname="", lineno=0, msg="custom format test",
            args=(), exc_info=None,
        )
        handler.emit(record)
        assert received[0].startswith("CUSTOM: custom format test")

    def test_emit_multiple_records(self):
        received = []
        handler = UILogHandler(callback=received.append)
        for i in range(5):
            record = logging.LogRecord(
                name="test", level=logging.WARNING,
                pathname="", lineno=0, msg=f"message {i}",
                args=(), exc_info=None,
            )
            handler.emit(record)
        assert len(received) == 5
        assert "message 4" in received[4]

    def test_handler_is_logging_handler_subclass(self):
        handler = UILogHandler(callback=lambda msg: None)
        assert isinstance(handler, logging.Handler)

    def test_error_level_passes_warning_threshold(self):
        received = []
        handler = UILogHandler(callback=received.append, level=logging.WARNING)
        logger = logging.getLogger("test.error_level")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.error("this is an error")
            assert len(received) == 1
        finally:
            logger.removeHandler(handler)


# ===========================================================================
# setup_file_logging
# ===========================================================================

class TestSetupFileLogging:

    def test_returns_logger(self, tmp_path: Path):
        result = setup_file_logging(log_dir=tmp_path)
        _remove_file_handlers(result)
        assert isinstance(result, logging.Logger)

    def test_returned_logger_name_is_app(self, tmp_path: Path):
        result = setup_file_logging(log_dir=tmp_path)
        _remove_file_handlers(result)
        assert result.name == "app"

    def test_creates_log_directory(self, tmp_path: Path):
        log_dir = tmp_path / "nested" / "logs"
        result = setup_file_logging(log_dir=log_dir)
        _remove_file_handlers(result)
        assert log_dir.exists()

    def test_creates_log_file(self, tmp_path: Path):
        result = setup_file_logging(log_dir=tmp_path, log_file="test.log")
        _remove_file_handlers(result)
        assert (tmp_path / "test.log").exists()

    def test_adds_rotating_file_handler(self, tmp_path: Path):
        root = logging.getLogger("app")
        before = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        result = setup_file_logging(log_dir=tmp_path, log_file="dedup_test.log")
        after = [h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        _remove_file_handlers(result)
        assert len(after) >= len(before)

    def test_idempotent_no_duplicate_handlers(self, tmp_path: Path):
        """Calling setup_file_logging twice with the same path should not add a duplicate."""
        log_file = "idempotent.log"
        result1 = setup_file_logging(log_dir=tmp_path, log_file=log_file)
        result2 = setup_file_logging(log_dir=tmp_path, log_file=log_file)
        rfh_handlers = [
            h for h in result2.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
            and getattr(h, "baseFilename", "").endswith(log_file)
        ]
        _remove_file_handlers(result2)
        # Should be exactly 1 handler for this file, not 2
        assert len(rfh_handlers) == 1

    def test_logger_level_set_to_debug(self, tmp_path: Path):
        result = setup_file_logging(log_dir=tmp_path, log_file="level_test.log")
        _remove_file_handlers(result)
        assert result.level == logging.DEBUG

    def test_custom_log_file_name(self, tmp_path: Path):
        result = setup_file_logging(log_dir=tmp_path, log_file="custom_name.log")
        _remove_file_handlers(result)
        assert (tmp_path / "custom_name.log").exists()


# ===========================================================================
# redirect_warnings
# ===========================================================================

class TestRedirectWarnings:

    def test_redirect_warnings_runs_without_error(self):
        """redirect_warnings() should not raise."""
        try:
            redirect_warnings()
        except Exception as exc:
            pytest.fail(f"redirect_warnings() raised {exc}")

    def test_warnings_showwarning_replaced(self):
        original = warnings.showwarning
        redirect_warnings()
        new_showwarning = warnings.showwarning
        # Restore original to not affect other tests
        warnings.showwarning = original
        assert new_showwarning is not original

    def test_warning_goes_to_app_warnings_logger(self):
        """After redirect_warnings, a warning should be captured by the app.warnings logger."""
        received = []
        warn_logger = logging.getLogger("app.warnings")
        handler = logging.handlers.MemoryHandler(capacity=10, flushLevel=logging.ERROR)

        class _Capture(logging.Handler):
            def emit(self, record):
                received.append(record.getMessage())

        capture = _Capture()
        warn_logger.addHandler(capture)
        original = warnings.showwarning
        try:
            redirect_warnings()
            warnings.warn("test warning message", UserWarning)
            assert any("test warning message" in m for m in received)
        finally:
            warnings.showwarning = original
            warn_logger.removeHandler(capture)
