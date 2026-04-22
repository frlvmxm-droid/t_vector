# -*- coding: utf-8 -*-
"""Contract tests for ``cluster_workflow_service`` cancel machinery.

The UI ``⏹ Отмена`` button in ``ui_widgets.progress.ProgressPanel``
ultimately calls ``event.set()`` on a ``threading.Event`` that the
panel passes to ``ClusteringWorkflow.run``. Between stages the
workflow invokes ``_check_cancelled``, which is the single place that
translates the set-event into a :class:`WorkflowCancelled` exception.

These tests lock in that contract so a regression elsewhere
(accidental ``.is_set`` → ``.wait``, silencing the exception,
forgetting a stage check) would be caught without spinning up a
Voilà subprocess.
"""
from __future__ import annotations

import threading

import pytest

from cluster_workflow_service import WorkflowCancelled, _check_cancelled


def test_check_cancelled_none_is_noop():
    # Call-sites without a cancel_event pass None — must never raise.
    _check_cancelled(None)


def test_check_cancelled_unset_event_is_noop():
    event = threading.Event()
    assert not event.is_set()
    _check_cancelled(event)


def test_check_cancelled_set_event_raises():
    event = threading.Event()
    event.set()
    with pytest.raises(WorkflowCancelled):
        _check_cancelled(event)


def test_cancel_exception_message_in_russian():
    # UI surfaces the message verbatim in the progress log — keep it
    # user-facing and in Russian.
    event = threading.Event()
    event.set()
    with pytest.raises(WorkflowCancelled) as exc_info:
        _check_cancelled(event)
    assert "отменена" in str(exc_info.value).lower()


def test_workflow_cancelled_is_runtime_error():
    # Panels catch ``WorkflowCancelled`` by its class, but a sloppy
    # ``except RuntimeError`` elsewhere must still work (it's a
    # RuntimeError subclass by design).
    assert issubclass(WorkflowCancelled, RuntimeError)


def test_cancel_event_clear_after_raise():
    # Contract: _check_cancelled doesn't mutate the event. Panels
    # reuse the same event across stages and rely on is_set staying
    # True until the worker thread unwinds.
    event = threading.Event()
    event.set()
    try:
        _check_cancelled(event)
    except WorkflowCancelled:
        pass
    assert event.is_set(), "_check_cancelled must not clear the event"
