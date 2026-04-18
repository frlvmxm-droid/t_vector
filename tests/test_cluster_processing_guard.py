import threading

import app_cluster
from app_cluster import ClusterTabMixin


class _DummyCluster(ClusterTabMixin):
    def __init__(self):
        self._proc_lock = threading.Lock()
        self._processing = False

    # run_cluster touches these methods only after successful preflight
    def _prepare_cluster_run_context(self):
        return None, None


def test_run_cluster_does_not_stick_processing_on_early_preflight_return():
    d = _DummyCluster()
    d.run_cluster()
    assert d._processing is False


def test_prepare_cluster_run_context_resets_processing_when_preconditions_fail(monkeypatch):
    class _DummyPrepare(ClusterTabMixin):
        def __init__(self):
            self._proc_lock = threading.Lock()
            self._processing = True

    d = _DummyPrepare()

    monkeypatch.setattr(app_cluster, "validate_cluster_preconditions", lambda _app: False)
    snap, files = d._prepare_cluster_run_context()

    assert snap is None and files is None
    assert d._processing is False
