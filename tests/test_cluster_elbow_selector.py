import app_cluster_service
from app_cluster_service import ClusterElbowSelector


def test_elbow_selector_short_ks_returns_first():
    assert ClusterElbowSelector.pick_elbow_k([1.0, 0.9], [2, 3]) == 2


def test_elbow_selector_flat_curve_fallbacks_to_valid_k():
    k = ClusterElbowSelector.pick_elbow_k([1.0, 1.0, 1.0, 1.0], [2, 3, 4, 5])
    assert k in {2, 3, 4, 5}


def test_elbow_selector_without_kneed_uses_fallback(monkeypatch):
    import builtins

    _orig_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "kneed":
            raise ImportError("kneed not installed")
        return _orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    k = ClusterElbowSelector.pick_elbow_k([10.0, 7.0, 5.0, 4.8], [2, 3, 4, 5])
    assert k in {2, 3, 4, 5}


def test_elbow_selector_with_monkeypatched_kneed(monkeypatch):
    class _FakeKL:
        def __init__(self, *args, **kwargs):
            self.knee = 4

    monkeypatch.setitem(app_cluster_service.__dict__, "KneeLocator", None)

    import types
    fake_kneed = types.SimpleNamespace(KneeLocator=_FakeKL)
    monkeypatch.setitem(__import__("sys").modules, "kneed", fake_kneed)

    assert ClusterElbowSelector.pick_elbow_k([10.0, 6.0, 4.0, 3.0], [2, 3, 4, 5]) == 4
