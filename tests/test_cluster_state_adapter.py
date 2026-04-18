from types import SimpleNamespace

import app_cluster_state_adapter


class _Anchors:
    def __init__(self, text: str):
        self._text = text

    def get(self, *_args):
        return self._text


def test_build_cluster_runtime_snapshot_adds_anchor_phrases(monkeypatch):
    app = SimpleNamespace(txt_anchors=_Anchors("a\n \n b "))
    monkeypatch.setattr(
        app_cluster_state_adapter,
        "build_validated_cluster_snapshot",
        lambda _app: {"cluster_algo": "kmeans"},
    )
    snap = app_cluster_state_adapter.build_cluster_runtime_snapshot(app)
    assert snap is not None
    assert snap["anchor_phrases"] == ["a", "b"]


def test_build_cluster_runtime_snapshot_returns_none_when_invalid(monkeypatch):
    monkeypatch.setattr(
        app_cluster_state_adapter,
        "build_validated_cluster_snapshot",
        lambda _app: None,
    )
    snap = app_cluster_state_adapter.build_cluster_runtime_snapshot(SimpleNamespace())
    assert snap is None
