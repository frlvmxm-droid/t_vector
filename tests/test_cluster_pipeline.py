from app_cluster_pipeline import build_cluster_role_context, build_t5_source_text
import app_cluster_pipeline
import pytest


def test_build_cluster_role_context_client_mode_sets_role_weights():
    snap = {
        "cluster_role_mode": "client",
        "ignore_chatbot_cluster": True,
    }
    ctx = build_cluster_role_context(snap)
    assert ctx.role_label == "Только клиент"
    assert ctx.ignore_chatbot_label == "да"
    assert ctx.cluster_snap["auto_profile"] == "off"
    assert ctx.cluster_snap["base_w"]["w_client"] == 3
    assert ctx.cluster_snap["base_w"]["w_operator"] == 0


def test_build_t5_source_text_uses_call_and_chat_columns():
    snap = {"call_col": "call", "chat_col": "chat"}
    cluster_snap = {"ignore_chatbot": True}
    header = ["id", "call", "chat"]
    row = [1, "хочу перевод", "где статус?"]
    text = build_t5_source_text(row, header, snap, cluster_snap)
    assert "хочу перевод" in text
    assert "где статус" in text


def test_build_t5_source_text_uses_precomputed_header_index():
    snap = {"call_col": "call", "chat_col": "chat"}
    cluster_snap = {"ignore_chatbot": True}
    header = ["id", "call", "chat"]
    row = [1, "хочу перевод", "где статус?"]
    header_index = {"id": 0, "call": 1, "chat": 2}

    text = build_t5_source_text(row, header, snap, cluster_snap, header_index=header_index)
    assert "хочу перевод" in text


def test_build_t5_source_text_resolves_case_insensitive_indexes(monkeypatch):
    snap = {"call_col": "CALL", "chat_col": "CHAT"}
    cluster_snap = {"ignore_chatbot": True}
    header = ["id", "call", "chat"]
    row = [1, "хочу перевод", "где статус?"]
    monkeypatch.setattr(
        app_cluster_pipeline,
        "parse_dialog_roles",
        lambda raw, **_kw: (str(raw), "", "", False),
    )
    text = build_t5_source_text(row, header, snap, cluster_snap)
    assert "хочу перевод" in text
    assert "где статус?" in text


def test_build_t5_source_text_output_identical_with_and_without_header_index():
    snap = {"call_col": "call", "chat_col": "chat"}
    cluster_snap = {"ignore_chatbot": True}
    header = ["id", "call", "chat"]
    row = [7, "КЛИЕНТ: хочу перевод", "ОПЕРАТОР: уже в работе"]
    text_plain = build_t5_source_text(row, header, snap, cluster_snap)
    text_indexed = build_t5_source_text(
        row,
        header,
        snap,
        cluster_snap,
        header_index={"id": 0, "call": 1, "chat": 2},
    )
    assert text_plain == text_indexed


def test_build_cluster_role_context_rejects_unsupported_algo():
    with pytest.raises(ValueError):
        build_cluster_role_context({"cluster_algo": "unknown", "cluster_vec_mode": "tfidf"})


def test_build_cluster_role_context_rejects_unsupported_vector_mode():
    with pytest.raises(ValueError):
        build_cluster_role_context({"cluster_algo": "kmeans", "cluster_vec_mode": "bad-mode"})
