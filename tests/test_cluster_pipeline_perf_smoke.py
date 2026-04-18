import time

from app_cluster_pipeline import build_t5_source_text, prepare_inputs


def test_prepare_inputs_perf_smoke():
    snap = {
        "cluster_algo": "kmeans",
        "cluster_vec_mode": "tfidf",
        "call_col": "call",
        "chat_col": "chat",
    }
    files = [f"f_{i}.xlsx" for i in range(200)]
    t0 = time.perf_counter()
    out = prepare_inputs(files, snap)
    dt = time.perf_counter() - t0
    assert len(out.files_snapshot) == 200
    assert dt < 0.2


def test_build_t5_source_text_perf_smoke():
    snap = {"call_col": "call", "chat_col": "chat"}
    cluster_snap = {"ignore_chatbot": True}
    header = ["id", "call", "chat"]
    header_index = {"id": 0, "call": 1, "chat": 2}
    row = [1, "перевод между счетами", "статус операции"]
    t0 = time.perf_counter()
    for _ in range(2000):
        build_t5_source_text(row, header, snap, cluster_snap, header_index=header_index)
    dt = time.perf_counter() - t0
    assert dt < 0.6


def test_build_t5_source_text_indexed_not_slower_than_plain_smoke():
    snap = {"call_col": "call", "chat_col": "chat"}
    cluster_snap = {"ignore_chatbot": True}
    header = ["id", "call", "chat"]
    row = [1, "перевод между счетами", "статус операции"]
    loops = 2500

    t0 = time.perf_counter()
    for _ in range(loops):
        build_t5_source_text(row, header, snap, cluster_snap)
    dt_plain = time.perf_counter() - t0

    header_index = {"id": 0, "call": 1, "chat": 2}
    t1 = time.perf_counter()
    for _ in range(loops):
        build_t5_source_text(row, header, snap, cluster_snap, header_index=header_index)
    dt_indexed = time.perf_counter() - t1

    assert dt_indexed <= dt_plain * 1.15
