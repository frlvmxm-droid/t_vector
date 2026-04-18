from hw_profile import tune_runtime_by_input_size


def test_tune_runtime_by_input_size_small_keeps_values():
    out = tune_runtime_by_input_size(input_bytes=100 * 1024 * 1024, chunk=4000, sbert_batch=64, kmeans_batch=2048)
    assert out["chunk"] == 4000
    assert out["sbert_batch"] == 64


def test_tune_runtime_by_input_size_large_reduces_values():
    out = tune_runtime_by_input_size(input_bytes=3 * 1024 ** 3, chunk=5000, sbert_batch=128, kmeans_batch=4096)
    assert out["chunk"] < 5000
    assert out["sbert_batch"] < 128
    assert out["kmeans_batch"] < 4096
