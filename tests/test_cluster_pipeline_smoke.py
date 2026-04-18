from app_cluster_pipeline import (
    prepare_inputs,
    build_vectors,
    run_clustering,
    postprocess_clusters,
    export_cluster_outputs,
)


def test_cluster_pipeline_smoke_structures():
    snap = {
        "cluster_role_mode": "all",
        "ignore_chatbot_cluster": True,
        "call_col": "call",
        "chat_col": "chat",
    }
    files_snapshot = ["a.xlsx", "b.xlsx"]
    prepared = prepare_inputs(files_snapshot, snap)
    vectors = build_vectors(prepared, snap)
    clustered = run_clustering(vectors, snap)
    post = postprocess_clusters(clustered, prepared, snap)
    exported = export_cluster_outputs(post, snap)

    assert prepared.files_snapshot == files_snapshot
    assert vectors.prepared is prepared
    assert clustered.vectors is vectors
    assert post.result is clustered
    assert exported.postprocessed is post
