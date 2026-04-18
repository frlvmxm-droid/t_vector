from cluster_export_service import build_cluster_model_bundle


def test_build_cluster_model_bundle_sets_contract_fields():
    b = build_cluster_model_bundle(
        schema_version=1,
        vectorizer=object(),
        algo="kmeans",
        k_clusters=4,
        kw=["a", "b"],
        centers=[[0.1, 0.2]],
    )
    assert b["artifact_type"] == "cluster_model_bundle"
    assert b["schema_version"] == 1
    assert b["algo"] == "kmeans"
    assert b["K"] == 4
    assert b["kw"] == ["a", "b"]
