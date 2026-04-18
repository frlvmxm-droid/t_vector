import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _imports_from(rel: str) -> dict[str, set[str]]:
    tree = ast.parse(_read(rel))
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.setdefault(node.module, set()).update(alias.name for alias in node.names)
    return out


def _top_level_imports(rel: str) -> set[str]:
    tree = ast.parse(_read(rel))
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def _calls(rel: str) -> set[tuple[str | None, str]]:
    tree = ast.parse(_read(rel))
    out: set[tuple[str | None, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = node.func.value.id if isinstance(node.func.value, ast.Name) else None
            out.add((base, node.func.attr))
    return out




def _has_function(rel: str, func_name: str) -> bool:
    tree = ast.parse(_read(rel))
    return any(isinstance(node, ast.FunctionDef) and node.name == func_name for node in ast.walk(tree))


def _class_has_method(rel: str, class_name: str, method_name: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(isinstance(item, ast.FunctionDef) and item.name == method_name for item in node.body)
    return False


def _has_attr_call(rel: str, base_name: str, attr_name: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == base_name and node.func.attr == attr_name:
                return True
    return False


def _has_bind_all_call(rel: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "bind_all"
        ):
            return True
    return False


def _has_string_constant(rel: str, value: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == value:
            return True
    return False


def _has_string_containing(rel: str, needle: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and needle in node.value:
            return True
    return False


def _has_attr_call_with_string_arg(rel: str, base_name: str, attr_name: str, needle: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == base_name
            and node.func.attr == attr_name
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and needle in arg.value:
                    return True
    return False


def _has_attr_assignment(rel: str, base_name: str, attr_name: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == base_name
                    and target.attr == attr_name
                ):
                    return True
    return False


def _has_call_with_keyword(rel: str, func_name: str, kw_name: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
            if any(kw.arg == kw_name for kw in node.keywords):
                return True
    return False


def _has_list_saved_kw_pattern(rel: str) -> bool:
    tree = ast.parse(_read(rel))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "list"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and isinstance(node.args[0].func.value, ast.Name)
            and node.args[0].func.value.id == "_saved"
            and node.args[0].func.attr == "get"
            and node.args[0].args
            and isinstance(node.args[0].args[0], ast.Constant)
            and node.args[0].args[0].value == "kw"
        ):
            return True
    return False


def test_single_joblib_load_entrypoint():
    offenders = []
    for py in ROOT.glob("*.py"):
        calls = _calls(py.name)
        if ("joblib", "load") in calls and py.name != "model_loader.py":
            offenders.append(py.name)
    assert offenders == [], f"joblib.load should stay only in model_loader.py: {offenders}"


def test_cluster_ui_uses_service_layer():
    cluster_imports = _imports_from("app_cluster.py")
    cluster_service_imports = cluster_imports.get("app_cluster_service", set())
    assert "ClusterElbowSelector" in cluster_service_imports
    assert "LLMClient" in cluster_service_imports
    cluster_loader_imports = cluster_imports.get("cluster_model_loader", set())
    assert "ensure_cluster_model_trusted" in cluster_loader_imports
    incr_imports = cluster_imports.get("cluster_incremental_service", set())
    assert "load_and_apply_incremental_model" in incr_imports
    export_imports = cluster_imports.get("cluster_export_service", set())
    assert "build_cluster_model_bundle" in export_imports
    assert "urllib.request" not in cluster_imports
    assert "urllib.error" not in cluster_imports
    assert "joblib" not in cluster_imports


def test_cluster_gpu_optional_imports_are_not_top_level():
    tops = _top_level_imports("app_cluster.py")
    assert "cuml.cluster" not in tops
    assert "cuml.manifold.umap" not in tops


def test_apply_ui_uses_service_layer():
    apply_imports = _imports_from("app_apply.py")
    assert "EnsemblePredictor" in apply_imports.get("app_apply_service", set())


def test_tabs_use_workflow_layer():
    train_imports = _imports_from("app_train.py")
    apply_imports = _imports_from("app_apply.py")
    cluster_imports = _imports_from("app_cluster.py")
    assert "validate_train_preconditions" in train_imports.get("app_train_workflow", set())
    assert "build_validated_train_snapshot" in train_imports.get("app_train_workflow", set())
    assert "validate_apply_preconditions" in apply_imports.get("app_apply_workflow", set())
    assert "build_validated_apply_snapshot" in apply_imports.get("app_apply_workflow", set())
    assert "prepare_cluster_run_context" in cluster_imports.get("cluster_run_coordinator", set())
    coord_imports = _imports_from("cluster_run_coordinator.py")
    assert "build_cluster_runtime_snapshot" in coord_imports.get("cluster_state_adapter", set())


def test_tabs_use_view_layer():
    train_imports = _imports_from("app_train.py")
    apply_imports = _imports_from("app_apply.py")
    cluster_imports = _imports_from("app_cluster.py")
    cluster_ui_builder_imports = _imports_from("cluster_ui_builder.py")
    assert "build_train_files_card" in train_imports.get("app_train_view", set())
    assert "build_apply_files_card" in apply_imports.get("app_apply_view", set())
    assert "build_cluster_primary_sections" in cluster_imports.get("cluster_ui_builder", set())
    assert "build_cluster_files_card" in cluster_ui_builder_imports.get("app_cluster_view", set())


def test_sbert_compat_patches_are_bootstrap_scoped():
    assert _has_function("ml_vectorizers.py", "_run_sbert_bootstrap_patches")
    assert _has_attr_call("ml_vectorizers.py", "ml_compat", "apply_early_compat_patches")


def test_cluster_run_has_preflight_context_and_processing_reset():
    assert _class_has_method("app_cluster.py", "ClusterTabMixin", "_prepare_cluster_run_context")
    assert _has_attr_assignment("cluster_run_coordinator.py", "app", "_processing")
    assert _has_attr_call_with_string_arg("app_cluster.py", "_log", "debug", "cluster cleanup gc.collect failed")
    assert _has_attr_call_with_string_arg("app_cluster.py", "_log", "debug", "cluster cleanup torch.cuda.empty_cache failed")


def test_scrollable_frame_uses_local_wheel_bindings():
    assert not _has_bind_all_call("ui_widgets.py")
    assert _has_string_containing("ui_widgets.py", "<MouseWheel>")


def test_cluster_t5_builder_uses_precomputed_header_index():
    assert _has_call_with_keyword("app_cluster.py", "build_t5_source_text", "header_index")


def test_cluster_incremental_loader_normalizes_kw_payload():
    tree = ast.parse(_read("app_cluster.py"))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "load_and_apply_incremental_model" in calls
    assert not _has_list_saved_kw_pattern("app_cluster.py")
    assert _has_attr_call("app_cluster.py", "_log", "exception")
