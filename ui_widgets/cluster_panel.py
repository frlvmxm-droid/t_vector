"""Voilà widget: 'Кластеризация' panel — files → ClusteringWorkflow.run → CSV."""
from __future__ import annotations

import os
import pathlib
import tempfile
import threading
import traceback
from typing import Any

from ui_widgets.io import download_link, save_upload_to_tmp
from ui_widgets.progress import ProgressPanel
from ui_widgets.theme import metric_card, section_card

_SUPPORTED_COMBOS = {
    "tfidf": ("kmeans", "agglo", "lda", "hdbscan"),
    "sbert": ("kmeans",),
    "combo": ("kmeans",),
    "ensemble": ("kmeans",),
}


def build_cluster_panel() -> Any:
    """Builds the 'Кластеризация' tab. Returns an ipywidgets container."""
    import ipywidgets as w

    # ── Inputs ─────────────────────────────────────────────────────────
    upload = w.FileUpload(
        accept=".xlsx,.csv", multiple=True,
        description="Загрузить файлы",
        layout=w.Layout(width="260px"),
    )
    shared_paths = w.Textarea(
        value="", placeholder="или пути на сервере (по одному на строку)",
        description="Пути:", rows=3,
        layout=w.Layout(width="440px"),
    )
    text_col = w.Text(value="text", description="Колонка текста:",
                      layout=w.Layout(width="240px"))

    # ── Algorithm choice ───────────────────────────────────────────────
    vec_mode = w.Dropdown(
        options=[
            ("TF-IDF (быстро, классика)", "tfidf"),
            ("SBERT (семантика, медленнее)", "sbert"),
            ("Combo (TF-IDF + SBERT, взвешено)", "combo"),
            ("Ensemble (TF-IDF + 2×SBERT, silhouette-winner)", "ensemble"),
        ],
        value="tfidf",
        description="Векторизация:",
        layout=w.Layout(width="440px"),
    )
    algo = w.Dropdown(
        options=[
            ("KMeans", "kmeans"),
            ("Agglomerative (Ward, ≤5000 строк)", "agglo"),
            ("LDA (topic model)", "lda"),
            ("HDBSCAN (сам находит K)", "hdbscan"),
        ],
        value="kmeans",
        description="Алгоритм:",
        layout=w.Layout(width="360px"),
    )
    k_clusters = w.IntSlider(value=8, min=2, max=100, step=1,
                             description="K:",
                             layout=w.Layout(width="360px"))

    # SBERT-specific
    sbert_model = w.Text(
        value="cointegrated/rubert-tiny2",
        description="SBERT model:",
        layout=w.Layout(width="440px"),
    )
    sbert_model2 = w.Text(
        value="",
        placeholder="второй SBERT (только для ensemble)",
        description="SBERT model #2:",
        layout=w.Layout(width="440px"),
    )
    # Combo-specific
    combo_alpha = w.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05,
                                description="combo α:",
                                layout=w.Layout(width="280px"))
    combo_svd_dim = w.IntSlider(value=200, min=50, max=500, step=50,
                                description="SVD dim:",
                                layout=w.Layout(width="280px"))

    # Common cluster knobs
    n_init_cluster = w.IntSlider(value=10, min=1, max=50, step=1,
                                 description="n_init:",
                                 layout=w.Layout(width="260px"))
    cluster_min_df = w.IntSlider(value=2, min=1, max=20, step=1,
                                 description="min_df:",
                                 layout=w.Layout(width="260px"))
    sbert_device = w.Dropdown(
        options=[("Auto", "auto"), ("CPU", "cpu"),
                 ("CUDA", "cuda"), ("MPS", "mps")],
        value="auto", description="SBERT device:",
        layout=w.Layout(width="260px"),
        style={"description_width": "initial"},
    )

    # HDBSCAN-specific knobs (visible only when algo=hdbscan)
    hdbscan_min_cs = w.IntSlider(value=5, min=2, max=100, step=1,
                                 description="min_cluster_size:",
                                 layout=w.Layout(width="320px"),
                                 style={"description_width": "initial"})
    hdbscan_min_samples = w.IntSlider(value=5, min=1, max=50, step=1,
                                      description="min_samples:",
                                      layout=w.Layout(width="280px"),
                                      style={"description_width": "initial"})
    hdbscan_eps = w.FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01,
                                description="eps:",
                                layout=w.Layout(width="260px"))

    # LDA-specific
    lda_max_iter = w.IntSlider(value=10, min=5, max=100, step=5,
                               description="LDA max_iter:",
                               layout=w.Layout(width="280px"),
                               style={"description_width": "initial"})

    # MiniBatch toggle
    minibatch_kmeans = w.Checkbox(value=False, description="MiniBatch KMeans")
    minibatch_size = w.IntSlider(value=1024, min=128, max=8192, step=128,
                                 description="batch:",
                                 layout=w.Layout(width="260px"))

    # UMAP projection (opt-in; honored by build_vectors in phase 11+)
    umap_enabled = w.Checkbox(value=False, description="UMAP projection")
    umap_n_components = w.IntSlider(value=10, min=2, max=64, step=1,
                                    description="n_comp:",
                                    layout=w.Layout(width="240px"))
    umap_n_neighbors = w.IntSlider(value=15, min=5, max=100, step=1,
                                   description="n_neigh:",
                                   layout=w.Layout(width="240px"))
    umap_min_dist = w.FloatSlider(value=0.1, min=0.0, max=1.0, step=0.01,
                                  description="min_dist:",
                                  layout=w.Layout(width="240px"))
    umap_metric = w.Dropdown(
        options=["cosine", "euclidean", "correlation"],
        value="cosine", description="metric:",
        layout=w.Layout(width="220px"),
    )
    umap_use_pca_pre = w.Checkbox(value=False, description="PCA pre-reduce")
    umap_box = w.VBox([
        w.HBox([umap_enabled, umap_use_pca_pre]),
        w.HBox([umap_n_components, umap_n_neighbors]),
        w.HBox([umap_min_dist, umap_metric]),
    ])
    umap_accordion = w.Accordion(children=[umap_box])
    umap_accordion.set_title(0, "Проекция (UMAP)")
    umap_accordion.selected_index = None

    # Postprocess switches (Phase 11 will wire them to services)
    _llm_api_key = os.environ.get("BRT_LLM_API_KEY", "")
    _llm_provider = os.environ.get("BRT_LLM_PROVIDER", "offline")
    use_llm_naming = w.Checkbox(
        value=False, description="LLM-naming кластеров",
        disabled=(not _llm_api_key and _llm_provider.lower() != "offline"),
    )
    use_t5_summary = w.Checkbox(value=False, description="T5 summary (ru-T5)")
    use_auto_k = w.Checkbox(value=False, description="Auto-K (silhouette)")
    merge_similar = w.Checkbox(value=False, description="Merge similar clusters")
    merge_threshold = w.FloatSlider(
        value=0.85, min=0.5, max=1.0, step=0.01,
        description="merge thr:",
        layout=w.Layout(width="280px"),
    )
    n_repr_examples = w.IntSlider(
        value=5, min=1, max=20, step=1,
        description="repr-K:",
        layout=w.Layout(width="240px"),
    )

    run_btn = w.Button(description="▶  Кластеризовать", button_style="primary",
                       layout=w.Layout(width="220px"))

    progress = ProgressPanel(log_height="220px")
    summary_out = w.Output()
    metric_cards = w.HTML("")
    download_box = w.VBox([])

    # ── Dynamic visibility ─────────────────────────────────────────────
    def _sync_visibility(*_args: Any) -> None:
        mode = vec_mode.value
        allowed = _SUPPORTED_COMBOS.get(mode, ())
        current_algos = [o[1] for o in algo.options]
        if algo.value not in allowed:
            # switch to first allowed
            for opt in algo.options:
                if opt[1] in allowed:
                    algo.value = opt[1]
                    break
        # tag unsupported algo options (can't disable per-option in Dropdown,
        # so we log a warning in progress panel on click instead)
        sbert_model.layout.display = "" if mode in ("sbert", "combo", "ensemble") else "none"
        sbert_model2.layout.display = "" if mode == "ensemble" else "none"
        combo_alpha.layout.display = "" if mode == "combo" else "none"
        combo_svd_dim.layout.display = "" if mode == "combo" else "none"
        sbert_device.layout.display = "" if mode in ("sbert", "combo", "ensemble") else "none"
        k_clusters.disabled = algo.value == "hdbscan" or use_auto_k.value
        hdbscan_min_cs.layout.display = "" if algo.value == "hdbscan" else "none"
        hdbscan_min_samples.layout.display = "" if algo.value == "hdbscan" else "none"
        hdbscan_eps.layout.display = "" if algo.value == "hdbscan" else "none"
        lda_max_iter.layout.display = "" if algo.value == "lda" else "none"
        _ = current_algos  # keep for future validation hooks

    vec_mode.observe(_sync_visibility, names="value")
    algo.observe(_sync_visibility, names="value")
    use_auto_k.observe(_sync_visibility, names="value")
    _sync_visibility()

    cancel_event_box: dict[str, threading.Event] = {}

    def _run_clustering() -> None:
        try:
            paths = _resolve_input_paths(upload, shared_paths.value)
            if not paths:
                progress.log("❌ Загрузите файл(ы) или укажите путь(и).")
                progress.mark_error("Нет входов")
                return
            progress.log(f"  входных файлов: {len(paths)}")

            allowed = _SUPPORTED_COMBOS.get(vec_mode.value, ())
            if algo.value not in allowed:
                progress.log(
                    f"❌ Комбо {vec_mode.value}+{algo.value} не поддержано в UI. "
                    f"Доступно для {vec_mode.value!r}: {allowed}."
                )
                progress.mark_error("Неподдержанное комбо")
                return

            tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="brt_cluster_"))
            out_csv = tmp_dir / "clusters.csv"

            snap = {
                "cluster_vec_mode": vec_mode.value,
                "cluster_algo": algo.value,
                "k_clusters": int(k_clusters.value),
                "text_col": text_col.value,
                "output_path": str(out_csv),
                "sbert_model": sbert_model.value.strip() or "cointegrated/rubert-tiny2",
                "sbert_device": sbert_device.value,
                "n_init_cluster": int(n_init_cluster.value),
                "cluster_min_df": int(cluster_min_df.value),
                # Post-process switches (Phase 11 service layer honors these)
                "cluster_auto_k": bool(use_auto_k.value),
                "use_llm_naming": bool(use_llm_naming.value),
                "use_t5_summary": bool(use_t5_summary.value),
                "merge_similar_clusters": bool(merge_similar.value),
                "merge_threshold": float(merge_threshold.value),
                "n_repr_examples": int(n_repr_examples.value),
                # MiniBatch KMeans
                "minibatch_kmeans": bool(minibatch_kmeans.value),
                "minibatch_batch_size": int(minibatch_size.value),
                # UMAP projection (honored by build_vectors when enabled)
                "use_umap": bool(umap_enabled.value),
                "umap_n_components": int(umap_n_components.value),
                "umap_n_neighbors": int(umap_n_neighbors.value),
                "umap_min_dist": float(umap_min_dist.value),
                "umap_metric": umap_metric.value,
                "umap_use_pca_pre": bool(umap_use_pca_pre.value),
            }
            if vec_mode.value == "ensemble":
                snap["sbert_model2"] = (
                    sbert_model2.value.strip() or snap["sbert_model"]
                )
            if vec_mode.value == "combo":
                snap["combo_alpha"] = float(combo_alpha.value)
                snap["combo_svd_dim"] = int(combo_svd_dim.value)
            if algo.value == "hdbscan":
                snap["hdbscan_min_cluster_size"] = int(hdbscan_min_cs.value)
                snap["hdbscan_min_samples"] = int(hdbscan_min_samples.value)
                snap["hdbscan_eps"] = float(hdbscan_eps.value)
            if algo.value == "lda":
                snap["lda_max_iter"] = int(lda_max_iter.value)

            progress.update(0.05, "Запуск pipeline…")
            from cluster_workflow_service import (
                ClusteringWorkflow,
                WorkflowCancelled,
            )
            try:
                result = ClusteringWorkflow.run(
                    files_snapshot=[str(p) for p in paths],
                    snap=snap,
                    log_cb=progress.log,
                    progress_cb=lambda p, s: progress.update(0.05 + 0.9 * float(p), s),
                    cancel_event=cancel_event_box.get("event"),
                )
            except WorkflowCancelled as exc:
                progress.log(f"🛑 {exc}")
                progress.mark_error("Отменено")
                return

            progress.update(0.98, "Формирование ссылки на скачивание…")
            summary_out.clear_output()
            with summary_out:
                print(f"Кластеров          : {result.n_clusters}")
                print(f"Шум (noise)        : {result.n_noise}")
                print(f"Файл результата    : {out_csv.name}")

            metric_cards.value = "<div style='display:flex; gap:12px; margin:4px 0;'>" + "".join([
                metric_card("КЛАСТЕРОВ", str(result.n_clusters),
                            f"{vec_mode.value} + {algo.value}"),
                metric_card("ШУМ (NOISE)", str(result.n_noise),
                            "только для HDBSCAN"),
                metric_card("ВХОДНЫХ ФАЙЛОВ", str(len(paths)),
                            f"колонка: {text_col.value!r}"),
            ]) + "</div>"

            download_box.children = (download_link(out_csv, f"📥 Скачать {out_csv.name}"),)
            progress.mark_done()
        except Exception as exc:  # noqa: BLE001 — UI entry point
            progress.log(f"❌ {exc}")
            progress.log(traceback.format_exc())
            progress.mark_error(str(exc))
        finally:
            run_btn.disabled = False
            run_btn.description = "Кластеризовать"

    def _on_click(_: Any) -> None:
        run_btn.disabled = True
        run_btn.description = "⏳ Кластеризация…"
        progress.reset("Старт…")
        summary_out.clear_output()
        metric_cards.value = ""
        download_box.children = ()
        event = threading.Event()
        cancel_event_box["event"] = event
        progress.attach_cancel_event(event)
        threading.Thread(target=_run_clustering, daemon=True).start()

    run_btn.on_click(_on_click)

    warn_html = w.HTML(
        "<div style='color:#f0b429;font-size:0.88em;margin-top:6px'>"
        "⚠ Поддержаны только комбо: "
        "<code>tfidf+{kmeans,agglo,lda,hdbscan}</code>, "
        "<code>sbert+kmeans</code>, "
        "<code>combo+kmeans</code>, <code>ensemble+kmeans</code>. "
        "Для BERTopic / SetFit / FASTopic используйте CLI с "
        "<code>--allow-skeleton</code> или desktop UI."
        "</div>"
    )

    sources_card = section_card(
        "ИСТОЧНИКИ",
        [w.HBox([upload, shared_paths]), text_col],
        subtitle="Один или несколько XLSX/CSV + колонка текста.",
    )
    algo_card = section_card(
        "АЛГОРИТМ",
        [
            w.HBox([vec_mode, algo]),
            w.HBox([k_clusters, use_auto_k]),
            w.HBox([n_init_cluster, cluster_min_df, sbert_device]),
            sbert_model,
            sbert_model2,
            w.HBox([combo_alpha, combo_svd_dim]),
            w.HBox([hdbscan_min_cs, hdbscan_min_samples, hdbscan_eps]),
            lda_max_iter,
            w.HBox([minibatch_kmeans, minibatch_size]),
            warn_html,
        ],
        subtitle="Векторизация → кластеризация.",
    )
    postproc_card = section_card(
        "ПОСТ-ОБРАБОТКА",
        [
            w.HBox([use_llm_naming, use_t5_summary]),
            w.HBox([merge_similar, merge_threshold, n_repr_examples]),
            umap_accordion,
        ],
        subtitle=(
            "Auto-K / LLM-naming / T5-summary / merge similar — "
            "требуют Phase 11 service-layer. UMAP применяется до кластеризации."
        ),
    )
    run_card = section_card(
        "РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ",
        [w.HBox([run_btn, download_box]), progress.widget, metric_cards, summary_out],
        subtitle="Запуск, прогресс и сводка по кластерам.",
    )
    return w.VBox([sources_card, algo_card, postproc_card, run_card])


# ─── helpers ────────────────────────────────────────────────────────────
def _resolve_input_paths(upload_widget: Any, shared: str) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    if upload_widget.value:
        out.extend(save_upload_to_tmp(upload_widget))
    for line in (shared or "").splitlines():
        line = line.strip()
        if not line:
            continue
        p = pathlib.Path(line).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Файл не найден: {p}")
        out.append(p)
    return out
