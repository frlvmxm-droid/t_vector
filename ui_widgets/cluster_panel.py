# -*- coding: utf-8 -*-
"""Voilà widget: 'Кластеризация' panel — files → ClusteringWorkflow.run → CSV."""
from __future__ import annotations

import pathlib
import tempfile
import threading
import traceback
from typing import Any, List

from ui_widgets.io import download_link, save_upload_to_tmp
from ui_widgets.progress import ProgressPanel


_SUPPORTED_COMBOS = {
    "tfidf": ("kmeans", "agglo", "lda", "hdbscan"),
    "sbert": ("kmeans",),
    "combo": ("kmeans",),
    "ensemble": ("kmeans",),
}


def build_cluster_panel() -> Any:
    """Builds the 'Кластеризация' tab. Returns an ipywidgets container."""
    import ipywidgets as w

    title = w.HTML("<h3 style='margin:6px 0'>🧩 Кластеризация</h3>")

    # ── Inputs ─────────────────────────────────────────────────────────
    upload = w.FileUpload(
        accept=".xlsx,.csv", multiple=True,
        description="Загрузить файлы",
        layout=w.Layout(width="320px"),
    )
    shared_paths = w.Textarea(
        value="", placeholder="или пути на сервере (по одному на строку)",
        description="Пути:", rows=3,
        layout=w.Layout(width="520px"),
    )
    text_col = w.Text(value="text", description="Колонка текста:",
                      layout=w.Layout(width="280px"))

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
        layout=w.Layout(width="520px"),
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
        layout=w.Layout(width="420px"),
    )
    k_clusters = w.IntSlider(value=8, min=2, max=100, step=1,
                             description="K:",
                             layout=w.Layout(width="420px"))

    # SBERT-specific
    sbert_model = w.Text(
        value="cointegrated/rubert-tiny2",
        description="SBERT model:",
        layout=w.Layout(width="520px"),
    )
    sbert_model2 = w.Text(
        value="",
        placeholder="второй SBERT (только для ensemble)",
        description="SBERT model #2:",
        layout=w.Layout(width="520px"),
    )
    # Combo-specific
    combo_alpha = w.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05,
                                description="combo α:",
                                layout=w.Layout(width="320px"))
    combo_svd_dim = w.IntSlider(value=200, min=50, max=500, step=50,
                                description="SVD dim:",
                                layout=w.Layout(width="320px"))

    run_btn = w.Button(description="Кластеризовать", button_style="primary",
                       icon="play", layout=w.Layout(width="200px"))

    progress = ProgressPanel(log_height="240px")
    summary_out = w.Output()
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
        k_clusters.disabled = algo.value == "hdbscan"
        _ = current_algos  # keep for future validation hooks

    vec_mode.observe(_sync_visibility, names="value")
    algo.observe(_sync_visibility, names="value")
    _sync_visibility()

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
            }
            if vec_mode.value == "ensemble":
                snap["sbert_model2"] = (
                    sbert_model2.value.strip() or snap["sbert_model"]
                )
            if vec_mode.value == "combo":
                snap["combo_alpha"] = float(combo_alpha.value)
                snap["combo_svd_dim"] = int(combo_svd_dim.value)

            progress.update(0.05, "Запуск pipeline…")
            from cluster_workflow_service import ClusteringWorkflow
            result = ClusteringWorkflow.run(
                files_snapshot=[str(p) for p in paths],
                snap=snap,
                log_cb=progress.log,
                progress_cb=lambda p, s: progress.update(0.05 + 0.9 * float(p), s),
            )

            progress.update(0.98, "Формирование ссылки на скачивание…")
            summary_out.clear_output()
            with summary_out:
                print(f"Кластеров          : {result.n_clusters}")
                print(f"Шум (noise)        : {result.n_noise}")
                print(f"Файл результата    : {out_csv.name}")

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
        download_box.children = ()
        threading.Thread(target=_run_clustering, daemon=True).start()

    run_btn.on_click(_on_click)

    warn_html = w.HTML(
        "<div style='color:#a66;font-size:0.92em'>"
        "⚠️ Поддержаны только комбо: "
        "tfidf+{kmeans,agglo,lda,hdbscan}, sbert+kmeans, "
        "combo+kmeans, ensemble+kmeans. Для BERTopic / SetFit / "
        "FASTopic используйте CLI с <code>--allow-skeleton</code> "
        "или desktop UI."
        "</div>"
    )

    return w.VBox([
        title,
        w.HTML("<i>Шаг 1 — входные файлы (можно несколько).</i>"),
        w.HBox([upload, shared_paths]),
        text_col,
        w.HTML("<i>Шаг 2 — выбор алгоритма.</i>"),
        w.HBox([vec_mode, algo]),
        k_clusters,
        sbert_model,
        sbert_model2,
        w.HBox([combo_alpha, combo_svd_dim]),
        warn_html,
        run_btn,
        progress.widget,
        w.HTML("<b>Результаты:</b>"),
        summary_out,
        download_box,
    ])


# ─── helpers ────────────────────────────────────────────────────────────
def _resolve_input_paths(upload_widget: Any, shared: str) -> List[pathlib.Path]:
    out: List[pathlib.Path] = []
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
