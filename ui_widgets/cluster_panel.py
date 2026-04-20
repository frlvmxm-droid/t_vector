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
            k_clusters,
            sbert_model,
            sbert_model2,
            w.HBox([combo_alpha, combo_svd_dim]),
            warn_html,
        ],
        subtitle="Векторизация → кластеризация.",
    )
    run_card = section_card(
        "РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ",
        [w.HBox([run_btn, download_box]), progress.widget, metric_cards, summary_out],
        subtitle="Запуск, прогресс и сводка по кластерам.",
    )
    return w.VBox([sources_card, algo_card, run_card])


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
