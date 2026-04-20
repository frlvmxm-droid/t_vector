# -*- coding: utf-8 -*-
"""Voilà widget: 'Обучение' panel — upload → train → download model.joblib."""
from __future__ import annotations

import pathlib
import tempfile
import threading
import traceback
from typing import Any

from ui_widgets.io import detect_tabular_format, download_link, save_upload_to_tmp
from ui_widgets.progress import ProgressPanel


def build_train_panel() -> Any:
    """Builds the 'Обучение' tab. Returns an ipywidgets container."""
    import ipywidgets as w

    title = w.HTML("<h3 style='margin:6px 0'>📚 Обучение модели</h3>")

    # ── Inputs ──────────────────────────────────────────────────────────
    upload = w.FileUpload(
        accept=".xlsx,.csv", multiple=False,
        description="Загрузить XLSX/CSV",
        layout=w.Layout(width="320px"),
    )
    shared_path = w.Text(
        value="", placeholder="или путь к файлу на сервере (для больших данных)",
        description="Путь:", layout=w.Layout(width="520px"),
    )
    text_col = w.Text(value="text", description="Колонка текста:",
                      layout=w.Layout(width="280px"))
    label_col = w.Text(value="label", description="Колонка меток:",
                       layout=w.Layout(width="280px"))

    C = w.FloatLogSlider(value=1.0, base=10, min=-2, max=2, step=0.1,
                         description="C (регуляризация):",
                         style={"description_width": "initial"},
                         layout=w.Layout(width="420px"))
    max_iter = w.IntSlider(value=2000, min=200, max=10000, step=100,
                           description="max_iter:",
                           layout=w.Layout(width="420px"))
    test_size = w.FloatSlider(value=0.2, min=0.05, max=0.5, step=0.05,
                              description="test_size:",
                              layout=w.Layout(width="320px"))
    balanced = w.Checkbox(value=False, description="class_weight=balanced")
    use_smote = w.Checkbox(value=True, description="SMOTE-oversampling")
    calib = w.Dropdown(options=["sigmoid", "isotonic"], value="sigmoid",
                       description="Калибровка:")
    fuzzy_dedup = w.Checkbox(value=False, description="Fuzzy-дедуп")
    fuzzy_thr = w.IntSlider(value=92, min=60, max=100, step=1,
                            description="threshold:",
                            layout=w.Layout(width="280px"))

    run_btn = w.Button(description="Обучить", button_style="primary",
                       icon="play", layout=w.Layout(width="160px"))

    # ── Outputs ─────────────────────────────────────────────────────────
    progress = ProgressPanel(log_height="240px")
    metrics_out = w.Output()
    download_box = w.VBox([])

    # ── Worker ──────────────────────────────────────────────────────────
    def _run_training() -> None:
        try:
            data_path = _resolve_data_path(upload, shared_path.value)
            if data_path is None:
                progress.log("❌ Загрузите файл или укажите путь.")
                progress.mark_error("Нет данных")
                return

            progress.update(0.02, f"Чтение {data_path.name}…")
            texts, labels = _read_tabular(data_path, text_col.value, label_col.value)
            if len(texts) < 4:
                progress.log(f"❌ Нужно ≥4 строк, получено {len(texts)}.")
                progress.mark_error("Мало данных")
                return
            if len(set(labels)) < 2:
                progress.log(f"❌ Нужно ≥2 классов, получено {len(set(labels))}.")
                progress.mark_error("Один класс")
                return
            progress.log(f"  строк: {len(texts)} | классов: {len(set(labels))}")

            progress.update(0.05, "Подготовка TF-IDF…")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import FeatureUnion
            features = FeatureUnion([
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                         min_df=2, max_df=0.95, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         min_df=2, max_df=0.95, sublinear_tf=True)),
            ])

            from app_train_service import TrainingWorkflow
            from ml_training import TrainingOptions
            options = TrainingOptions(
                use_smote=use_smote.value,
                calib_method=calib.value,
                use_fuzzy_dedup=fuzzy_dedup.value,
                fuzzy_dedup_threshold=int(fuzzy_thr.value),
            )

            progress.update(0.1, "Обучение…")
            wf = TrainingWorkflow()
            pipe, clf_type, _rep, classes, _cm, extras = wf.fit_and_evaluate(
                X=texts, y=labels, features=features,
                C=float(C.value), max_iter=int(max_iter.value),
                balanced=bool(balanced.value),
                test_size=float(test_size.value), random_state=42,
                options=options,
                log_cb=progress.log,
                progress_cb=lambda p, s: progress.update(0.1 + 0.85 * float(p), s),
            )

            progress.update(0.97, "Сохранение модели…")
            tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="brt_train_"))
            model_path = tmp_dir / "model.joblib"
            bundle = {
                "artifact_type": "train_model_bundle",
                "schema_version": 1,
                "pipeline": pipe,
                "config": {
                    "clf_type": clf_type,
                    "text_col": text_col.value,
                    "label_col": label_col.value,
                    "snap": {
                        "C": float(C.value),
                        "max_iter": int(max_iter.value),
                        "balanced": bool(balanced.value),
                        "test_size": float(test_size.value),
                        "use_smote": bool(use_smote.value),
                        "calib_method": calib.value,
                        "use_fuzzy_dedup": bool(fuzzy_dedup.value),
                        "fuzzy_dedup_threshold": int(fuzzy_thr.value),
                    },
                },
                "per_class_thresholds": dict(extras.get("per_class_thresholds", {})),
                "eval_metrics": {
                    "macro_f1": float(extras.get("macro_f1", 0.0)),
                    "accuracy": float(extras.get("accuracy", 0.0)),
                    "n_train": int(extras.get("n_train", 0)),
                    "n_test": int(extras.get("n_test", 0)),
                },
            }
            wf.persist_artifact(bundle, str(model_path))

            metrics_out.clear_output()
            with metrics_out:
                print(f"Тип классификатора : {clf_type}")
                print(f"macro-F1           : {bundle['eval_metrics']['macro_f1']:.3f}")
                print(f"accuracy           : {bundle['eval_metrics']['accuracy']:.3f}")
                print(f"train / test rows  : "
                      f"{bundle['eval_metrics']['n_train']} / "
                      f"{bundle['eval_metrics']['n_test']}")
                print(f"Классы ({len(classes or [])}): {list(classes or [])[:10]}"
                      + ("…" if classes and len(classes) > 10 else ""))

            download_box.children = (download_link(model_path, "📥 Скачать model.joblib"),)
            progress.mark_done()
        except Exception as exc:  # noqa: BLE001 — UI entry point
            progress.log(f"❌ {exc}")
            progress.log(traceback.format_exc())
            progress.mark_error(str(exc))
        finally:
            run_btn.disabled = False
            run_btn.description = "Обучить"

    def _on_click(_: Any) -> None:
        run_btn.disabled = True
        run_btn.description = "⏳ Обучение…"
        progress.reset("Старт…")
        metrics_out.clear_output()
        download_box.children = ()
        threading.Thread(target=_run_training, daemon=True).start()

    run_btn.on_click(_on_click)

    return w.VBox([
        title,
        w.HTML("<i>Шаг 1 — выберите файл (Upload для &lt;50&nbsp;МБ, иначе путь).</i>"),
        w.HBox([upload, shared_path]),
        w.HBox([text_col, label_col]),
        w.HTML("<i>Шаг 2 — параметры обучения.</i>"),
        w.HBox([C, max_iter]),
        w.HBox([test_size, balanced, use_smote]),
        w.HBox([calib, fuzzy_dedup, fuzzy_thr]),
        w.HTML("<i>Шаг 3 — нажмите «Обучить». После завершения появится ссылка на скачивание.</i>"),
        run_btn,
        progress.widget,
        w.HTML("<b>Метрики:</b>"),
        metrics_out,
        download_box,
    ])


# ─── helpers ────────────────────────────────────────────────────────────
def _resolve_data_path(upload_widget: Any, shared: str) -> pathlib.Path | None:
    if upload_widget.value:
        return save_upload_to_tmp(upload_widget)[0]
    shared = (shared or "").strip()
    if shared:
        p = pathlib.Path(shared).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Файл не найден: {p}")
        return p
    return None


def _read_tabular(path: pathlib.Path, text_col: str, label_col: str):
    import pandas as pd
    fmt = detect_tabular_format(path)
    df = pd.read_excel(path) if fmt == "xlsx" else pd.read_csv(path)
    for c in (text_col, label_col):
        if c not in df.columns:
            raise KeyError(f"В файле нет колонки {c!r}. Есть: {list(df.columns)}")
    texts = df[text_col].astype(str).fillna("").tolist()
    labels = df[label_col].astype(str).fillna("").tolist()
    filt = [(t, l) for t, l in zip(texts, labels) if t.strip() and l.strip()]
    if not filt:
        raise ValueError("После фильтрации пустых строк не осталось данных.")
    return [t for t, _ in filt], [l for _, l in filt]
