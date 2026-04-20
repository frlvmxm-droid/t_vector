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
from ui_widgets.theme import section_card


def build_train_panel() -> Any:
    """Builds the 'Обучение' tab. Returns an ipywidgets container."""
    import ipywidgets as w

    # ── Inputs ──────────────────────────────────────────────────────────
    upload = w.FileUpload(
        accept=".xlsx,.csv", multiple=False,
        description="Загрузить XLSX/CSV",
        layout=w.Layout(width="260px"),
    )
    shared_path = w.Text(
        value="", placeholder="или путь к файлу на сервере (для больших данных)",
        description="Путь:", layout=w.Layout(width="440px"),
    )
    text_col = w.Text(value="text", description="Колонка текста:",
                      layout=w.Layout(width="240px"))
    label_col = w.Text(value="label", description="Колонка меток:",
                       layout=w.Layout(width="240px"))

    C = w.FloatLogSlider(value=1.0, base=10, min=-2, max=2, step=0.1,
                         description="C (регуляризация):",
                         style={"description_width": "initial"},
                         layout=w.Layout(width="360px"))
    max_iter = w.IntSlider(value=2000, min=200, max=10000, step=100,
                           description="max_iter:",
                           layout=w.Layout(width="360px"))
    test_size = w.FloatSlider(value=0.2, min=0.05, max=0.5, step=0.05,
                              description="test_size:",
                              layout=w.Layout(width="280px"))
    balanced = w.Checkbox(value=False, description="class_weight=balanced")
    use_smote = w.Checkbox(value=True, description="SMOTE-oversampling")
    calib = w.Dropdown(options=["sigmoid", "isotonic"], value="sigmoid",
                       description="Калибровка:")
    fuzzy_dedup = w.Checkbox(value=False, description="Fuzzy-дедуп")
    fuzzy_thr = w.IntSlider(value=92, min=60, max=100, step=1,
                            description="threshold:",
                            layout=w.Layout(width="240px"))

    # ── Auto-profile + device (parity with desktop) ─────────────────────
    auto_profile = w.Dropdown(
        options=[
            ("Smart (по умолчанию)", "smart"),
            ("Strict (для чистого датасета)", "strict"),
            ("Manual (ничего не переопределять)", "manual"),
        ],
        value="smart",
        description="Auto-profile:",
        layout=w.Layout(width="360px"),
        style={"description_width": "initial"},
    )
    sbert_device = w.Dropdown(
        options=[("Auto", "auto"), ("CPU", "cpu"),
                 ("CUDA", "cuda"), ("MPS", "mps")],
        value="auto",
        description="SBERT device:",
        layout=w.Layout(width="280px"),
        style={"description_width": "initial"},
    )

    # ── Advanced Accordion (TrainingOptions extras) ─────────────────────
    use_label_smoothing = w.Checkbox(value=False, description="Label smoothing")
    label_smoothing_eps = w.FloatSlider(
        value=0.05, min=0.0, max=0.3, step=0.01,
        description="ls-eps:",
        layout=w.Layout(width="280px"),
    )
    run_cv = w.Checkbox(value=False, description="K-fold CV (5)")
    use_hard_negatives = w.Checkbox(value=False, description="Hard negatives")
    use_field_dropout = w.Checkbox(value=False, description="Field dropout [TAG]")
    field_dropout_prob = w.FloatSlider(
        value=0.3, min=0.0, max=1.0, step=0.05,
        description="fd-prob:", layout=w.Layout(width="260px"),
    )
    field_dropout_copies = w.IntSlider(
        value=2, min=1, max=5, step=1,
        description="fd-copies:", layout=w.Layout(width="220px"),
    )
    oversample_strategy = w.Dropdown(
        options=[("cap", "cap"),
                 ("augment_light", "augment_light"),
                 ("augment_medium", "augment_medium")],
        value="cap", description="over-strat:",
        layout=w.Layout(width="260px"),
        style={"description_width": "initial"},
    )
    max_dup_per_sample = w.IntSlider(
        value=5, min=1, max=20, step=1,
        description="max-dup:", layout=w.Layout(width="220px"),
    )

    advanced_box = w.VBox([
        w.HBox([use_label_smoothing, label_smoothing_eps]),
        w.HBox([run_cv, use_hard_negatives]),
        w.HBox([use_field_dropout, field_dropout_prob, field_dropout_copies]),
        w.HBox([oversample_strategy, max_dup_per_sample]),
    ])
    advanced = w.Accordion(children=[advanced_box])
    advanced.set_title(0, "Advanced · Label smoothing / K-fold / Hard negatives / Field dropout")
    advanced.selected_index = None  # collapsed by default

    # ── Auto-profile → programmatic defaults ────────────────────────────
    _PROFILES = {
        "smart":  {"fuzzy_dedup": True,  "fuzzy_thr": 92, "calib": "sigmoid",
                   "use_smote": True,  "balanced": False, "run_cv": False,
                   "use_hard_negatives": False},
        "strict": {"fuzzy_dedup": True,  "fuzzy_thr": 95, "calib": "isotonic",
                   "use_smote": True,  "balanced": True,  "run_cv": True,
                   "use_hard_negatives": True},
    }

    def _apply_profile(*_args: Any) -> None:
        prof = _PROFILES.get(auto_profile.value)
        if prof is None:
            return  # manual — untouched
        fuzzy_dedup.value = prof["fuzzy_dedup"]
        fuzzy_thr.value = prof["fuzzy_thr"]
        calib.value = prof["calib"]
        use_smote.value = prof["use_smote"]
        balanced.value = prof["balanced"]
        run_cv.value = prof["run_cv"]
        use_hard_negatives.value = prof["use_hard_negatives"]

    auto_profile.observe(_apply_profile, names="value")
    _apply_profile()

    run_btn = w.Button(description="Обучить", button_style="primary",
                       icon="play", layout=w.Layout(width="180px"))

    # ── Outputs ─────────────────────────────────────────────────────────
    progress = ProgressPanel(log_height="220px")
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
                use_label_smoothing=bool(use_label_smoothing.value),
                label_smoothing_eps=float(label_smoothing_eps.value),
                run_cv=bool(run_cv.value),
                use_hard_negatives=bool(use_hard_negatives.value),
                use_field_dropout=bool(use_field_dropout.value),
                field_dropout_prob=float(field_dropout_prob.value),
                field_dropout_copies=int(field_dropout_copies.value),
                oversample_strategy=oversample_strategy.value,
                max_dup_per_sample=int(max_dup_per_sample.value),
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
                        "auto_profile": auto_profile.value,
                        "sbert_device": sbert_device.value,
                        "use_label_smoothing": bool(use_label_smoothing.value),
                        "label_smoothing_eps": float(label_smoothing_eps.value),
                        "run_cv": bool(run_cv.value),
                        "use_hard_negatives": bool(use_hard_negatives.value),
                        "use_field_dropout": bool(use_field_dropout.value),
                        "field_dropout_prob": float(field_dropout_prob.value),
                        "field_dropout_copies": int(field_dropout_copies.value),
                        "oversample_strategy": oversample_strategy.value,
                        "max_dup_per_sample": int(max_dup_per_sample.value),
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

    columns_accordion = w.Accordion(
        children=[w.VBox([w.HBox([text_col, label_col])])],
    )
    columns_accordion.set_title(0, "Колонки файлов")
    columns_accordion.selected_index = None  # collapsed

    sources_card = section_card(
        "ИСТОЧНИКИ",
        [w.HBox([upload, shared_path]), columns_accordion],
        subtitle="Данные для обучения (XLSX/CSV). Колонки по умолчанию: text / label.",
    )
    params_card = section_card(
        "ПАРАМЕТРЫ ОБУЧЕНИЯ",
        [
            w.HBox([auto_profile, sbert_device]),
            w.HBox([C, max_iter]),
            w.HBox([test_size, balanced, use_smote]),
            w.HBox([calib, fuzzy_dedup, fuzzy_thr]),
            advanced,
        ],
        subtitle="TF-IDF (word 1–2 + char 3–5) → LinearSVC → CalibratedClassifierCV.",
    )
    run_card = section_card(
        "ОБУЧЕНИЕ",
        [w.HBox([run_btn]), progress.widget],
        subtitle="Worker-thread; прогресс стримится в браузер.",
    )
    metrics_card = section_card(
        "МЕТРИКИ И АРТЕФАКТ",
        [metrics_out, download_box],
        subtitle="macro-F1 / accuracy / размер сплитов + скачивание model.joblib.",
    )

    return w.VBox([sources_card, params_card, run_card, metrics_card])


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
