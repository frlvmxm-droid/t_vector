# -*- coding: utf-8 -*-
"""Voilà widget: 'Обучение' panel — upload → train → download model.joblib."""
from __future__ import annotations

import pathlib
import tempfile
import threading
import traceback
from collections.abc import Callable
from typing import Any

from ui_widgets.io import detect_tabular_format, download_link, save_upload_to_tmp
from ui_widgets.progress import ProgressPanel
from ui_widgets.theme import section_card

# Keys that live in ``widgets_by_key`` for session save/restore but must
# NOT enter ``bundle["config"]["snap"]`` — either they travel in the
# outer ``config`` dict (text/label columns) or they are local-only
# (filesystem path picker).
_TRAIN_SNAP_EXCLUDE: frozenset[str] = frozenset({
    "text_col", "label_col", "shared_path",
})


def build_train_panel() -> tuple[Any, dict[str, Any], Callable[[], dict[str, Any]]]:
    """Builds the 'Обучение' tab.

    Returns ``(vbox, widgets_by_key, snap_fn)`` so ``notebook_app`` can wire
    session save/restore: ``widgets_by_key`` maps ``"train.<name>"`` keys to
    the underlying widgets (``.value`` restorable), ``snap_fn`` returns the
    current snap dict for ``DebouncedSaver``.
    """
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

    # ── Preprocessing (parity with desktop train tab screenshot) ────────
    # Controls piped into make_hybrid_vectorizer(...) below: n-gram ranges,
    # min_df / max_features / sublinear_tf, Russian stop-words, noise
    # tokens / phrases, per-field TF-IDF, SVD (LSA), lemmatization, and
    # dialog meta-features. Exclusion editors live in their own accordion.
    _desc_style = {"description_width": "initial"}
    char_ng_min = w.BoundedIntText(
        value=2, min=1, max=10, description="Симв. n-граммы от:",
        layout=w.Layout(width="200px"), style=_desc_style,
    )
    char_ng_max = w.BoundedIntText(
        value=9, min=1, max=10, description="до:",
        layout=w.Layout(width="120px"), style=_desc_style,
    )
    word_ng_min = w.BoundedIntText(
        value=1, min=1, max=5, description="Слов. n-граммы от:",
        layout=w.Layout(width="200px"), style=_desc_style,
    )
    word_ng_max = w.BoundedIntText(
        value=3, min=1, max=5, description="до:",
        layout=w.Layout(width="120px"), style=_desc_style,
    )
    min_df = w.BoundedIntText(
        value=3, min=1, max=50, description="Мин. частота слова:",
        layout=w.Layout(width="220px"), style=_desc_style,
    )
    max_features = w.BoundedIntText(
        value=300_000, min=10_000, max=2_000_000, step=10_000,
        description="Макс. признаков:",
        layout=w.Layout(width="240px"), style=_desc_style,
    )
    sublinear_tf = w.Checkbox(value=True, description="Логарифм TF")

    use_stop_words = w.Checkbox(value=True, description="Стоп-слова (рус.)")
    use_noise_tokens = w.Checkbox(value=True, description="Шумовые токены")
    use_noise_phrases = w.Checkbox(value=True, description="Шумовые фразы")
    use_per_field = w.Checkbox(value=True, description="TF-IDF по полям")
    use_svd = w.Checkbox(value=True, description="SVD (LSA)")
    svd_components = w.BoundedIntText(
        value=200, min=10, max=500, description="Компоненты:",
        layout=w.Layout(width="200px"), style=_desc_style,
    )
    use_lemma = w.Checkbox(value=True, description="Лемматизация")
    use_meta = w.Checkbox(value=False, description="Мета-признаки диалога")

    # Exclusion editors — persisted to user_exclusions.json via config.exclusions.
    # Textareas pre-filled from disk; "Сохранить" button flushes edits.
    try:
        from config.exclusions import load_exclusions
        _excl = load_exclusions()
    except Exception:  # noqa: BLE001 — missing module must not block UI
        _excl = {"stop_words": [], "noise_tokens": [], "noise_phrases": []}

    excl_stop_words = w.Textarea(
        value="\n".join(_excl.get("stop_words", [])),
        placeholder="По слову в строке — добавляются к русским стоп-словам",
        layout=w.Layout(width="95%", height="110px"),
    )
    excl_noise_tokens = w.Textarea(
        value="\n".join(_excl.get("noise_tokens", [])),
        placeholder="По токену в строке — исключаются из TF-IDF",
        layout=w.Layout(width="95%", height="110px"),
    )
    excl_noise_phrases = w.Textarea(
        value="\n".join(_excl.get("noise_phrases", [])),
        placeholder="По фразе в строке — удаляются из текста до токенизации",
        layout=w.Layout(width="95%", height="110px"),
    )
    excl_save_btn = w.Button(
        description="💾 Сохранить исключения", button_style="",
        layout=w.Layout(width="240px"),
    )
    excl_reload_btn = w.Button(
        description="↻ Перечитать с диска",
        layout=w.Layout(width="200px"),
    )
    excl_status = w.HTML("")

    def _lines(text: str) -> list[str]:
        return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    def _on_save_excl(_btn: Any) -> None:
        try:
            from config.exclusions import save_exclusions
            save_exclusions({
                "stop_words":    _lines(excl_stop_words.value),
                "noise_tokens":  _lines(excl_noise_tokens.value),
                "noise_phrases": _lines(excl_noise_phrases.value),
            })
            excl_status.value = (
                f"<span style='color:#1de9b6'>Сохранено: "
                f"{len(_lines(excl_stop_words.value))} стоп-слов, "
                f"{len(_lines(excl_noise_tokens.value))} токенов, "
                f"{len(_lines(excl_noise_phrases.value))} фраз</span>"
            )
        except Exception as exc:  # noqa: BLE001
            excl_status.value = f"<span style='color:#ff5252'>Ошибка: {exc}</span>"

    def _on_reload_excl(_btn: Any) -> None:
        try:
            from config.exclusions import load_exclusions as _reload
            # bypass cache
            import config.exclusions as _mod
            _mod._cache = None
            data = _reload()
            excl_stop_words.value = "\n".join(data.get("stop_words", []))
            excl_noise_tokens.value = "\n".join(data.get("noise_tokens", []))
            excl_noise_phrases.value = "\n".join(data.get("noise_phrases", []))
            excl_status.value = "<span style='color:#1de9b6'>Перечитано с диска.</span>"
        except Exception as exc:  # noqa: BLE001
            excl_status.value = f"<span style='color:#ff5252'>Ошибка: {exc}</span>"

    excl_save_btn.on_click(_on_save_excl)
    excl_reload_btn.on_click(_on_reload_excl)

    excl_accordion = w.Accordion(children=[w.VBox([
        w.HTML("<div class='brt-field-label'>Стоп-слова (по строке)</div>"),
        excl_stop_words,
        w.HTML("<div class='brt-field-label'>Шумовые токены</div>"),
        excl_noise_tokens,
        w.HTML("<div class='brt-field-label'>Шумовые фразы</div>"),
        excl_noise_phrases,
        w.HBox([excl_save_btn, excl_reload_btn]),
        excl_status,
    ])])
    excl_accordion.set_title(0, "Редактировать слова-исключения (user_exclusions.json)")
    excl_accordion.selected_index = None  # collapsed by default

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

    # ── Session snap wiring ─────────────────────────────────────────────
    # Namespaced with "train." to avoid collisions with apply/cluster keys
    # (e.g. text_col lives in all three panels). Kept as the single source
    # of truth: both ``snap_fn`` (session save) and ``_service_snap``
    # (model-config bundle) derive from this dict.
    widgets_by_key: dict[str, Any] = {
        "train.text_col": text_col,
        "train.label_col": label_col,
        "train.C": C,
        "train.max_iter": max_iter,
        "train.test_size": test_size,
        "train.balanced": balanced,
        "train.use_smote": use_smote,
        "train.calib_method": calib,
        "train.use_fuzzy_dedup": fuzzy_dedup,
        "train.fuzzy_dedup_threshold": fuzzy_thr,
        "train.auto_profile": auto_profile,
        "train.sbert_device": sbert_device,
        "train.use_label_smoothing": use_label_smoothing,
        "train.label_smoothing_eps": label_smoothing_eps,
        "train.run_cv": run_cv,
        "train.use_hard_negatives": use_hard_negatives,
        "train.use_field_dropout": use_field_dropout,
        "train.field_dropout_prob": field_dropout_prob,
        "train.field_dropout_copies": field_dropout_copies,
        "train.oversample_strategy": oversample_strategy,
        "train.max_dup_per_sample": max_dup_per_sample,
        "train.shared_path": shared_path,
        # Preprocessing (text analysis) parity with desktop screenshot.
        "train.char_ng_min": char_ng_min,
        "train.char_ng_max": char_ng_max,
        "train.word_ng_min": word_ng_min,
        "train.word_ng_max": word_ng_max,
        "train.min_df": min_df,
        "train.max_features": max_features,
        "train.sublinear_tf": sublinear_tf,
        "train.use_stop_words": use_stop_words,
        "train.use_noise_tokens": use_noise_tokens,
        "train.use_noise_phrases": use_noise_phrases,
        "train.use_per_field": use_per_field,
        "train.use_svd": use_svd,
        "train.svd_components": svd_components,
        "train.use_lemma": use_lemma,
        "train.use_meta": use_meta,
    }

    def snap_fn() -> dict[str, Any]:
        """Full namespaced snap for session save/restore."""
        return {key: widget.value for key, widget in widgets_by_key.items()}

    def _service_snap() -> dict[str, Any]:
        """Bare-key snap for ``bundle["config"]["snap"]``.

        Excludes keys that belong elsewhere (``text_col``/``label_col``
        live in the outer ``config`` dict) or are session-only
        (``shared_path`` is a filesystem picker, not a training param).
        """
        return {
            key.split(".", 1)[1]: widget.value
            for key, widget in widgets_by_key.items()
            if key.split(".", 1)[1] not in _TRAIN_SNAP_EXCLUDE
        }

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
            from ml_vectorizers import make_hybrid_vectorizer
            # Textareas may carry unsaved edits — they take precedence over
            # the on-disk user_exclusions.json so users can try settings
            # without committing them.
            _extra_stop = _lines(excl_stop_words.value)
            _extra_tok  = _lines(excl_noise_tokens.value)
            _extra_ph   = _lines(excl_noise_phrases.value)
            # Per-field needs base_weights; without them make_hybrid_vectorizer
            # silently falls back to the legacy unified TF-IDF. Mirror the
            # desktop defaults (w_desc=2 / w_client=3 / w_operator=1).
            _base_w = (
                {"w_desc": 2, "w_client": 3, "w_operator": 1}
                if use_per_field.value else None
            )
            features = make_hybrid_vectorizer(
                char_ng=(int(char_ng_min.value), int(char_ng_max.value)),
                word_ng=(int(word_ng_min.value), int(word_ng_max.value)),
                min_df=int(min_df.value),
                max_features=int(max_features.value),
                sublinear_tf=bool(sublinear_tf.value),
                use_stop_words=bool(use_stop_words.value),
                extra_stop_words=_extra_stop or None,
                extra_noise_tokens=_extra_tok or None,
                extra_noise_phrases=_extra_ph or None,
                use_noise_tokens=bool(use_noise_tokens.value),
                use_noise_phrases=bool(use_noise_phrases.value),
                use_per_field=bool(use_per_field.value),
                base_weights=_base_w,
                use_svd=bool(use_svd.value),
                svd_components=int(svd_components.value),
                use_lemma=bool(use_lemma.value),
                use_meta=bool(use_meta.value),
            )

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
                    "snap": _service_snap(),
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
    preprocessing_card = section_card(
        "ПРЕДОБРАБОТКА ТЕКСТА",
        [
            w.HBox([char_ng_min, char_ng_max, word_ng_min, word_ng_max]),
            w.HBox([min_df, max_features, sublinear_tf]),
            w.HBox([use_stop_words, use_noise_tokens, use_noise_phrases, use_per_field]),
            w.HBox([use_svd, svd_components, use_lemma, use_meta]),
            excl_accordion,
        ],
        subtitle="n-граммы, шумовые фильтры, лемматизация, SVD (LSA), мета-признаки.",
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
        subtitle="PhraseRemover → hybrid TF-IDF → LinearSVC → CalibratedClassifierCV.",
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

    vbox = w.VBox([sources_card, preprocessing_card, params_card, run_card, metrics_card])
    return vbox, widgets_by_key, snap_fn


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
