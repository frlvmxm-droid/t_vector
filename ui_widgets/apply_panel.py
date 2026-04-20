# -*- coding: utf-8 -*-
"""Voilà widget: 'Применение' panel — model.joblib + data → predictions.xlsx."""
from __future__ import annotations

import pathlib
import tempfile
import threading
import traceback
from typing import Any

from ui_widgets.io import detect_tabular_format, download_link, save_upload_to_tmp
from ui_widgets.progress import ProgressPanel
from ui_widgets.theme import metric_card, metric_row, section_card


def build_apply_panel() -> Any:
    """Builds the 'Применение' tab. Returns an ipywidgets container."""
    import ipywidgets as w

    # ── Model ──────────────────────────────────────────────────────────
    model_upload = w.FileUpload(
        accept=".joblib", multiple=False,
        description="Загрузить model.joblib",
        layout=w.Layout(width="260px"),
    )
    model_path_txt = w.Text(
        value="", placeholder="или путь к model.joblib на сервере",
        description="Путь к модели:",
        layout=w.Layout(width="440px"),
    )

    # ── Data ───────────────────────────────────────────────────────────
    data_upload = w.FileUpload(
        accept=".xlsx,.csv", multiple=False,
        description="Данные",
        layout=w.Layout(width="260px"),
    )
    data_path_txt = w.Text(
        value="", placeholder="или путь к XLSX/CSV на сервере",
        description="Путь к данным:",
        layout=w.Layout(width="440px"),
    )
    text_col = w.Text(value="text", description="Колонка текста:",
                      layout=w.Layout(width="240px"))

    # ── Params ─────────────────────────────────────────────────────────
    default_thr = w.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05,
                                description="threshold:",
                                layout=w.Layout(width="300px"))
    thr_mode = w.Dropdown(options=["review_only", "strict"], value="review_only",
                          description="Режим порога:")
    out_format = w.Dropdown(options=["xlsx", "csv"], value="xlsx",
                            description="Формат:")

    run_btn = w.Button(description="▶  Классифицировать",
                       button_style="primary",
                       layout=w.Layout(width="220px"))

    progress = ProgressPanel(log_height="200px")
    summary_out = w.Output()
    metric_cards = w.HTML("")
    download_box = w.VBox([])

    def _run_apply() -> None:
        try:
            model_path = _resolve_path(model_upload, model_path_txt.value, ".joblib")
            data_path = _resolve_path(data_upload, data_path_txt.value, (".xlsx", ".csv"))
            if model_path is None or data_path is None:
                progress.log("❌ Нужна и модель, и данные.")
                progress.mark_error("Нет входов")
                return

            progress.update(0.05, "Загрузка модели…")
            from model_loader import load_model_artifact
            from apply_prediction_service import (
                predict_with_thresholds, validate_apply_bundle,
            )
            bundle = load_model_artifact(str(model_path))
            validate_apply_bundle(bundle)
            progress.log(f"  модель: {model_path.name}")
            progress.log(f"  macro-F1 (train): {bundle.get('eval_metrics', {}).get('macro_f1', 0.0):.3f}")

            progress.update(0.2, f"Чтение {data_path.name}…")
            texts = _read_text_col(data_path, text_col.value)
            progress.log(f"  строк для предсказания: {len(texts)}")

            progress.update(0.4, "Предсказание…")
            import numpy as np
            proba = bundle["pipeline"].predict_proba(texts)
            classes = bundle["pipeline"].classes_

            progress.update(0.75, "Применение порогов…")
            result = predict_with_thresholds(
                np.asarray(proba),
                classes=classes,
                per_class_thresholds=bundle.get("per_class_thresholds"),
                default_threshold=float(default_thr.value),
                threshold_mode=thr_mode.value,
            )

            progress.update(0.92, "Формирование результата…")
            out_path = _write_result(
                texts,
                result.labels,
                result.confidences,
                result.needs_review,
                fmt=out_format.value,
                text_col=text_col.value,
            )

            summary_out.clear_output()
            n = len(texts)
            reviewed = sum(result.needs_review)
            import numpy as _np
            confs = _np.asarray(result.confidences, dtype=float)
            high = int((confs >= 0.85).sum())
            with summary_out:
                print(f"Всего строк        : {n}")
                print(f"needs_review       : {reviewed} ({100 * reviewed / max(n,1):.1f}%)")
                print(f"classes            : {list(classes)}")
                print(f"Файл результата    : {out_path.name}")

            metric_cards.value = "<div style='display:flex; gap:12px; margin:4px 0;'>" + "".join([
                metric_card("ВСЕГО СТРОК", f"{n:,}".replace(",", " "),
                            "обработано только что"),
                metric_card("ВЫСОКАЯ УВЕРЕННОСТЬ", f"{high:,}".replace(",", " "),
                            f"{100 * high / max(n, 1):.0f}% · ≥ 0.85"),
                metric_card("ТРЕБУЕТ REVIEW", f"{reviewed:,}".replace(",", " "),
                            f"{100 * reviewed / max(n, 1):.1f}% · < {float(default_thr.value):.2f}"),
            ]) + "</div>"

            download_box.children = (download_link(out_path, f"📥 Скачать {out_path.name}"),)
            progress.mark_done()
        except Exception as exc:  # noqa: BLE001 — UI entry point
            progress.log(f"❌ {exc}")
            progress.log(traceback.format_exc())
            progress.mark_error(str(exc))
        finally:
            run_btn.disabled = False
            run_btn.description = "Применить"

    def _on_click(_: Any) -> None:
        run_btn.disabled = True
        run_btn.description = "⏳ Применение…"
        progress.reset("Старт…")
        summary_out.clear_output()
        metric_cards.value = ""
        download_box.children = ()
        threading.Thread(target=_run_apply, daemon=True).start()

    run_btn.on_click(_on_click)

    sources_card = section_card(
        "ИСТОЧНИКИ",
        [
            w.HBox([model_upload, model_path_txt]),
            w.HBox([data_upload, data_path_txt]),
            text_col,
        ],
        subtitle="Модель и входные данные для пакетной классификации.",
    )
    params_card = section_card(
        "ПОРОГ → REVIEW",
        [w.HBox([default_thr, thr_mode, out_format])],
        subtitle="Строки ниже порога уходят в needs_review или strict-reject.",
    )
    run_card = section_card(
        "РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ",
        [w.HBox([run_btn, download_box]), progress.widget, metric_cards, summary_out],
        subtitle="Запуск, прогресс и сводка по предсказаниям.",
    )
    return w.VBox([sources_card, params_card, run_card])


# ─── helpers ────────────────────────────────────────────────────────────
def _resolve_path(
    upload_widget: Any,
    shared: str,
    allowed_suffix,
) -> pathlib.Path | None:
    allowed = (allowed_suffix,) if isinstance(allowed_suffix, str) else tuple(allowed_suffix)
    if upload_widget.value:
        p = save_upload_to_tmp(upload_widget)[0]
        if p.suffix.lower() not in allowed:
            raise ValueError(f"Ожидался файл с расширением {allowed}, получен {p.suffix!r}")
        return p
    shared = (shared or "").strip()
    if shared:
        p = pathlib.Path(shared).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Файл не найден: {p}")
        if p.suffix.lower() not in allowed:
            raise ValueError(f"Ожидался файл с расширением {allowed}, получен {p.suffix!r}")
        return p
    return None


def _read_text_col(path: pathlib.Path, col: str) -> list[str]:
    import pandas as pd
    fmt = detect_tabular_format(path)
    df = pd.read_excel(path) if fmt == "xlsx" else pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"В файле нет колонки {col!r}. Есть: {list(df.columns)}")
    return df[col].astype(str).fillna("").tolist()


def _write_result(
    texts: list[str],
    labels: list[str],
    confidences: list[float],
    needs_review: list[int],
    *,
    fmt: str,
    text_col: str,
) -> pathlib.Path:
    import pandas as pd
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="brt_apply_"))
    df = pd.DataFrame({
        text_col: texts,
        "predicted_label": labels,
        "confidence": confidences,
        "needs_review": needs_review,
    })
    if fmt == "xlsx":
        out = tmp / "predictions.xlsx"
        df.to_excel(out, index=False)
    else:
        out = tmp / "predictions.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
    return out
