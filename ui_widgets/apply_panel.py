# -*- coding: utf-8 -*-
"""Voilà widget: 'Применение' panel — model.joblib + data → predictions.xlsx."""
from __future__ import annotations

import os
import pathlib
import tempfile
import threading
import traceback
from typing import Any, Dict, List

from ui_widgets.io import detect_tabular_format, download_link, save_upload_to_tmp
from ui_widgets.predictions_table import build_predictions_preview
from ui_widgets.progress import ProgressPanel
from ui_widgets.theme import (
    badge, header_chip_row, metric_card, section_card,
)
from ui_widgets.trust_prompt import (
    TrustDenied,
    build_confirm_prompt,
    ensure_trusted_model_path_interactive,
    get_trust_store,
)


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
    inspect_btn = w.Button(
        description="Осмотреть модель", button_style="",
        layout=w.Layout(width="180px"),
    )

    # ── Ensemble (optional 2nd model) ──────────────────────────────────
    use_ensemble = w.Checkbox(value=False, description="Ансамбль (2-я модель)")
    ensemble_upload = w.FileUpload(
        accept=".joblib", multiple=False,
        description="2-я модель",
        layout=w.Layout(width="220px"),
    )
    ensemble_path_txt = w.Text(
        value="", placeholder="или путь к 2-й model.joblib",
        description="Путь #2:",
        layout=w.Layout(width="400px"),
    )
    ensemble_weight = w.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.05,
        description="вес 2-й:",
        layout=w.Layout(width="260px"),
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
    use_fallback_other = w.Checkbox(
        value=True, description="«Другое» (fallback-label)",
    )
    mark_ambiguous = w.Checkbox(
        value=True, description="Неоднозначность (top1−top2<0.10) → review",
    )
    ambiguity_thr = w.FloatSlider(
        value=0.10, min=0.01, max=0.50, step=0.01,
        description="ambig-Δ:", layout=w.Layout(width="260px"),
    )

    # ── Per-class thresholds (filled after bundle inspect) ─────────────
    per_class_box = w.VBox([])
    per_class_accordion = w.Accordion(children=[per_class_box])
    per_class_accordion.set_title(0, "Пороги по классам (заполняется после осмотра модели)")
    per_class_accordion.selected_index = None
    _per_class_sliders: Dict[str, Any] = {}

    # ── LLM-rerank ─────────────────────────────────────────────────────
    _llm_api_key = os.environ.get("BRT_LLM_API_KEY", "")
    _llm_provider = os.environ.get("BRT_LLM_PROVIDER", "offline")
    _llm_model = os.environ.get("BRT_LLM_MODEL", "")
    llm_rerank = w.Checkbox(
        value=False,
        description="LLM-rerank низко-уверенных строк",
        disabled=(not _llm_api_key and _llm_provider.lower() != "offline"),
    )
    llm_top_k = w.IntSlider(
        value=3, min=2, max=5, step=1,
        description="top-K:", layout=w.Layout(width="240px"),
    )
    llm_status = w.HTML(
        "<div style='font-size:11px;margin-left:8px;color:#6db39a'>"
        f"provider={_llm_provider} · model={_llm_model or '—'} · "
        f"api-key={'✔' if _llm_api_key else '—'}</div>"
    )

    # ── Outputs ────────────────────────────────────────────────────────
    run_btn = w.Button(description="▶  Классифицировать",
                       button_style="primary",
                       layout=w.Layout(width="220px"))
    progress = ProgressPanel(log_height="200px")
    summary_out = w.Output()
    metric_cards = w.HTML("")
    header_chips = w.HTML("")
    download_box = w.VBox([])
    preview_box = w.VBox([])

    # ── Trust prompt (shown only when a new .joblib is loaded) ─────────
    trust_host, confirm_trust = build_confirm_prompt()

    def _trusted_load(path: pathlib.Path) -> dict:
        """Wraps load_model_artifact with trust-store verification."""
        from model_loader import load_model_artifact
        ensure_trusted_model_path_interactive(
            path, log_cb=progress.log, confirm_cb=confirm_trust,
        )
        store = get_trust_store()
        return load_model_artifact(
            str(path),
            precomputed_sha256=store.get_hash(str(path)),
            require_trusted=True,
            trusted_paths=store.trusted_canonical_paths(),
        )

    # ── Bundle inspection (populates per-class + header chips) ─────────
    def _inspect_model(*_a: Any) -> None:
        try:
            model_path = _resolve_path(
                model_upload, model_path_txt.value, ".joblib",
            )
            if model_path is None:
                progress.log("ℹ️  укажите модель для осмотра")
                return
            from apply_prediction_service import validate_apply_bundle
            bundle = _trusted_load(model_path)
            validate_apply_bundle(bundle)
            _refresh_header_chips(bundle, data_path=None)
            _rebuild_per_class(bundle)
            progress.log(f"  осмотр модели: {model_path.name} ok")
        except TrustDenied as exc:
            progress.log(f"🛑 осмотр отменён: {exc}")
        except Exception as exc:  # noqa: BLE001
            progress.log(f"❌ осмотр модели: {exc}")

    def _rebuild_per_class(bundle: dict) -> None:
        import ipywidgets as wgt
        thresholds = dict(bundle.get("per_class_thresholds") or {})
        rows: List[Any] = []
        _per_class_sliders.clear()
        for cls in sorted(thresholds):
            s = wgt.FloatSlider(
                value=float(thresholds[cls]),
                min=0.0, max=1.0, step=0.01,
                description=str(cls),
                style={"description_width": "initial"},
                layout=wgt.Layout(width="420px"),
            )
            _per_class_sliders[str(cls)] = s
            rows.append(s)
        if not rows:
            rows.append(wgt.HTML(
                "<div style='color:#6db39a;font-size:11px'>"
                "в bundle нет per-class-порогов — применится default_threshold</div>"
            ))
        per_class_box.children = tuple(rows)
        per_class_accordion.set_title(
            0, f"Пороги по классам ({len(_per_class_sliders)} кл.)",
        )

    def _refresh_header_chips(bundle: dict, *, data_path: pathlib.Path | None) -> None:
        evm = bundle.get("eval_metrics", {}) or {}
        n_train = int(evm.get("n_train", 0))
        n_test = int(evm.get("n_test", 0))
        f1 = float(evm.get("macro_f1", 0.0))
        trust_ok = bool(bundle.get("artifact_type") == "train_model_bundle")
        chips = [
            badge(f"trust-check {'ok' if trust_ok else 'err'}",
                  "ok" if trust_ok else "err"),
            badge(f"macro-F1 {f1:.3f}", "info"),
            badge(f"train/test {n_train}/{n_test}", "info"),
        ]
        try:
            classes = list(bundle["pipeline"].classes_)
            chips.append(badge(f"classes {len(classes)}", "info"))
        except Exception:
            pass
        if data_path is not None:
            fmt = detect_tabular_format(data_path)
            encoding = "utf-8" if fmt == "csv" else "binary"
            chips.append(badge(f"fmt {fmt}", "info"))
            chips.append(badge(f"encoding {encoding}", "info"))
        header_chips.value = (
            "<div class='brt-header-chip-row'>"
            + "".join(chips)
            + "</div>"
        )

    inspect_btn.on_click(_inspect_model)

    # ── Main worker ────────────────────────────────────────────────────
    def _run_apply() -> None:
        try:
            model_path = _resolve_path(model_upload, model_path_txt.value, ".joblib")
            data_path = _resolve_path(data_upload, data_path_txt.value, (".xlsx", ".csv"))
            if model_path is None or data_path is None:
                progress.log("❌ Нужна и модель, и данные.")
                progress.mark_error("Нет входов")
                return

            progress.update(0.05, "Загрузка модели…")
            from apply_prediction_service import (
                predict_with_thresholds, validate_apply_bundle,
            )
            bundle = _trusted_load(model_path)
            validate_apply_bundle(bundle)
            _refresh_header_chips(bundle, data_path=data_path)
            progress.log(f"  модель: {model_path.name}")
            progress.log(f"  macro-F1 (train): {bundle.get('eval_metrics', {}).get('macro_f1', 0.0):.3f}")

            bundle2 = None
            if use_ensemble.value:
                m2_path = _resolve_path(
                    ensemble_upload, ensemble_path_txt.value, ".joblib",
                )
                if m2_path is not None:
                    bundle2 = _trusted_load(m2_path)
                    validate_apply_bundle(bundle2)
                    progress.log(f"  модель #2: {m2_path.name}")

            progress.update(0.2, f"Чтение {data_path.name}…")
            texts = _read_text_col(data_path, text_col.value)
            progress.log(f"  строк для предсказания: {len(texts)}")

            progress.update(0.4, "Предсказание…")
            import numpy as np
            proba = np.asarray(bundle["pipeline"].predict_proba(texts))
            classes = list(bundle["pipeline"].classes_)
            if bundle2 is not None:
                proba2 = np.asarray(bundle2["pipeline"].predict_proba(texts))
                classes2 = list(bundle2["pipeline"].classes_)
                if classes2 == classes:
                    w2 = float(ensemble_weight.value)
                    proba = (1.0 - w2) * proba + w2 * proba2
                    progress.log(f"  ансамбль: weight #2 = {w2:.2f}")
                else:
                    progress.log("⚠ классы ансамбль-моделей не совпали; 2-я модель проигнорирована")

            # Merge UI-edited per-class thresholds with bundle defaults
            merged_thr = dict(bundle.get("per_class_thresholds") or {})
            for cls, slider in _per_class_sliders.items():
                merged_thr[cls] = float(slider.value)

            progress.update(0.75, "Применение порогов…")
            result = predict_with_thresholds(
                proba,
                classes=classes,
                per_class_thresholds=merged_thr,
                default_threshold=float(default_thr.value),
                threshold_mode=thr_mode.value,
                fallback_label="other_label" if use_fallback_other.value else None,
                ambiguity_threshold=float(ambiguity_thr.value),
            )

            labels = list(result.labels)
            confidences = list(result.confidences)
            needs_review = list(result.needs_review)
            if mark_ambiguous.value:
                for i, amb in enumerate(result.is_ambiguous):
                    if amb and not needs_review[i]:
                        needs_review[i] = 1

            # LLM-rerank (opt-in, worker-thread; graceful skip on miss)
            if llm_rerank.value:
                _maybe_llm_rerank(
                    texts=texts, proba=proba, classes=classes,
                    labels=labels, confidences=confidences,
                    needs_review=needs_review,
                    top_k=int(llm_top_k.value),
                    default_threshold=float(default_thr.value),
                    merged_thresholds=merged_thr,
                    log=progress.log,
                )

            progress.update(0.92, "Формирование результата…")
            out_path = _write_result(
                texts, labels, confidences, needs_review,
                fmt=out_format.value, text_col=text_col.value,
            )

            summary_out.clear_output()
            n = len(texts)
            reviewed = sum(needs_review)
            confs_arr = np.asarray(confidences, dtype=float)
            high = int((confs_arr >= 0.85).sum())
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
            preview_box.children = (
                build_predictions_preview(
                    texts, labels, confidences, needs_review, limit=12,
                ),
            )
            progress.mark_done()
        except TrustDenied as exc:
            progress.log(f"🛑 загрузка отменена: {exc}")
            progress.mark_error("Отменено пользователем")
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
        preview_box.children = ()
        threading.Thread(target=_run_apply, daemon=True).start()

    run_btn.on_click(_on_click)

    sources_card = section_card(
        "ИСТОЧНИКИ",
        [
            w.HBox([model_upload, model_path_txt, inspect_btn]),
            w.HBox([use_ensemble, ensemble_upload, ensemble_path_txt, ensemble_weight]),
            w.HBox([data_upload, data_path_txt]),
            text_col,
            header_chips,
            trust_host,
        ],
        subtitle="Модель + (опц.) 2-я модель-ансамбль + входные данные.",
    )
    params_card = section_card(
        "ПОРОГ → REVIEW",
        [
            w.HBox([default_thr, thr_mode, out_format]),
            w.HBox([use_fallback_other, mark_ambiguous, ambiguity_thr]),
            per_class_accordion,
        ],
        subtitle="Пороги + «Другое» fallback + флаг неоднозначности top1−top2.",
    )
    llm_card = section_card(
        "LLM-RERANK (ОПЦИОНАЛЬНО)",
        [w.HBox([llm_rerank, llm_top_k, llm_status])],
        subtitle=(
            "Перевыставить метки на строках < threshold через LLM. "
            "Требует BRT_LLM_API_KEY + BRT_LLM_PROVIDER + BRT_LLM_MODEL."
        ),
    )
    run_card = section_card(
        "РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ",
        [
            w.HBox([run_btn, download_box]),
            progress.widget, metric_cards, summary_out, preview_box,
        ],
        subtitle="Запуск, прогресс, сводка + превью 12 строк.",
    )
    return w.VBox([sources_card, params_card, llm_card, run_card])


# ─── LLM rerank wrapper ────────────────────────────────────────────────
def _maybe_llm_rerank(
    *,
    texts: List[str],
    proba: Any,
    classes: List[str],
    labels: List[str],
    confidences: List[float],
    needs_review: List[int],
    top_k: int,
    default_threshold: float,
    merged_thresholds: dict,
    log: Any,
) -> None:
    """Edit labels/confidences in-place for rows that need review."""
    try:
        import numpy as np
        from llm_reranker import rerank_top_k
    except Exception as exc:  # noqa: BLE001
        log(f"⚠ LLM-rerank пропущен: импорт упал ({exc})")
        return
    provider = os.environ.get("BRT_LLM_PROVIDER", "offline")
    api_key = os.environ.get("BRT_LLM_API_KEY", "")
    model = os.environ.get("BRT_LLM_MODEL", "")
    if provider.lower() != "offline" and (not api_key or not model):
        log("⚠ LLM-rerank пропущен: BRT_LLM_API_KEY / BRT_LLM_MODEL не заданы")
        return

    idx_low = [i for i, r in enumerate(needs_review) if r]
    if not idx_low:
        log("ℹ LLM-rerank: все строки уверенны, rerank не нужен")
        return
    log(f"  LLM-rerank: {len(idx_low)} строк на пересчёт (provider={provider})")

    order = np.argsort(-proba[idx_low], axis=1)[:, :top_k]
    top_candidates = [[classes[j] for j in row] for row in order]
    sub_texts = [texts[i] for i in idx_low]
    sub_argmax = [labels[i] for i in idx_low]
    try:
        reranked = rerank_top_k(
            texts=sub_texts,
            top_candidates=top_candidates,
            argmax_labels=sub_argmax,
            provider=provider, model=model, api_key=api_key,
            log_fn=log,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"⚠ LLM-rerank провалился: {exc}")
        return
    n_updated = 0
    for i_sub, i_full in enumerate(idx_low):
        new = reranked[i_sub]
        if new and new != labels[i_full]:
            labels[i_full] = new
            # Re-check against merged thresholds; clear needs_review if now passes
            thr = float(merged_thresholds.get(new, default_threshold))
            if confidences[i_full] >= thr:
                needs_review[i_full] = 0
            n_updated += 1
    log(f"  LLM-rerank: обновлено {n_updated} меток")


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
