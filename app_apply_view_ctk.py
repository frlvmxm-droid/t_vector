# -*- coding: utf-8 -*-
"""
app_apply_view_ctk.py — view-слой вкладки «Классификация» на CustomTkinter.

build_apply_tab(app, parent) создаёт все виджеты, от которых зависят handler'ы в ApplyTabMixin:
  app.apply_tree (ttk.Treeview), app.apply_log, app._chart_canvas (tk.Canvas),
  app._apply_ctx_menu (tk.Menu), app.cb_pred (ttk.Combobox),
  app._thresh_inner (ttk.Frame — хост для populate_class_thresholds),
  app.apply_progress_pb, app.btn_apply, app.btn_apply_stop.
"""
from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk

import customtkinter as ctk

from ui_theme_ctk import (
    COLORS,
    font_label, font_sm, font_base, font_md_bold, font_mono,
)
from app_train_view_ctk import Card, Pill, Metric, _field_label, _separator
from ui_widgets import Tooltip


def build_apply_tab(app, parent: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
    """
    Строит вкладку «Классификация». Все виджеты, нужные ApplyTabMixin,
    присваиваются на `app`.
    """
    scroll = ctk.CTkScrollableFrame(
        parent,
        fg_color=COLORS["bg"],
        scrollbar_fg_color=COLORS["panel"],
        scrollbar_button_color=COLORS["border"],
        scrollbar_button_hover_color=COLORS["accent3"],
    )
    scroll.pack(fill="both", expand=True)

    # ── 1. Источники: модель + файл ─────────────────────────────────
    src_card = Card(scroll, title="Источники",
                    subtitle="Модель и входные данные для пакетной классификации")
    src_card.pack(fill="x", padx=20, pady=(20, 12))

    src_grid = ctk.CTkFrame(src_card.body, fg_color="transparent")
    src_grid.pack(fill="x")
    src_grid.grid_columnconfigure((0, 1), weight=1, uniform="src")

    # Модель
    model_col = ctk.CTkFrame(src_grid, fg_color="transparent")
    model_col.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    _field_label(model_col, "Модель .joblib")
    model_row = ctk.CTkFrame(model_col, fg_color="transparent")
    model_row.pack(fill="x")
    model_entry = ctk.CTkEntry(
        model_row, font=font_mono(),
        textvariable=getattr(app, "model_file", None),
        placeholder_text="model/baseline.joblib",
    )
    model_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    ctk.CTkButton(model_row, text="Выбрать…", width=90,
                  command=getattr(app, "pick_model", lambda: None)).pack(side="left")

    # Файл для классификации
    file_col = ctk.CTkFrame(src_grid, fg_color="transparent")
    file_col.grid(row=0, column=1, sticky="ew", padx=(6, 0))
    _field_label(file_col, "Excel / CSV для классификации")
    file_row = ctk.CTkFrame(file_col, fg_color="transparent")
    file_row.pack(fill="x")
    file_entry = ctk.CTkEntry(
        file_row, font=font_mono(),
        textvariable=getattr(app, "apply_file", None),
        placeholder_text="apply.xlsx",
    )
    file_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    ctk.CTkButton(file_row, text="Выбрать…", width=90,
                  command=getattr(app, "pick_apply_file", lambda: None)).pack(side="left")

    _separator(src_card.body)

    # Параметры
    p_grid = ctk.CTkFrame(src_card.body, fg_color="transparent")
    p_grid.pack(fill="x")
    p_grid.grid_columnconfigure(1, weight=1)
    p_grid.grid_columnconfigure(3, weight=1)

    # Колонка для предсказания (защищена от _refresh_combobox_values)
    ctk.CTkLabel(p_grid, text="Колонка pred:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=0, sticky="e", padx=(0, 4), pady=3)
    app.cb_pred = ttk.Combobox(
        p_grid, textvariable=getattr(app, "pred_col", None),
        state="normal", width=30,
        values=["pred_marker1", "pred_label", "pred_reason"],
    )
    app.cb_pred.grid(row=0, column=1, sticky="ew", padx=(0, 16), pady=3)


    # ── 2. Per-class пороги ─────────────────────────────────────────
    thr_card = Card(scroll, title="Per-class пороги (тюнинг без переобучения)")
    thr_card.pack(fill="x", padx=20, pady=(0, 12))

    thr_hdr = ctk.CTkFrame(thr_card.body, fg_color="transparent")
    thr_hdr.pack(fill="x", pady=(0, 6))
    ctk.CTkLabel(thr_hdr, text="Порог для каждого класса (0.0 – 1.0)",
                 font=font_sm(), text_color=COLORS["muted"]).pack(side="left")
    ctk.CTkButton(
        thr_hdr, text="Авто-оптимизация", width=140,
        fg_color="transparent", border_width=1, border_color=COLORS["border2"],
        font=font_sm(),
        command=getattr(app, "_load_thresholds_from_model", lambda: None),
    ).pack(side="right")
    Tooltip(thr_hdr.winfo_children()[-1],
            "Загрузить per-class пороги, сохранённые в модели при обучении.\n"
            "Требует загруженной модели (.joblib).")

    # Canvas+ttk.Frame — required by populate_class_thresholds() which calls
    # winfo_children() and expects plain ttk rows (not CTk internals).
    _thr_host = ctk.CTkFrame(
        thr_card.body, fg_color=COLORS["panel2"], corner_radius=6,
        border_width=1, border_color=COLORS["border2"], height=160,
    )
    _thr_host.pack(fill="x")
    _thr_host.pack_propagate(False)
    _thr_canvas = tk.Canvas(_thr_host, bg=COLORS["panel2"], highlightthickness=0)
    _thr_sb = ttk.Scrollbar(_thr_host, orient="vertical", command=_thr_canvas.yview)
    _thr_canvas.configure(yscrollcommand=_thr_sb.set)
    _thr_sb.pack(side="right", fill="y")
    _thr_canvas.pack(side="left", fill="both", expand=True)
    app._thresh_inner = ttk.Frame(_thr_canvas)
    app._thresh_canvas_win = _thr_canvas.create_window((0, 0), window=app._thresh_inner, anchor="nw")
    app._thresh_inner.bind(
        "<Configure>",
        lambda e, c=_thr_canvas: c.configure(scrollregion=c.bbox("all")),
    )
    _thr_canvas.bind(
        "<Configure>",
        lambda e, c=_thr_canvas, w=app._thresh_canvas_win: c.itemconfig(w, width=e.width),
    )
    app._thresh_canvas = _thr_canvas
    app._thresh_class_rows = []

    # ── 3. Запуск + прогресс ────────────────────────────────────────
    run_card = Card(scroll, title="Запуск классификации")
    run_card.pack(fill="x", padx=20, pady=(0, 12))

    actions_row = ctk.CTkFrame(run_card.body, fg_color="transparent")
    actions_row.pack(fill="x", pady=(0, 8))
    app.btn_apply = ctk.CTkButton(
        actions_row, text="▶  Классифицировать", width=200,
        fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
        font=font_md_bold(),
        command=getattr(app, "run_apply", lambda: None),
    )
    app.btn_apply.pack(side="left", padx=(0, 8))
    Tooltip(app.btn_apply,
            "F5 или Ctrl+Enter — запустить пакетную классификацию.\n"
            "Результат сохраняется в Excel рядом с исходным файлом.")
    app.btn_apply_stop = ctk.CTkButton(
        actions_row, text="⏹ Стоп", width=90,
        fg_color="transparent", border_width=1, border_color=COLORS["border2"],
        state="disabled",
        command=getattr(app, "_request_cancel", lambda: None),
    )
    app.btn_apply_stop.pack(side="left", padx=(0, 16))
    Tooltip(app.btn_apply_stop, "Остановить классификацию после текущего батча.")
    _btn_export = ctk.CTkButton(
        actions_row, text="📥 Экспорт .xlsx", width=140,
        fg_color="transparent", border_width=1, border_color=COLORS["border2"],
        command=getattr(app, "export_predictions", lambda: None),
    )
    _btn_export.pack(side="left", padx=(0, 8))
    Tooltip(_btn_export, "Сохранить результаты последней классификации в выбранный файл Excel.")

    from constants import CLASS_DIR
    _btn_open_dir = ctk.CTkButton(
        actions_row, text="📂 Открыть папку", width=140,
        fg_color="transparent", border_width=1, border_color=COLORS["border2"],
        command=lambda: getattr(app, "_open_directory", lambda _: None)(CLASS_DIR),
    )
    _btn_open_dir.pack(side="left")
    Tooltip(_btn_open_dir, f"Открыть папку с файлами классификации:\n{CLASS_DIR}")

    ph_row = ctk.CTkFrame(run_card.body, fg_color="transparent")
    ph_row.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(ph_row, textvariable=getattr(app, "apply_status", None),
                 font=font_mono(), text_color=COLORS["accent2"], anchor="w").pack(
        side="left", fill="x", expand=True)
    ctk.CTkLabel(ph_row, textvariable=getattr(app, "apply_pct", None),
                 font=font_mono(), text_color=COLORS["muted"]).pack(side="right")

    app.apply_progress_pb = ctk.CTkProgressBar(run_card.body, height=8,
                                                progress_color=COLORS["accent"])
    app.apply_progress_pb.set(0)
    app.apply_progress_pb.pack(fill="x", pady=(0, 8))

    _apply_progress_var = getattr(app, "apply_progress", None)
    if _apply_progress_var is not None:
        def _update_apply_pb(*_):
            try:
                app.apply_progress_pb.set(
                    min(1.0, _apply_progress_var.get() / 100.0))
            except Exception:
                pass
        _apply_progress_var.trace_add("write", _update_apply_pb)

    app.apply_log = ctk.CTkTextbox(
        run_card.body, height=160, font=font_mono(),
        fg_color=COLORS["bg"], text_color=COLORS["muted"],
        wrap="none", state="disabled",
    )
    app.apply_log.pack(fill="x")

    # ── 3.5. Ансамбль (2-я модель) ───────────────────────────────────
    ens_card = Card(scroll, title="Ансамбль (2-я модель)",
                    subtitle="Усредняет предсказания двух моделей")
    ens_card.pack(fill="x", padx=20, pady=(0, 12))

    ens_sw_row = ctk.CTkFrame(ens_card.body, fg_color="transparent")
    ens_sw_row.pack(fill="x", pady=(0, 6))
    ctk.CTkSwitch(
        ens_sw_row, text="Включить ансамбль",
        variable=getattr(app, "use_ensemble", None),
        font=font_sm(), text_color=COLORS["fg"],
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    ).pack(side="left")

    ens_file_row = ctk.CTkFrame(ens_card.body, fg_color="transparent")
    ens_file_row.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(ens_file_row, text="Модель 2:", font=font_sm(),
                 text_color=COLORS["muted"], width=80).pack(side="left")
    ctk.CTkEntry(
        ens_file_row, font=font_mono(),
        textvariable=getattr(app, "ensemble_model2", None),
        placeholder_text="model2.joblib",
    ).pack(side="left", fill="x", expand=True, padx=(0, 6))
    ctk.CTkButton(ens_file_row, text="Выбрать…", width=90,
                  command=getattr(app, "pick_ensemble_model", lambda: None)).pack(side="left")

    ens_w_row = ctk.CTkFrame(ens_card.body, fg_color="transparent")
    ens_w_row.pack(fill="x")
    ctk.CTkLabel(ens_w_row, text="Вес модели 1:", font=font_sm(),
                 text_color=COLORS["muted"], width=110).pack(side="left")
    _ens_w1_var = getattr(app, "ensemble_w1", None)
    _ens_disp = tk.StringVar(value=f"{_ens_w1_var.get():.2f}" if _ens_w1_var else "0.50")
    _ens_slider = ctk.CTkSlider(
        ens_w_row, from_=0.1, to=0.9,
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    )
    if _ens_w1_var is not None:
        _ens_slider.set(_ens_w1_var.get())
        def _on_ens_w(val, v=_ens_w1_var, dv=_ens_disp):
            v.set(round(val, 2)); dv.set(f"{val:.2f}")
        _ens_slider.configure(command=_on_ens_w)
    _ens_slider.pack(side="left", fill="x", expand=True, padx=(6, 6))
    ctk.CTkLabel(ens_w_row, textvariable=_ens_disp, width=36,
                 font=font_mono(), text_color=COLORS["fg"]).pack(side="left")

    # ── 3.6. LLM-переранжирование ────────────────────────────────────
    llm_card = Card(scroll, title="LLM-переранжирование",
                    subtitle="Отправляет неуверенные предсказания на верификацию LLM")
    llm_card.pack(fill="x", padx=20, pady=(0, 12))

    llm_sw_row = ctk.CTkFrame(llm_card.body, fg_color="transparent")
    llm_sw_row.pack(fill="x", pady=(0, 8))
    ctk.CTkSwitch(
        llm_sw_row, text="Включить LLM-переранжирование",
        variable=getattr(app, "use_llm_rerank", None),
        font=font_sm(), text_color=COLORS["fg"],
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    ).pack(side="left")
    Tooltip(llm_sw_row.winfo_children()[-1],
            "Предсказания с уверенностью [low, high] будут перепроверены через LLM.")

    llm_params = ctk.CTkFrame(llm_card.body, fg_color="transparent")
    llm_params.pack(fill="x")
    llm_params.grid_columnconfigure((1, 3), weight=1)
    ctk.CTkLabel(llm_params, text="Top-K:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=0, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(llm_params, width=60, font=font_base(),
                 textvariable=getattr(app, "llm_rerank_top_k", None)).grid(
        row=0, column=1, sticky="w", padx=(0, 16), pady=3)
    ctk.CTkLabel(llm_params, text="Порог low:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=2, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(llm_params, width=60, font=font_base(),
                 textvariable=getattr(app, "llm_rerank_low", None)).grid(
        row=0, column=3, sticky="w", pady=3)
    ctk.CTkLabel(llm_params, text="Порог high:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=1, column=0, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(llm_params, width=60, font=font_base(),
                 textvariable=getattr(app, "llm_rerank_high", None)).grid(
        row=1, column=1, sticky="w", padx=(0, 16), pady=3)

    # ── 3.7. Метки и неопределённость ────────────────────────────────
    flags_card = Card(scroll, title="Метки и неопределённость")
    flags_card.pack(fill="x", padx=20, pady=(0, 12))

    _sw_other = ctk.CTkSwitch(
        flags_card.body, text="Метка «Другое» (ниже порога)",
        variable=getattr(app, "use_other_label", None),
        font=font_sm(), text_color=COLORS["fg"],
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    )
    _sw_other.pack(anchor="w")
    Tooltip(_sw_other,
            "Тексты с max_proba < порога получают метку 'Другое'.\n"
            "Порог задаётся слайдером ниже.")
    # sub-row: other_label_threshold
    _othr_row = ctk.CTkFrame(flags_card.body, fg_color="transparent")
    _othr_row.pack(fill="x", padx=(28, 0), pady=(0, 6))
    ctk.CTkLabel(_othr_row, text="Порог:", font=font_sm(),
                 text_color=COLORS["muted"], width=52).pack(side="left")
    _othr_var = getattr(app, "other_label_threshold", None)
    _othr_disp = tk.StringVar(value=f"{_othr_var.get():.2f}" if _othr_var else "0.50")
    def _on_othr(val, _dv=_othr_disp): _dv.set(f"{float(val):.2f}")
    ctk.CTkSlider(_othr_row, from_=0.0, to=1.0, variable=_othr_var,
                  command=_on_othr, height=18,
                  progress_color=COLORS["accent"],
                  button_color=COLORS["accent2"]).pack(side="left", fill="x", expand=True, padx=4)
    ctk.CTkLabel(_othr_row, textvariable=_othr_disp, width=36,
                 font=font_sm(), text_color=COLORS["fg"]).pack(side="left")

    _sw_amb = ctk.CTkSwitch(
        flags_card.body, text="Детектор неоднозначности",
        variable=getattr(app, "use_ambiguity_detector", None),
        font=font_sm(), text_color=COLORS["fg"],
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    )
    _sw_amb.pack(anchor="w")
    Tooltip(_sw_amb,
            "Помечает тексты, где разница между top-1 и top-2 вероятностью < epsilon.\n"
            "Добавляет колонку ambiguous=True в результат.")
    # sub-row: ambiguity_epsilon
    _amb_row = ctk.CTkFrame(flags_card.body, fg_color="transparent")
    _amb_row.pack(fill="x", padx=(28, 0), pady=(0, 4))
    ctk.CTkLabel(_amb_row, text="Epsilon:", font=font_sm(),
                 text_color=COLORS["muted"], width=52).pack(side="left")
    _amb_var = getattr(app, "ambiguity_epsilon", None)
    _amb_disp = tk.StringVar(value=f"{_amb_var.get():.2f}" if _amb_var else "0.07")
    def _on_amb(val, _dv=_amb_disp): _dv.set(f"{float(val):.2f}")
    ctk.CTkSlider(_amb_row, from_=0.0, to=0.30, variable=_amb_var,
                  command=_on_amb, height=18,
                  progress_color=COLORS["accent"],
                  button_color=COLORS["accent2"]).pack(side="left", fill="x", expand=True, padx=4)
    ctk.CTkLabel(_amb_row, textvariable=_amb_disp, width=36,
                 font=font_sm(), text_color=COLORS["fg"]).pack(side="left")

    # ── 4. Bar-chart результатов ─────────────────────────────────────
    res_card = Card(scroll, title="Результаты классификации")
    res_card.pack(fill="x", padx=20, pady=(0, 20))

    # Bar-chart (заполняется после run_apply через _draw_result_chart)
    app._chart_canvas = tk.Canvas(res_card.body, bg=COLORS["panel2"],
                                  highlightthickness=0, bd=0, height=0)
    app._chart_canvas.pack(fill="x")

    # apply_tree — hidden stub; populated by ApplyTabMixin for export / copy
    _stub_frame = tk.Frame(scroll)  # orphan, not packed
    app.apply_tree = ttk.Treeview(
        _stub_frame,
        columns=("label", "count", "pct", "avg_proba", "examples"),
        show="headings",
    )
    for cid, heading, width in [
        ("label",     "Описание проблемы",  280),
        ("count",     "Кол-во обращений",   130),
        ("pct",       "% попадания",         90),
        ("avg_proba", "Ср. уверенность",     90),
        ("examples",  "Примеры текстов",    700),
    ]:
        app.apply_tree.heading(
            cid, text=heading, anchor="w",
            command=lambda c=cid: getattr(app, "_sort_apply_tree", lambda _: None)(c),
        )
        app.apply_tree.column(cid, width=width, anchor="w")

    # Контекстное меню (работает через правую кнопку мыши в любом месте вкладки)
    app._apply_ctx_menu = tk.Menu(scroll, tearoff=0)
    app._apply_ctx_menu.add_command(
        label="Копировать строку",
        command=getattr(app, "_copy_apply_row", lambda: None),
    )
    app._apply_ctx_menu.add_command(
        label="Копировать всю таблицу",
        command=getattr(app, "_copy_apply_all", lambda: None),
    )
    app.apply_tree.bind("<Button-3>",
                        getattr(app, "_on_apply_tree_rclick", lambda e: None))

    return scroll
