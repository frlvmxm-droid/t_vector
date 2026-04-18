# -*- coding: utf-8 -*-
"""
app_apply_view_ctk.py — view-слой вкладки «Классификация» на CustomTkinter.

Подключение в app.py:
    from app_apply_view_ctk import build_apply_tab
    build_apply_tab(self, parent_frame)

Ожидаемые методы/атрибуты у `app`:
    app.model_file        : tk.StringVar
    app.apply_file        : tk.StringVar
    app.pick_model()      : выбор файла модели
    app.pick_apply_file() : выбор файла классификации
    app.start_apply()     : запуск пакетной классификации
    app.export_predictions(): экспорт результатов
"""
from __future__ import annotations

import customtkinter as ctk

from ui_theme_ctk import (
    COLORS,
    font_label, font_sm, font_base, font_md_bold, font_lg_bold, font_mono,
)
from app_train_view_ctk import Card, Pill, Metric, _field_label, _separator


# Sample data (заменить на реальные данные из app.predictions_df)
SAMPLE_PREDICTIONS = [
    (102841, "Не приходит код подтверждения для входа в приложение",          "Мобильное прил.",     0.91, False),
    (102842, "Хочу узнать комиссию за перевод в Сбер по СБП на 250 тыс",      "Перевод СБП",         0.88, False),
    (102843, "Карта заблокирована после оплаты в магазине, разблокируйте",    "Карты — блокировка",  0.96, False),
    (102844, "Подскажите по реструктуризации кредита, потерял работу",        "Кредит — реструктур", 0.84, False),
    (102845, "не работает приложение пишет ошибку 503 уже второй день",       "Мобильное прил.",     0.79, False),
    (102846, "хочу закрыть счёт и забрать остаток наличными в отделении",     "Прочее",              0.42, True),
    (102847, "когда придут деньги по возврату товара озон, уже неделю жду",   "Прочее",              0.38, True),
    (102848, "выпустите цифровую карту мир для перевода зп",                  "Карты — выпуск",      0.93, False),
    (102849, "досрочное погашение по ипотеке, сколько процентов сэкономлю",   "Ипотека — заявка",    0.71, False),
    (102850, "оператор был груб, отказал в обслуживании, прошу разобраться",  "Прочее",              0.33, True),
]


def build_apply_tab(app, parent: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
    scroll = ctk.CTkScrollableFrame(parent, fg_color=COLORS["bg"])
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
    model_entry = ctk.CTkEntry(model_row, font=font_mono(),
                               placeholder_text="model/baseline.joblib")
    model_entry.insert(0, "model/baseline_v3_854.joblib")
    model_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    ctk.CTkButton(model_row, text="Выбрать…", width=90,
                  command=getattr(app, "pick_model", lambda: None)).pack(side="left")
    pills = ctk.CTkFrame(model_col, fg_color="transparent")
    pills.pack(fill="x", pady=(8, 0))
    Pill(pills, "trust-check OK", "success").pack(side="left", padx=(0, 4))
    Pill(pills, "F1=0.854 · 10 классов", "accent").pack(side="left", padx=(0, 4))
    Pill(pills, "26 588 train rows").pack(side="left")

    # Файл
    file_col = ctk.CTkFrame(src_grid, fg_color="transparent")
    file_col.grid(row=0, column=1, sticky="ew", padx=(6, 0))
    _field_label(file_col, "Excel / CSV для классификации")
    file_row = ctk.CTkFrame(file_col, fg_color="transparent")
    file_row.pack(fill="x")
    file_entry = ctk.CTkEntry(file_row, font=font_mono(),
                              placeholder_text="apply.xlsx")
    file_entry.insert(0, "входящие_2025_w14.xlsx")
    file_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    ctk.CTkButton(file_row, text="Выбрать…", width=90,
                  command=getattr(app, "pick_apply_file", lambda: None)).pack(side="left")
    pills2 = ctk.CTkFrame(file_col, fg_color="transparent")
    pills2.pack(fill="x", pady=(8, 0))
    Pill(pills2, "8 421 строк").pack(side="left", padx=(0, 4))
    Pill(pills2, "9 колонок").pack(side="left", padx=(0, 4))
    Pill(pills2, "UTF-8", "warning").pack(side="left")

    _separator(src_card.body)

    # Колонки + порог
    params_grid = ctk.CTkFrame(src_card.body, fg_color="transparent")
    params_grid.pack(fill="x")
    params_grid.grid_columnconfigure((0, 1, 2), weight=1, uniform="p3")

    # Колонка с описанием
    c0 = ctk.CTkFrame(params_grid, fg_color="transparent")
    c0.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    _field_label(c0, "Колонка с описанием")
    ctk.CTkOptionMenu(c0, values=["Описание обращения", "Текст звонка", "Текст чата"],
                      font=font_base()).pack(fill="x")

    # Колонка ID
    c1 = ctk.CTkFrame(params_grid, fg_color="transparent")
    c1.grid(row=0, column=1, sticky="ew", padx=6)
    _field_label(c1, "Колонка ID")
    ctk.CTkOptionMenu(c1, values=["id_обращения", "request_id", "ticket_id"],
                      font=font_base()).pack(fill="x")

    # Threshold slider
    c2 = ctk.CTkFrame(params_grid, fg_color="transparent")
    c2.grid(row=0, column=2, sticky="ew", padx=(6, 0))
    _field_label(c2, "Порог уверенности → review")
    sl_row = ctk.CTkFrame(c2, fg_color="transparent")
    sl_row.pack(fill="x")
    thr_lbl = ctk.CTkLabel(sl_row, text="0.50", font=font_mono(),
                           text_color=COLORS["accent2"], width=40)
    thr_lbl.pack(side="right")

    def _on_thr(v):
        thr_lbl.configure(text=f"{float(v):.2f}")
        app._apply_thr = float(v)

    sl = ctk.CTkSlider(sl_row, from_=0, to=1, number_of_steps=20, command=_on_thr)
    sl.set(0.5)
    sl.pack(side="left", fill="x", expand=True, padx=(0, 6))

    # ── 2. Запуск + результаты ──────────────────────────────────────
    actions = ctk.CTkFrame(None, fg_color="transparent")
    ctk.CTkButton(actions, text="📥  Экспорт .xlsx", width=130,
                  fg_color="transparent", border_width=1, border_color=COLORS["border2"],
                  command=getattr(app, "export_predictions", lambda: None)).pack(side="left", padx=(0, 6))
    ctk.CTkButton(actions, text="▶  Классифицировать", width=170, height=36,
                  fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                  text_color="#04221a" if COLORS["bg"].startswith("#0") else "#ffffff",
                  font=font_md_bold(),
                  command=getattr(app, "start_apply", lambda: None)).pack(side="left")

    res_card = Card(scroll,
                    title="Результаты классификации",
                    subtitle="входящие_2025_w14.xlsx · обработано 2 минуты назад",
                    right=actions)
    res_card.pack(fill="x", padx=20, pady=(0, 12))

    metrics_row = ctk.CTkFrame(res_card.body, fg_color="transparent")
    metrics_row.pack(fill="x")
    metrics_row.grid_columnconfigure((0, 1, 2), weight=1, uniform="m")
    Metric(metrics_row, "Всего строк", "8 421").grid(row=0, column=0, sticky="ew", padx=4)
    Metric(metrics_row, "Высокая уверенность", "5 979", "71% · ≥ 0.85").grid(row=0, column=1, sticky="ew", padx=4)
    m_review = Metric(metrics_row, "Требует review", "691", "8.2% · < 0.50")
    m_review.grid(row=0, column=2, sticky="ew", padx=4)

    # ── 3. Таблица предсказаний ────────────────────────────────────
    filter_seg = ctk.CTkSegmentedButton(
        None, values=["Все", "Высокая", "< 0.50", "⚑ Review"],
        command=lambda v: _refilter_predictions(app, v),
    )
    filter_seg.set("Все")

    pred_card = Card(
        scroll, title="Предсказания",
        subtitle=f"{len(SAMPLE_PREDICTIONS)} строк (превью)",
        right=filter_seg,
    )
    pred_card.pack(fill="x", padx=20, pady=(0, 12))

    table = ctk.CTkScrollableFrame(pred_card.body, fg_color=COLORS["entry"],
                                   corner_radius=8, height=300,
                                   border_width=1, border_color=COLORS["border2"])
    table.pack(fill="x")

    # Header
    hdr = ctk.CTkFrame(table, fg_color=COLORS["panel2"])
    hdr.pack(fill="x")
    headers = [("ID", 70), ("ТЕКСТ", 0), ("ПРЕДСКАЗАНИЕ", 180), ("УВЕРЕННОСТЬ", 130), ("ФЛАГИ", 60)]
    for i, (txt, w) in enumerate(headers):
        lbl = ctk.CTkLabel(hdr, text=txt, font=font_label(),
                           text_color=COLORS["muted"], anchor="w")
        lbl.grid(row=0, column=i, sticky="ew", padx=12, pady=8)
        if w:
            hdr.grid_columnconfigure(i, weight=0, minsize=w)
        else:
            hdr.grid_columnconfigure(i, weight=1)

    for row_data in SAMPLE_PREDICTIONS:
        _prediction_row(table, row_data)

    app._apply_pred_table = table

    return scroll


def _prediction_row(parent, row_data):
    """Одна строка с текстом, предсказанием, conf-bar, флагом."""
    rid, text, pred, conf, review = row_data

    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x")
    f.grid_columnconfigure(0, weight=0, minsize=70)
    f.grid_columnconfigure(1, weight=1)
    f.grid_columnconfigure(2, weight=0, minsize=180)
    f.grid_columnconfigure(3, weight=0, minsize=130)
    f.grid_columnconfigure(4, weight=0, minsize=60)

    ctk.CTkLabel(f, text=f"#{rid}", font=font_mono(),
                 text_color=COLORS["muted"], anchor="w").grid(
        row=0, column=0, sticky="ew", padx=12, pady=8)

    # truncated text
    truncated = text if len(text) < 70 else text[:67] + "…"
    ctk.CTkLabel(f, text=truncated, font=font_base(),
                 text_color=COLORS["fg"], anchor="w").grid(
        row=0, column=1, sticky="ew", padx=12, pady=8)

    pill_kind = "warning" if conf < 0.5 else "accent"
    Pill(f, pred, pill_kind).grid(row=0, column=2, sticky="w", padx=12, pady=8)

    # confidence bar
    conf_frame = ctk.CTkFrame(f, fg_color="transparent")
    conf_frame.grid(row=0, column=3, sticky="ew", padx=12, pady=8)
    pb = ctk.CTkProgressBar(conf_frame, width=60, height=4,
                            progress_color=(COLORS["error"] if conf < 0.5
                                            else COLORS["warning"] if conf < 0.75
                                            else COLORS["accent"]))
    pb.set(conf)
    pb.pack(side="left", padx=(0, 6))
    ctk.CTkLabel(conf_frame, text=f"{int(conf*100)}%", font=font_mono(),
                 text_color=COLORS["muted"]).pack(side="left")

    # flag
    flag_text = "⚑" if review else ""
    ctk.CTkLabel(f, text=flag_text, font=font_md_bold(),
                 text_color=COLORS["warning"]).grid(row=0, column=4, padx=12, pady=8)

    sep = ctk.CTkFrame(parent, height=1, fg_color=COLORS["border2"])
    sep.pack(fill="x")


def _refilter_predictions(app, value):
    """Заглушка — в реальном проекте подмени на app.refilter_predictions(value)."""
    print(f"→ filter predictions: {value}")
