# -*- coding: utf-8 -*-
"""
app_train_view_ctk.py — переписанный view-слой вкладки «Обучение» на CustomTkinter.

Сравни со старым app_train_view.py — структура та же, но виджеты CTk.
Бизнес-логика (`app.add_train_files`, `app.lb_train`, и т.д.) трогается минимально:
view только создаёт виджеты и вешает существующие команды.

Подключение в app.py:
    from app_train_view_ctk import build_train_tab
    train_tab = build_train_tab(self, parent_frame)
"""
from __future__ import annotations

import customtkinter as ctk
from ui_theme_ctk import (
    COLORS,
    font_label, font_sm, font_base, font_md_bold, font_lg_bold, font_mono,
)


# ─────────────────────────────────────────────────────────────────────
# Переиспользуемые блоки (в духе компонентов из прототипа)
# ─────────────────────────────────────────────────────────────────────

class Card(ctk.CTkFrame):
    """Аналог .card из прототипа: панель + опциональный заголовок."""
    def __init__(self, parent, title: str | None = None, subtitle: str | None = None,
                 right=None, **kwargs):
        super().__init__(parent, corner_radius=10, fg_color=COLORS["panel"], **kwargs)
        if title:
            header = ctk.CTkFrame(self, fg_color="transparent")
            header.pack(fill="x", padx=18, pady=(14, 8))
            left = ctk.CTkFrame(header, fg_color="transparent")
            left.pack(side="left")
            ctk.CTkLabel(left, text=title.upper(), font=font_label(),
                         text_color=COLORS["accent2"]).pack(anchor="w")
            if subtitle:
                ctk.CTkLabel(left, text=subtitle, font=font_sm(),
                             text_color=COLORS["muted"]).pack(anchor="w", pady=(2, 0))
            if right is not None:
                right.pack(in_=header, side="right")
        # внутренний контейнер для содержимого (паддинг)
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=18, pady=(0, 14))


def Pill(parent, text: str, kind: str = "default") -> ctk.CTkLabel:
    """Маленький бейдж как .pill из прототипа."""
    fg_map = {
        "default": (COLORS["panel2"], COLORS["muted"]),
        "accent":  (COLORS["select"], COLORS["accent2"]),
        "success": (COLORS["panel2"], COLORS["success"]),
        "warning": (COLORS["panel2"], COLORS["warning"]),
        "error":   (COLORS["panel2"], COLORS["error"]),
    }
    bg, fg = fg_map.get(kind, fg_map["default"])
    return ctk.CTkLabel(parent, text=text, font=font_sm(),
                        fg_color=bg, text_color=fg,
                        corner_radius=999, padx=10, pady=2)


def Metric(parent, label: str, value: str, delta: str | None = None) -> ctk.CTkFrame:
    """Тайл с метрикой (KPI)."""
    f = ctk.CTkFrame(parent, fg_color=COLORS["panel2"], corner_radius=8,
                     border_width=1, border_color=COLORS["border2"])
    inner = ctk.CTkFrame(f, fg_color="transparent")
    inner.pack(padx=14, pady=12)
    ctk.CTkLabel(inner, text=label.upper(), font=font_label(),
                 text_color=COLORS["muted"]).pack(anchor="w")
    ctk.CTkLabel(inner, text=value, font=font_lg_bold(),
                 text_color=COLORS["fg"]).pack(anchor="w", pady=(4, 0))
    if delta:
        ctk.CTkLabel(inner, text=delta, font=font_sm(),
                     text_color=COLORS["success"]).pack(anchor="w", pady=(2, 0))
    return f


# ─────────────────────────────────────────────────────────────────────
# Сборщик вкладки «Обучение»
# ─────────────────────────────────────────────────────────────────────

def build_train_tab(app, parent: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
    """
    Строит вкладку «Обучение» внутри `parent`. `app` — экземпляр главного окна
    с методами/переменными (add_train_files, k_clusters и т.п.).
    """
    scroll = ctk.CTkScrollableFrame(parent, fg_color=COLORS["bg"])
    scroll.pack(fill="both", expand=True)

    # ── 1. Файлы обучающей выборки ──────────────────────────────────
    files_card = Card(
        scroll,
        title="Файлы обучающей выборки",
        subtitle="3 файла · 26 588 строк · 9 колонок",
    )
    files_card.pack(fill="x", padx=20, pady=(20, 12))

    btns_row = ctk.CTkFrame(files_card.body, fg_color="transparent")
    btns_row.pack(fill="x", pady=(0, 10))
    ctk.CTkButton(btns_row, text="+ Добавить…", width=120,
                  command=getattr(app, "add_train_files", lambda: None)).pack(side="left", padx=(0, 6))
    ctk.CTkButton(btns_row, text="📁 Папка…", width=110,
                  command=getattr(app, "add_train_folder", lambda: None)).pack(side="left", padx=(0, 6))
    ctk.CTkButton(btns_row, text="📊 Статистика", width=120,
                  fg_color="transparent", border_width=1, border_color=COLORS["border2"],
                  command=getattr(app, "show_dataset_stats", lambda: None)).pack(side="left")

    # «Таблица» файлов через CTkScrollableFrame со строками
    table = ctk.CTkScrollableFrame(files_card.body, fg_color=COLORS["entry"],
                                   corner_radius=8, height=140,
                                   border_width=1, border_color=COLORS["border2"])
    table.pack(fill="x")
    _table_header(table, ["", "Файл", "Строки", "Колонки", "Размер"])
    sample = [
        ("обращения_2024_q3.xlsx", "12 480", "9", "4.2 МБ"),
        ("обращения_2024_q4.xlsx", "14 102", "9", "4.8 МБ"),
        ("разметка_операторы.csv", "3 206",  "6", "0.9 МБ"),
    ]
    for row in sample:
        _table_row(table, ("📄",) + row)
    app._train_files_table = table  # ссылку наружу — для обновления

    # ── 2. Конфиг (две колонки) ─────────────────────────────────────
    config_row = ctk.CTkFrame(scroll, fg_color="transparent")
    config_row.pack(fill="x", padx=20, pady=(0, 12))
    config_row.grid_columnconfigure(0, weight=1, uniform="cfg")
    config_row.grid_columnconfigure(1, weight=1, uniform="cfg")

    # 2a. Векторизация
    vec_card = Card(config_row, title="Векторизация и базовая модель")
    vec_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

    _field_label(vec_card.body, "Базовая модель")
    base_seg = ctk.CTkSegmentedButton(
        vec_card.body, values=["С нуля", "Дообучение от .joblib"],
        command=lambda v: setattr(app, "_train_base_mode", v),
    )
    base_seg.set("С нуля")
    base_seg.pack(fill="x", pady=(0, 14))

    _field_label(vec_card.body, "Векторизатор")
    vec_var = ctk.StringVar(value="hybrid")
    for val, txt in [
        ("tfidf",    "TF-IDF (быстрый, baseline)"),
        ("sbert",    "SBERT — нейросетевые эмбеддинги"),
        ("hybrid",   "Гибрид TF-IDF + SBERT (рекомендуется)"),
        ("ensemble", "Ансамбль · TF-IDF + 2× SBERT"),
    ]:
        ctk.CTkRadioButton(vec_card.body, text=txt, value=val, variable=vec_var,
                           font=font_base()).pack(anchor="w", pady=2)
    app._train_vec_var = vec_var

    # 2b. Параметры
    params_card = Card(config_row, title="Параметры обучения")
    params_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    grid = ctk.CTkFrame(params_card.body, fg_color="transparent")
    grid.pack(fill="x")
    grid.grid_columnconfigure(0, weight=1, uniform="p")
    grid.grid_columnconfigure(1, weight=1, uniform="p")

    for i, (lbl, default) in enumerate([
        ("Test split", "0.20"),
        ("N grams TF-IDF", "1, 2"),
        ("Min class size", "20"),
        ("Random state", "42"),
    ]):
        cell = ctk.CTkFrame(grid, fg_color="transparent")
        cell.grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=4)
        _field_label(cell, lbl)
        ctk.CTkEntry(cell, font=font_base()).pack(fill="x")
        # запомнить значение по умолчанию
        cell.winfo_children()[-1].insert(0, default)

    _separator(params_card.body)

    _field_label(params_card.body, "Опции")
    for txt, default in [
        ("CalibratedClassifierCV — калибровка вероятностей", True),
        ("Стратифицированный split по классам", True),
        ("ML-аугментация миноритарных классов", False),
        ("Guardrails: проверка дисбаланса > 50:1", True),
    ]:
        cb = ctk.CTkCheckBox(params_card.body, text=txt, font=font_base())
        if default:
            cb.select()
        cb.pack(anchor="w", pady=2)

    # ── 3. Прогресс обучения ────────────────────────────────────────
    run_btn = ctk.CTkButton(
        None, text="▶  Запустить обучение", width=200, height=36,
        fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
        text_color="#04221a" if COLORS["bg"].startswith("#0") else "#ffffff",
        font=font_md_bold(),
        command=getattr(app, "start_training", lambda: None),
    )
    progress_card = Card(scroll, title="Запуск обучения", right=run_btn)
    progress_card.pack(fill="x", padx=20, pady=(0, 12))

    # KPI-метрики
    metrics_row = ctk.CTkFrame(progress_card.body, fg_color="transparent")
    metrics_row.pack(fill="x", pady=(0, 12))
    metrics_row.grid_columnconfigure((0, 1, 2), weight=1, uniform="m")
    Metric(metrics_row, "Текущая F1", "0.854", "+0.007 vs baseline").grid(row=0, column=0, sticky="ew", padx=4)
    Metric(metrics_row, "Точность macro", "0.871", "+0.012").grid(row=0, column=1, sticky="ew", padx=4)
    Metric(metrics_row, "Время", "08:42 мин").grid(row=0, column=2, sticky="ew", padx=4)

    # Прогресс-бар + статус
    stage_row = ctk.CTkFrame(progress_card.body, fg_color="transparent")
    stage_row.pack(fill="x", pady=(0, 6))
    stage_lbl = ctk.CTkLabel(stage_row,
                             text="Калибровка вероятностей · fold 3/5",
                             font=font_mono(), text_color=COLORS["accent2"])
    stage_lbl.pack(side="left")
    pct_lbl = ctk.CTkLabel(stage_row, text="67%", font=font_mono(), text_color=COLORS["muted"])
    pct_lbl.pack(side="right")
    app._train_stage_lbl = stage_lbl
    app._train_pct_lbl = pct_lbl

    pb = ctk.CTkProgressBar(progress_card.body, height=10)
    pb.set(0.67)
    pb.pack(fill="x")
    app._train_progress = pb

    # Лог (CTkTextbox)
    log = ctk.CTkTextbox(progress_card.body, height=180, font=font_mono(),
                        fg_color=COLORS["bg"], text_color=COLORS["muted"],
                        wrap="none")
    log.pack(fill="x", pady=(12, 0))
    log.insert("end",
        "14:22:01  INFO  Загружено 3 файла, 26 588 строк, 10 классов\n"
        "14:22:04  INFO  Лемматизация · pymystem3 · 24.1 сек\n"
        "14:22:28  OK    TF-IDF словарь: 84 213 признаков\n"
        "14:23:11  INFO  SBERT rubert-tiny2 · 26 588 × 312 · GPU\n"
        "14:25:42  WARN  Класс «Прочее» содержит >50% noise-токенов\n"
        "14:26:08  OK    Hybrid features · sparse 84525 dims · L2-norm\n"
        "14:28:55  INFO  CalibratedClassifierCV(method='sigmoid', cv=5) · fold 3/5\n"
    )
    log.configure(state="disabled")
    app._train_log = log

    return scroll


# ─────────────────────────────────────────────────────────────────────
# Внутренние утилиты для «таблицы» и подписей
# ─────────────────────────────────────────────────────────────────────

def _table_header(parent, cols):
    f = ctk.CTkFrame(parent, fg_color=COLORS["panel2"])
    f.pack(fill="x")
    for i, c in enumerate(cols):
        ctk.CTkLabel(f, text=c.upper(), font=font_label(),
                     text_color=COLORS["muted"], anchor="w").grid(
            row=0, column=i, sticky="ew", padx=12, pady=8)
    for i in range(len(cols)):
        f.grid_columnconfigure(i, weight=1 if i == 1 else 0)


def _table_row(parent, cells):
    f = ctk.CTkFrame(parent, fg_color="transparent",
                     border_width=0)
    f.pack(fill="x")
    for i, c in enumerate(cells):
        ctk.CTkLabel(f, text=c, font=font_base() if i != 1 else font_mono(),
                     text_color=COLORS["fg"], anchor="w").grid(
            row=0, column=i, sticky="ew", padx=12, pady=6)
    for i in range(len(cells)):
        f.grid_columnconfigure(i, weight=1 if i == 1 else 0)
    # тонкий разделитель снизу
    sep = ctk.CTkFrame(parent, height=1, fg_color=COLORS["border2"])
    sep.pack(fill="x")


def _field_label(parent, text: str):
    ctk.CTkLabel(parent, text=text.upper(), font=font_label(),
                 text_color=COLORS["muted"], anchor="w").pack(
        anchor="w", pady=(0, 4))


def _separator(parent):
    ctk.CTkFrame(parent, height=1, fg_color=COLORS["border2"]).pack(
        fill="x", pady=14)
