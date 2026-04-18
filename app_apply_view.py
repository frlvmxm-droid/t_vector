# -*- coding: utf-8 -*-
"""View-слой для вкладки классификации."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui_widgets import Tooltip


def build_apply_files_card(app, parent) -> None:
    card = ttk.Frame(parent, style="Card.TFrame", padding=12)
    card.pack(fill="x", pady=(0, 10))
    ttk.Label(card, text="Модель (.joblib):", style="Card.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Entry(card, textvariable=app.model_file, width=85).grid(row=0, column=1, sticky="we", padx=10)
    btn_frame_model = ttk.Frame(card)
    btn_frame_model.grid(row=0, column=2)
    btn_model = ttk.Button(btn_frame_model, text="Выбрать…", command=app.pick_model)
    btn_model.pack(side="left")
    btn_model_rec = ttk.Button(btn_frame_model, text="▾", width=2,
                               command=lambda: app._show_recents_menu(
                                   btn_model_rec, "model_file", app._pick_model_from_path))
    btn_model_rec.pack(side="left", padx=(2, 0))
    Tooltip(btn_model,     "Выбрать файл обученной модели (.joblib).\nМодель должна быть обучена этим же приложением.")
    Tooltip(btn_model_rec, "Недавние модели — открыть без диалога.")

    ttk.Label(card, text="Excel для классификации:", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(card, textvariable=app.apply_file, width=85).grid(row=1, column=1, sticky="we", padx=10, pady=(8, 0))
    btn_frame_excel = ttk.Frame(card)
    btn_frame_excel.grid(row=1, column=2, pady=(8, 0))
    btn_excel = ttk.Button(btn_frame_excel, text="Выбрать…", command=app.pick_apply_file)
    btn_excel.pack(side="left")
    btn_excel_rec = ttk.Button(btn_frame_excel, text="▾", width=2,
                               command=lambda: app._show_recents_menu(
                                   btn_excel_rec, "apply_file", app._pick_apply_from_path))
    btn_excel_rec.pack(side="left", padx=(2, 0))
    Tooltip(btn_excel,     "Выбрать Excel или CSV файл для классификации.\nЗаголовки колонок будут подставлены в выпадающие списки ниже.")
    Tooltip(btn_excel_rec, "Недавние файлы классификации — открыть без диалога.")
    card.columnconfigure(1, weight=1)

