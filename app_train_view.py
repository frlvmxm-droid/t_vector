# -*- coding: utf-8 -*-
"""View-слой для вкладки обучения (widgets/layout)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui_theme import ENTRY_BG, FG, ACCENT
from ui_widgets_tk import Tooltip


def build_train_files_card(app, parent: tk.Widget) -> None:
    """Секция выбора и управления обучающими файлами."""
    card = ttk.Frame(parent, style="Card.TFrame", padding=12)
    card.pack(fill="x", pady=(0, 10))
    ttk.Label(card, text="Файлы Excel для обучения (можно несколько):").grid(row=0, column=0, sticky="w")
    app.lb_train = tk.Listbox(
        card, height=4,
        bg=ENTRY_BG, fg=FG,
        selectbackground=ACCENT, selectforeground="#ffffff",
        relief="flat", borderwidth=0, activestyle="none",
        font=("Segoe UI", 10),
    )
    app.lb_train.grid(row=0, column=1, sticky="we", padx=10)
    btns = ttk.Frame(card)
    btns.grid(row=0, column=2, sticky="e")
    btn_row = ttk.Frame(btns)
    btn_row.pack(fill="x")
    btn_add = ttk.Button(btn_row, text="Добавить…", command=app.add_train_files)
    btn_add.pack(side="left", fill="x", expand=True)
    btn_recent = ttk.Button(btn_row, text="▾", width=2,
                            command=lambda: app._show_recents_menu(
                                btn_recent, "train_files", app._add_train_file_from_path))
    btn_recent.pack(side="left", padx=(2, 0))
    Tooltip(btn_add,    "Добавить Excel/CSV файлы в список обучающей выборки.\nМожно выбрать несколько файлов сразу.")
    Tooltip(btn_recent, "Недавние файлы обучения — добавить без диалога.")
    btn_del = ttk.Button(btns, text="Удалить", command=app.remove_train_file)
    btn_del.pack(fill="x", pady=(6, 0))
    Tooltip(btn_del, "Удалить выделенный файл из списка обучающей выборки.")
    btn_clr = ttk.Button(btns, text="Очистить", command=app.clear_train_files)
    btn_clr.pack(fill="x", pady=(6, 0))
    Tooltip(btn_clr, "Очистить весь список файлов обучающей выборки.")
    btn_stats = ttk.Button(btns, text="📊 Статистика", style="Secondary.TButton", command=app.show_dataset_stats)
    btn_stats.pack(fill="x", pady=(6, 0))
    Tooltip(btn_stats, "Показывает в логе статистику датасета:\n"
                       "кол-во строк, классов, распределение меток,\n"
                       "топ-5 крупнейших и мельчайших классов, % дисбаланса.")
    card.columnconfigure(1, weight=1)
