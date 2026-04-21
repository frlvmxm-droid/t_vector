# -*- coding: utf-8 -*-
"""View-слой для вкладки кластеризации."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui_theme import ENTRY_BG, FG, ACCENT
from ui_widgets_tk import Tooltip


def build_cluster_files_card(app, parent):
    card = ttk.Frame(parent, style="Card.TFrame", padding=12)
    card.pack(fill="x", pady=(0, 10))
    ttk.Label(card, text="Файлы для кластеризации:", style="Card.TLabel").grid(
        row=0, column=0, sticky="nw", pady=(2, 0)
    )
    app.lb_cluster = tk.Listbox(
        card, height=4,
        bg=ENTRY_BG, fg=FG,
        selectbackground=ACCENT, selectforeground="#ffffff",
        relief="flat", borderwidth=0, activestyle="none",
        font=("Segoe UI", 10),
    )
    app.lb_cluster.grid(row=0, column=1, sticky="we", padx=10)
    btns = ttk.Frame(card)
    btns.grid(row=0, column=2, sticky="ne")
    btn_files = ttk.Button(btns, text="Файл(ы)…", command=app.add_cluster_files)
    btn_files.pack(fill="x")
    Tooltip(btn_files, "Добавить Excel/CSV файлы в список кластеризации.\nМожно выбрать несколько файлов сразу.")
    btn_folder = ttk.Button(btns, text="Папка…", command=app.add_cluster_folder)
    btn_folder.pack(fill="x", pady=(6, 0))
    Tooltip(btn_folder, "Добавить все Excel/CSV файлы из выбранной папки.")
    btn_del = ttk.Button(btns, text="Удалить", command=app.remove_cluster_file, state="disabled")
    btn_del.pack(fill="x", pady=(6, 0))
    Tooltip(btn_del, "Удалить выделенный файл из списка кластеризации.")
    btn_clr = ttk.Button(btns, text="Очистить", command=app.clear_cluster_files, state="disabled")
    btn_clr.pack(fill="x", pady=(6, 0))
    Tooltip(btn_clr, "Очистить список файлов для кластеризации.")
    app._cluster_btn_del = btn_del
    app._cluster_btn_clr = btn_clr
    app._sync_cluster_file_buttons()
    card.columnconfigure(1, weight=1)
    return card


def build_cluster_basic_settings_sections(app, parent, card):
    """Базовые секции настроек кластеризации (до блока выбора алгоритма)."""
    detect_row = ttk.Frame(card)
    detect_row.grid(row=1, column=1, columnspan=2, sticky="we", padx=10, pady=(6, 0))
    btn_detect = ttk.Button(
        detect_row,
        text="🔍  Определить оптимальные параметры",
        command=app._auto_detect_cluster_params,
    )
    btn_detect.pack(side="left")
    Tooltip(
        btn_detect,
        "Быстро анализирует файлы кластеризации:\n"
        "  • размер датасета и длину текстов\n"
        "  • наличие диалоговых ролей (OPERATOR / CLIENT / CHATBOT)\n"
        "  • разнообразие и плотность текстовых столбцов\n\n"
        "По результатам автоматически устанавливает:\n"
        "  K, алгоритм, векторизацию, UMAP, n_init, метрику K и др.",
    )
    app._detect_status_var = tk.StringVar(value="")
    ttk.Label(
        detect_row,
        textvariable=app._detect_status_var,
        style="Card.Muted.TLabel",
    ).pack(side="left", padx=(12, 0))

    cols = ttk.LabelFrame(parent, text="Колонки и параметры", padding=12)
    cols.pack(fill="x", pady=(0, 6))
    app._combobox(cols, 0, "Описание:", app.desc_col, "Описание обращения")
    app._combobox(cols, 1, "Текст звонка:", app.call_col, "Транскрипт звонка")
    app._combobox(cols, 2, "Текст чата:", app.chat_col, "Транскрипт чата")

    row = ttk.Frame(cols, style="Card.TFrame")
    row.grid(row=3, column=1, sticky="w", padx=10, pady=(8, 0))
    lbl_k = ttk.Label(row, text="K (кластеров):", style="Card.TLabel")
    lbl_k.pack(side="left")
    Tooltip(lbl_k, "Число кластеров для алгоритма KMeans.\n\n"
                   "При выключённом «Авто-подбор K»:\n"
                   "  Используется ровно это число кластеров.\n\n"
                   "При включённом «Авто-подбор K» (метод локтя):\n"
                   "  K служит стартовой точкой — алгоритм ищет оптимум\n"
                   "  в диапазоне около этого значения.\n\n"
                   "Рекомендуется начать с K = 10–20 и уточнить вручную.")
    sp_k = ttk.Spinbox(row, from_=2, to=200, textvariable=app.k_clusters, width=6)
    sp_k.pack(side="left", padx=8)
    cb_elbow = ttk.Checkbutton(row, text="Авто-подбор K", variable=app.use_elbow, style="Card.TCheckbutton")
    cb_elbow.pack(side="left", padx=(14, 0))
    app.attach_help(cb_elbow, "Авто-подбор K (метод локтя)",
                    "Если включено — алгоритм автоматически выбирает оптимальное число кластеров K\n"
                    "по кривой inertia (метод локтя) в диапазоне около заданного K.\n"
                    "Это ориентировочный результат. Для финала рекомендуется проверить несколько K вручную.",
                    "Авто-подбор K по методу локтя")
    app._cluster_k_widgets = [sp_k, cb_elbow]

    role_frm = ttk.LabelFrame(parent, text="Роли диалога и контекст", padding=(12, 6))
    role_frm.pack(fill="x", pady=(0, 6))
    cb_bot = ttk.Checkbutton(
        role_frm, text="Игнорировать чат-бота / SYSTEM",
        variable=app.ignore_chatbot_cluster,
    )
    cb_bot.pack(side="left")
    app.attach_help(
        cb_bot,
        "Игнорировать чат-бота",
        "Не включать в кластеризацию реплики с метками:\n"
        "  CHATBOT / БОТ / SYSTEM / СИСТЕМА\n\n"
        "Реплики чат-бота — это стандартные технические фразы\n"
        "(«оператор подключается», «оцените работу сотрудника» и т.п.),\n"
        "которые не несут смысловой нагрузки обращения.\n\n"
        "Рекомендуется включать.",
        "Удалять реплики CHATBOT / SYSTEM из текста кластеризации",
    )

    ttk.Label(role_frm, text="  Источник текста:").pack(side="left", padx=(20, 4))
    for _text, _val, _tip in [
        ("Весь диалог", "all", "Используются все доступные поля:\n  Описание, диалог (CLIENT + OPERATOR), суммаризация, ответы."),
        ("Только клиент", "client", "Кластеризация только по репликам КЛИЕНТА + описанию."),
        ("Только оператор", "operator", "Кластеризация только по репликам ОПЕРАТОРА + ответам банка."),
    ]:
        rb = ttk.Radiobutton(role_frm, text=_text, variable=app.cluster_role_mode, value=_val)
        rb.pack(side="left", padx=4)
        app.attach_help(rb, f"Источник: {_text}", _tip, _tip)

    noise_frm = ttk.LabelFrame(parent, text="Фильтрация шума", padding=(12, 6))
    noise_frm.pack(fill="x", pady=(0, 6))
    cb_sw = ttk.Checkbutton(noise_frm, text="Стоп-слова", variable=app.use_stop_words)
    cb_sw.pack(side="left")
    app.attach_help(cb_sw, "Стоп-слова", "Исключить из TF-IDF частые служебные слова.", "Удалять русские служебные слова из TF-IDF")
    cb_nt = ttk.Checkbutton(noise_frm, text="Шумовые токены", variable=app.use_noise_tokens)
    cb_nt.pack(side="left", padx=(16, 0))
    app.attach_help(cb_nt, "Шумовые токены", "Удалять технические токены-маски и SDK-маркеры.", "Удалять технические токены-маски и SDK-маркеры")
    cb_np = ttk.Checkbutton(noise_frm, text="Шумовые фразы", variable=app.use_noise_phrases)
    cb_np.pack(side="left", padx=(16, 0))
    app.attach_help(cb_np, "Шумовые фразы", "Regex-удаление шаблонных фраз до TF-IDF.", "Regex-удаление шаблонных фраз до TF-IDF")

    vec_frm = ttk.LabelFrame(parent, text="Векторизация текстов", padding=(12, 6))
    vec_frm.pack(fill="x", pady=(0, 6))
    vec_rb_row = ttk.Frame(vec_frm, style="Card.TFrame")
    vec_rb_row.pack(fill="x")
    app._cluster_vec_rbs = []
    for text, val in [("TF-IDF", "tfidf"), ("SBERT (нейросетевые эмбеддинги)", "sbert"), ("Комбо (TF-IDF + SBERT)", "combo"), ("Ансамбль (TF-IDF + 2 SBERT)", "ensemble")]:
        rb = ttk.Radiobutton(vec_rb_row, text=text, variable=app.cluster_vec_mode, value=val)
        rb.pack(side="left", padx=6)
        app._cluster_vec_rbs.append(rb)

    app._sbert_clust_widgets = []
    app._ensemble_widgets = []
    app._combo_clust_widgets = []
    app._tfidf_svd_row = None

    kw_row = ttk.Frame(vec_frm, style="Card.TFrame")
    kw_row.pack(fill="x", pady=(4, 0))
    cb_ctfidf = ttk.Checkbutton(
        kw_row,
        text="c-TF-IDF ключевые слова (рекомендуется)",
        variable=app.use_ctfidf_keywords,
        style="Card.TCheckbutton",
    )
    cb_ctfidf.pack(side="left")
    app.attach_help(
        cb_ctfidf,
        "c-TF-IDF для ключевых слов кластеров",
        "Класс-взвешенный TF-IDF (BERTopic-алгоритм) вместо простого усреднения.",
        "c-TF-IDF: кластер-специфичные ключевые слова",
    )


def build_cluster_algo_main_section(app, parent):
    """Основной блок выбора алгоритма кластеризации."""
    algo_main_frm = ttk.LabelFrame(parent, text="Алгоритм кластеризации", padding=(12, 6))
    algo_main_frm.pack(fill="x", pady=(0, 6))

    algo_rb_row = ttk.Frame(algo_main_frm, style="Card.TFrame")
    algo_rb_row.pack(fill="x")
    app._cluster_algo_rbs = []
    for text, val, tip in [
        ("KMeans", "kmeans", "MiniBatchKMeans — быстрый классический алгоритм."),
        ("HDBSCAN", "hdbscan", "HDBSCAN — иерархическая плотностная кластеризация."),
        ("LDA (тематическое моделирование)", "lda", "Latent Dirichlet Allocation — тематическое моделирование."),
        ("Иерархическая (2 уровня)", "hierarchical", "Двухуровневая иерархическая кластеризация."),
        ("BERTopic-подобная", "bertopic", "BERTopic-style pipeline без внешней зависимости."),
        ("FASTopic (ускоренный BERTopic)", "fastopic", "Ускоренный режим темы: lightweight c-TF-IDF."),
    ]:
        rb = ttk.Radiobutton(algo_rb_row, text=text, variable=app.cluster_algo, value=val)
        rb.pack(side="left", padx=6)
        app.attach_help(rb, f"Алгоритм: {text}", tip, tip)
        app._cluster_algo_rbs.append(rb)
