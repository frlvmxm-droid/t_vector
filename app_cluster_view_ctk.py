# -*- coding: utf-8 -*-
"""
app_cluster_view_ctk.py — view-слой вкладки «Кластеризация» на CustomTkinter.

build_cluster_tab(app, parent) создаёт все виджеты, от которых зависят handler'ы:
  app.lb_cluster, app.cluster_log, app.btn_cluster, app.btn_cluster_stop,
  app.cluster_progress_pb, app.cb_sbert_clust_model, app.cb_sbert_clust_device,
  app.cb_umap_metric, app.cb_llm_provider, app._sbert_clust_widgets.
"""
from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk

import customtkinter as ctk

from ui_theme_ctk import (
    COLORS,
    font_label, font_sm, font_base, font_md_bold, font_lg_bold, font_mono,
)
from app_train_view_ctk import Card, Pill, _field_label, _separator
from ui_widgets import Tooltip


# Демо-данные для scatter-визуализации
CLUSTER_SEEDS = [
    {"id": 0, "name": "Не приходит SMS / push",        "color": "#f97316", "cx": 0.30, "cy": 0.32, "r": 0.10, "n": 412, "q": "high"},
    {"id": 1, "name": "Блокировка после оплаты",       "color": "#fb923c", "cx": 0.62, "cy": 0.28, "r": 0.09, "n": 338, "q": "high"},
    {"id": 2, "name": "Перевод СБП — лимит превышен",  "color": "#fbbf24", "cx": 0.78, "cy": 0.55, "r": 0.08, "n": 287, "q": "medium"},
    {"id": 3, "name": "Реструктуризация кредита",      "color": "#a78bfa", "cx": 0.20, "cy": 0.62, "r": 0.10, "n": 256, "q": "high"},
    {"id": 4, "name": "Курс валюты — претензия",       "color": "#60a5fa", "cx": 0.45, "cy": 0.78, "r": 0.07, "n": 198, "q": "medium"},
    {"id": 5, "name": "Возврат товара / chargeback",   "color": "#34d399", "cx": 0.70, "cy": 0.80, "r": 0.08, "n": 176, "q": "medium"},
    {"id": 6, "name": "Жалоба на оператора",           "color": "#f87171", "cx": 0.36, "cy": 0.18, "r": 0.06, "n": 121, "q": "low"},
    {"id": 7, "name": "Открытие депозита",             "color": "#e879f9", "cx": 0.85, "cy": 0.18, "r": 0.06, "n":  98, "q": "high"},
]

CLUSTER_KEYWORDS = {
    0: ["смс", "код", "не приходит", "push", "вход"],
    1: ["карта", "заблокировали", "оплата", "магазин", "разблокировать"],
    2: ["сбп", "перевод", "лимит", "превышен", "не проходит"],
    3: ["кредит", "реструктуризация", "снижение", "просрочка", "каникулы"],
    4: ["курс", "валюта", "комиссия", "обмен", "доллар"],
    5: ["возврат", "товар", "озон", "не пришли", "опротестовать"],
    6: ["оператор", "грубость", "жалоба", "отказали", "хамство"],
    7: ["депозит", "вклад", "открыть", "процент", "ставка"],
}


def build_cluster_tab(app, parent: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
    """
    Строит вкладку «Кластеризация». Все виджеты, нужные ClusterTabMixin,
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

    app._cluster_selected = 0

    # ── 1. Файлы для кластеризации ──────────────────────────────────
    files_card = Card(scroll, title="Файлы для кластеризации",
                      subtitle="Добавьте Excel / CSV с неразмеченными текстами")
    files_card.pack(fill="x", padx=20, pady=(20, 12))

    btns = ctk.CTkFrame(files_card.body, fg_color="transparent")
    btns.pack(fill="x", pady=(0, 6))
    _btn_cf = ctk.CTkButton(btns, text="+ Файл(ы)…", width=110,
                            command=getattr(app, "add_cluster_files", lambda: None))
    _btn_cf.pack(side="left", padx=(0, 6))
    Tooltip(_btn_cf, "Добавить один или несколько Excel/CSV-файлов для кластеризации.")
    _btn_cfold = ctk.CTkButton(btns, text="📁 Папка…", width=110,
                               command=getattr(app, "add_cluster_folder", lambda: None))
    _btn_cfold.pack(side="left", padx=(0, 6))
    Tooltip(_btn_cfold, "Рекурсивно добавить все .xlsx/.csv из выбранной папки.")
    _btn_auto = ctk.CTkButton(btns, text="✨ Определить параметры", width=200,
                              fg_color="transparent", border_width=1,
                              border_color=COLORS["border2"],
                              command=getattr(app, "_auto_detect_cluster_params", lambda: None))
    _btn_auto.pack(side="left")
    Tooltip(_btn_auto, "Автоматически подобрать оптимальные параметры кластеризации\n"
                       "на основе размера датасета, длины текстов и разнообразия.")

    lb_host = ctk.CTkFrame(files_card.body, fg_color=COLORS["entry"],
                           corner_radius=8, border_width=1, border_color=COLORS["border2"])
    lb_host.pack(fill="x", pady=(0, 4))
    app.lb_cluster = tk.Listbox(
        lb_host, bg=COLORS["entry"], fg=COLORS["fg"],
        selectbackground=COLORS["select"], selectforeground=COLORS["fg"],
        height=4, selectmode="extended", bd=0, highlightthickness=0,
        relief="flat", font=("Consolas", 10),
    )
    _lb_sb = tk.Scrollbar(lb_host, command=app.lb_cluster.yview, bg=COLORS["panel"])
    app.lb_cluster.configure(yscrollcommand=_lb_sb.set)
    app.lb_cluster.pack(side="left", fill="both", expand=True, padx=4, pady=4)
    _lb_sb.pack(side="right", fill="y", pady=4)

    # ── 2. Конфигурация ─────────────────────────────────────────────
    cfg_row = ctk.CTkFrame(scroll, fg_color="transparent")
    cfg_row.pack(fill="x", padx=20, pady=(0, 12))
    cfg_row.grid_columnconfigure((0, 1), weight=1, uniform="cfg")

    # Алгоритм
    algo_card = Card(cfg_row, title="Алгоритм")
    algo_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

    _cluster_algo_var = getattr(app, "cluster_algo", None)
    for val, txt in [
        ("kmeans",       "MiniBatchKMeans · быстрый baseline"),
        ("hdbscan",      "HDBSCAN · плотностной, авто-K"),
        ("hierarchical", "Иерархическая · 2 уровня"),
        ("bertopic",     "BERTopic-подобный pipeline"),
        ("lda",          "LDA · тематическое моделирование"),
    ]:
        ctk.CTkRadioButton(algo_card.body, text=txt, value=val,
                           variable=_cluster_algo_var,
                           font=font_base()).pack(anchor="w", pady=2)

    _separator(algo_card.body)

    k_row = ctk.CTkFrame(algo_card.body, fg_color="transparent")
    k_row.pack(fill="x")
    _field_label(k_row, "K (кластеров)")
    k_inner = ctk.CTkFrame(k_row, fg_color="transparent")
    k_inner.pack(fill="x")
    ctk.CTkEntry(k_inner, width=70, font=font_base(),
                 textvariable=getattr(app, "k_clusters", None)).pack(
        side="left", padx=(0, 12))
    ctk.CTkCheckBox(k_inner, text="Авто-подбор K (метод локтя)",
                    font=font_base(),
                    variable=getattr(app, "use_elbow", None)).pack(side="left")

    # Векторизация + SBERT (защищённые комбобоксы)
    vec_card = Card(cfg_row, title="Векторизация")
    vec_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    _cluster_vec_var = getattr(app, "cluster_vec_mode", None)
    for val, txt in [
        ("tfidf",    "TF-IDF"),
        ("sbert",    "SBERT эмбеддинги"),
        ("combo",    "Комбо · TF-IDF + SBERT"),
        ("ensemble", "Ансамбль · TF-IDF + 2× SBERT"),
    ]:
        ctk.CTkRadioButton(vec_card.body, text=txt, value=val,
                           variable=_cluster_vec_var,
                           font=font_base()).pack(anchor="w", pady=2)

    _separator(vec_card.body)

    vec_cfg = ctk.CTkFrame(vec_card.body, fg_color="transparent")
    vec_cfg.pack(fill="x")
    vec_cfg.grid_columnconfigure(1, weight=1)

    # SBERT model (защищённый)
    ctk.CTkLabel(vec_cfg, text="SBERT модель:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=0, sticky="e", padx=(0, 4), pady=3)
    app.cb_sbert_clust_model = ttk.Combobox(
        vec_cfg, textvariable=getattr(app, "sbert_model2", None),
        state="normal", width=28,
        values=_sbert_models_list(),
    )
    app.cb_sbert_clust_model.grid(row=0, column=1, sticky="ew", pady=3)

    # SBERT device (защищённый)
    ctk.CTkLabel(vec_cfg, text="SBERT device:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=1, column=0, sticky="e", padx=(0, 4), pady=3)
    app.cb_sbert_clust_device = ttk.Combobox(
        vec_cfg, textvariable=getattr(app, "sbert_device", None),
        state="readonly", width=28,
        values=getattr(app, "gpu_device_values", ["auto", "cpu", "cuda"]),
    )
    app.cb_sbert_clust_device.grid(row=1, column=1, sticky="ew", pady=3)

    # UMAP metric (защищённый)
    ctk.CTkLabel(vec_cfg, text="UMAP metric:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=2, column=0, sticky="e", padx=(0, 4), pady=3)
    app.cb_umap_metric = ttk.Combobox(
        vec_cfg, textvariable=getattr(app, "umap_metric", None),
        state="readonly", width=28,
        values=["cosine", "euclidean", "manhattan", "correlation"],
    )
    app.cb_umap_metric.grid(row=2, column=1, sticky="ew", pady=3)

    app._sbert_clust_widgets = [app.cb_sbert_clust_model, app.cb_sbert_clust_device]

    # ── 3. Выходные колонки ─────────────────────────────────────────
    out_card = Card(scroll, title="Выходные колонки")
    out_card.pack(fill="x", padx=20, pady=(0, 12))

    out_grid = ctk.CTkFrame(out_card.body, fg_color="transparent")
    out_grid.pack(fill="x")
    out_grid.grid_columnconfigure((1, 3), weight=1)

    ctk.CTkLabel(out_grid, text="ID кластера:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=0, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(out_grid, font=font_base(),
                 textvariable=getattr(app, "cluster_id_col", None)).grid(
        row=0, column=1, sticky="ew", padx=(0, 16), pady=3)

    ctk.CTkLabel(out_grid, text="Ключевые слова:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=2, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(out_grid, font=font_base(),
                 textvariable=getattr(app, "cluster_kw_col", None)).grid(
        row=0, column=3, sticky="ew", pady=3)

    # LLM провайдер (защищённый)
    ctk.CTkLabel(out_grid, text="LLM провайдер:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=1, column=0, sticky="e", padx=(0, 4), pady=3)
    app.cb_llm_provider = ttk.Combobox(
        out_grid, textvariable=getattr(app, "llm_provider", None),
        state="readonly", width=20,
        values=["anthropic", "openai", "gigachat", "qwen", "ollama"],
    )
    app.cb_llm_provider.grid(row=1, column=1, sticky="ew", padx=(0, 16), pady=3)
    app.cb_llm_provider.bind(
        "<<ComboboxSelected>>",
        getattr(app, "_on_llm_provider_changed", lambda e: None),
    )

    # ── 3.2. UMAP / PCA карточка ────────────────────────────────────
    umap_card = Card(scroll, title="Снижение размерности (UMAP / PCA)",
                     subtitle="Понижает размерность эмбеддингов перед кластеризацией")
    umap_card.pack(fill="x", padx=20, pady=(0, 12))

    umap_sw_row = ctk.CTkFrame(umap_card.body, fg_color="transparent")
    umap_sw_row.pack(fill="x", pady=(0, 8))
    ctk.CTkSwitch(
        umap_sw_row, text="Включить UMAP",
        variable=getattr(app, "use_umap", None),
        font=font_sm(), text_color=COLORS["fg"],
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    ).pack(side="left", padx=(0, 24))
    Tooltip(umap_sw_row.winfo_children()[-1],
            "UMAP снижает размерность до umap_n_components для KMeans/HDBSCAN.\n"
            "Ускоряет кластеризацию и улучшает разделимость для SBERT-эмбеддингов.")
    ctk.CTkSwitch(
        umap_sw_row, text="PCA перед UMAP",
        variable=getattr(app, "use_pca_before_umap", None),
        font=font_sm(), text_color=COLORS["fg"],
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    ).pack(side="left")
    Tooltip(umap_sw_row.winfo_children()[-1],
            "Применяет PCA → pca_n_components перед UMAP.\nУскоряет UMAP на больших датасетах.")

    umap_grid = ctk.CTkFrame(umap_card.body, fg_color="transparent")
    umap_grid.pack(fill="x")
    umap_grid.grid_columnconfigure((1, 3), weight=1)
    _umap_fields = [
        ("Компоненты UMAP:", "umap_n_components", 0, 0),
        ("Соседи (n_neighbors):", "umap_n_neighbors", 0, 2),
        ("Компоненты PCA:", "pca_n_components", 1, 0),
    ]
    for _lbl, _attr, _r, _c in _umap_fields:
        ctk.CTkLabel(umap_grid, text=_lbl, font=font_sm(),
                     text_color=COLORS["muted"], anchor="e").grid(
            row=_r, column=_c, sticky="e", padx=(0, 4), pady=3)
        ctk.CTkEntry(umap_grid, width=80, font=font_base(),
                     textvariable=getattr(app, _attr, None)).grid(
            row=_r, column=_c + 1, sticky="w", padx=(0, 16), pady=3)

    umap_dist_row = ctk.CTkFrame(umap_card.body, fg_color="transparent")
    umap_dist_row.pack(fill="x", pady=(4, 0))
    ctk.CTkLabel(umap_dist_row, text="min_dist:", font=font_sm(),
                 text_color=COLORS["muted"], width=80).pack(side="left")
    _umap_md = getattr(app, "umap_min_dist", None)
    _umap_md_disp = tk.StringVar(value=f"{_umap_md.get():.2f}" if _umap_md else "0.10")
    _umap_md_slider = ctk.CTkSlider(
        umap_dist_row, from_=0.0, to=0.99,
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    )
    if _umap_md is not None:
        _umap_md_slider.set(_umap_md.get())
        def _on_umap_md(val, v=_umap_md, dv=_umap_md_disp):
            v.set(round(val, 2)); dv.set(f"{val:.2f}")
        _umap_md_slider.configure(command=_on_umap_md)
    _umap_md_slider.pack(side="left", fill="x", expand=True, padx=(6, 6))
    ctk.CTkLabel(umap_dist_row, textvariable=_umap_md_disp, width=36,
                 font=font_mono(), text_color=COLORS["fg"]).pack(side="left")

    # ── 3.3. HDBSCAN детальные параметры ────────────────────────────
    hdbscan_card = Card(scroll, title="Параметры HDBSCAN",
                        subtitle="Отображается при выборе алгоритма HDBSCAN")
    # будет показана/скрыта через _on_cluster_algo_change

    hdbscan_grid = ctk.CTkFrame(hdbscan_card.body, fg_color="transparent")
    hdbscan_grid.pack(fill="x")
    hdbscan_grid.grid_columnconfigure((1, 3), weight=1)
    _hdb_fields = [
        ("min_cluster_size:", "hdbscan_min_cluster_size", 0, 0),
        ("min_samples (0=авто):", "hdbscan_min_samples", 0, 2),
    ]
    for _lbl, _attr, _r, _c in _hdb_fields:
        ctk.CTkLabel(hdbscan_grid, text=_lbl, font=font_sm(),
                     text_color=COLORS["muted"], anchor="e").grid(
            row=_r, column=_c, sticky="e", padx=(0, 4), pady=3)
        ctk.CTkEntry(hdbscan_grid, width=80, font=font_base(),
                     textvariable=getattr(app, _attr, None)).grid(
            row=_r, column=_c + 1, sticky="w", padx=(0, 16), pady=3)

    hdbscan_eps_row = ctk.CTkFrame(hdbscan_card.body, fg_color="transparent")
    hdbscan_eps_row.pack(fill="x", pady=(4, 4))
    ctk.CTkLabel(hdbscan_eps_row, text="epsilon (0=выкл):", font=font_sm(),
                 text_color=COLORS["muted"], width=140).pack(side="left")
    _hdb_eps = getattr(app, "hdbscan_eps", None)
    _hdb_eps_disp = tk.StringVar(value=f"{_hdb_eps.get():.2f}" if _hdb_eps else "0.00")
    _hdb_eps_slider = ctk.CTkSlider(
        hdbscan_eps_row, from_=0.0, to=0.5,
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    )
    if _hdb_eps is not None:
        _hdb_eps_slider.set(_hdb_eps.get())
        def _on_hdb_eps(val, v=_hdb_eps, dv=_hdb_eps_disp):
            v.set(round(val, 2)); dv.set(f"{val:.2f}")
        _hdb_eps_slider.configure(command=_on_hdb_eps)
    _hdb_eps_slider.pack(side="left", fill="x", expand=True, padx=(6, 6))
    ctk.CTkLabel(hdbscan_eps_row, textvariable=_hdb_eps_disp, width=36,
                 font=font_mono(), text_color=COLORS["fg"]).pack(side="left")

    ctk.CTkSwitch(
        hdbscan_card.body, text="Повторная кластеризация шума (recluster_noise)",
        variable=getattr(app, "recluster_noise", None),
        font=font_sm(), text_color=COLORS["fg"],
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    ).pack(anchor="w", pady=(4, 0))

    # ── 3.4. BERTopic параметры (условные) ──────────────────────────
    bertopic_card = Card(scroll, title="Параметры BERTopic",
                         subtitle="Отображается при выборе алгоритма BERTopic")

    bertopic_grid = ctk.CTkFrame(bertopic_card.body, fg_color="transparent")
    bertopic_grid.pack(fill="x")
    bertopic_grid.grid_columnconfigure((1, 3), weight=1)
    ctk.CTkLabel(bertopic_grid, text="nr_topics (авто/число):", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=0, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(bertopic_grid, width=90, font=font_base(),
                 textvariable=getattr(app, "bertopic_nr_topics", None)).grid(
        row=0, column=1, sticky="w", padx=(0, 16), pady=3)
    ctk.CTkLabel(bertopic_grid, text="min_topic_size:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=2, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(bertopic_grid, width=90, font=font_base(),
                 textvariable=getattr(app, "bertopic_min_topic_size", None)).grid(
        row=0, column=3, sticky="w", pady=3)

    # ── 3.5. LDA параметры (условные) ───────────────────────────────
    lda_card = Card(scroll, title="Параметры LDA",
                    subtitle="Отображается при выборе алгоритма LDA")

    lda_grid = ctk.CTkFrame(lda_card.body, fg_color="transparent")
    lda_grid.pack(fill="x")
    lda_grid.grid_columnconfigure((1, 3), weight=1)
    ctk.CTkLabel(lda_grid, text="Кол-во тем:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=0, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(lda_grid, width=80, font=font_base(),
                 textvariable=getattr(app, "lda_n_topics", None)).grid(
        row=0, column=1, sticky="w", padx=(0, 16), pady=3)
    ctk.CTkLabel(lda_grid, text="max_iter:", font=font_sm(),
                 text_color=COLORS["muted"], anchor="e").grid(
        row=0, column=2, sticky="e", padx=(0, 4), pady=3)
    ctk.CTkEntry(lda_grid, width=80, font=font_base(),
                 textvariable=getattr(app, "lda_max_iter", None)).grid(
        row=0, column=3, sticky="w", pady=3)

    # Показывать условные карточки при изменении algo
    _cond_cards = {
        "hdbscan":   hdbscan_card,
        "bertopic":  bertopic_card,
        "lda":       lda_card,
    }

    def _on_cluster_algo_change(*_):
        algo = getattr(app, "cluster_algo", None)
        val = algo.get() if algo else "kmeans"
        for _key, _card in _cond_cards.items():
            if val == _key:
                _card.pack(fill="x", padx=20, pady=(0, 12))
            else:
                _card.pack_forget()

    _cluster_algo_var2 = getattr(app, "cluster_algo", None)
    if _cluster_algo_var2 is not None:
        _cluster_algo_var2.trace_add("write", _on_cluster_algo_change)
        _on_cluster_algo_change()  # инициализация

    # ── 3.6. Слияние похожих кластеров ──────────────────────────────
    merge_card = Card(scroll, title="Слияние похожих кластеров",
                      subtitle="Объединяет кластеры с cosine-сходством выше порога")
    merge_card.pack(fill="x", padx=20, pady=(0, 12))

    merge_sw_row = ctk.CTkFrame(merge_card.body, fg_color="transparent")
    merge_sw_row.pack(fill="x", pady=(0, 6))
    ctk.CTkSwitch(
        merge_sw_row, text="Включить слияние",
        variable=getattr(app, "merge_similar_clusters", None),
        font=font_sm(), text_color=COLORS["fg"],
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    ).pack(side="left")

    merge_thr_row = ctk.CTkFrame(merge_card.body, fg_color="transparent")
    merge_thr_row.pack(fill="x")
    ctk.CTkLabel(merge_thr_row, text="Порог слияния:", font=font_sm(),
                 text_color=COLORS["muted"], width=120).pack(side="left")
    _merge_thr = getattr(app, "merge_threshold", None)
    _merge_thr_disp = tk.StringVar(value=f"{_merge_thr.get():.2f}" if _merge_thr else "0.85")
    _merge_slider = ctk.CTkSlider(
        merge_thr_row, from_=0.5, to=0.99,
        progress_color=COLORS["accent"], button_color=COLORS["accent2"],
    )
    if _merge_thr is not None:
        _merge_slider.set(_merge_thr.get())
        def _on_merge(val, v=_merge_thr, dv=_merge_thr_disp):
            v.set(round(val, 2)); dv.set(f"{val:.2f}")
        _merge_slider.configure(command=_on_merge)
    _merge_slider.pack(side="left", fill="x", expand=True, padx=(6, 6))
    ctk.CTkLabel(merge_thr_row, textvariable=_merge_thr_disp, width=36,
                 font=font_mono(), text_color=COLORS["fg"]).pack(side="left")

    # ── 4. Запуск + прогресс + лог ──────────────────────────────────
    run_card = Card(scroll, title="Запуск кластеризации")
    run_card.pack(fill="x", padx=20, pady=(0, 12))

    actions_row = ctk.CTkFrame(run_card.body, fg_color="transparent")
    actions_row.pack(fill="x", pady=(0, 8))
    app.btn_cluster = ctk.CTkButton(
        actions_row, text="▶  Запустить кластеризацию", width=230,
        fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
        font=font_md_bold(),
        command=getattr(app, "run_cluster", lambda: None),
    )
    app.btn_cluster.pack(side="left", padx=(0, 8))
    Tooltip(app.btn_cluster,
            "F5 или Ctrl+Enter — запустить кластеризацию текстов.\n"
            "Результат экспортируется в Excel в папку clusters/.")
    app.btn_cluster_stop = ctk.CTkButton(
        actions_row, text="⏹ Стоп", width=90,
        fg_color="transparent", border_width=1, border_color=COLORS["border2"],
        state="disabled",
        command=getattr(app, "_request_cancel", lambda: None),
    )
    app.btn_cluster_stop.pack(side="left", padx=(0, 16))
    Tooltip(app.btn_cluster_stop, "Остановить кластеризацию после текущего шага.")
    _btn_cexp = ctk.CTkButton(
        actions_row, text="📥 Экспорт .xlsx", width=140,
        fg_color="transparent", border_width=1, border_color=COLORS["border2"],
        command=getattr(app, "export_cluster_results", lambda: None),
    )
    _btn_cexp.pack(side="left")
    Tooltip(_btn_cexp, "Сохранить результаты последней кластеризации\nв выбранную папку.")

    ph_row = ctk.CTkFrame(run_card.body, fg_color="transparent")
    ph_row.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(ph_row, textvariable=getattr(app, "cluster_status", None),
                 font=font_mono(), text_color=COLORS["accent2"], anchor="w").pack(
        side="left", fill="x", expand=True)
    ctk.CTkLabel(ph_row, textvariable=getattr(app, "cluster_pct", None),
                 font=font_mono(), text_color=COLORS["muted"]).pack(side="right")

    app.cluster_progress_pb = ctk.CTkProgressBar(run_card.body, height=8,
                                                   progress_color=COLORS["accent"])
    app.cluster_progress_pb.set(0)
    app.cluster_progress_pb.pack(fill="x", pady=(0, 8))

    _cluster_progress_var = getattr(app, "cluster_progress", None)
    if _cluster_progress_var is not None:
        def _update_cluster_pb(*_):
            try:
                app.cluster_progress_pb.set(
                    min(1.0, _cluster_progress_var.get() / 100.0))
            except Exception:
                pass
        _cluster_progress_var.trace_add("write", _update_cluster_pb)

    app.cluster_log = ctk.CTkTextbox(
        run_card.body, height=180, font=font_mono(),
        fg_color=COLORS["bg"], text_color=COLORS["muted"],
        wrap="none", state="disabled",
    )
    app.cluster_log.pack(fill="x")

    # ── 5. Визуализация (scatter + легенда) ─────────────────────────
    viz_card = Card(scroll, title="Визуализация кластеров",
                    subtitle="UMAP 2D · демо-данные (обновится после запуска)")
    viz_card.pack(fill="x", padx=20, pady=(0, 20))

    viz_split = ctk.CTkFrame(viz_card.body, fg_color="transparent")
    viz_split.pack(fill="x")
    viz_split.grid_columnconfigure(0, weight=1)
    viz_split.grid_columnconfigure(1, weight=0, minsize=300)

    scatter_holder = ctk.CTkFrame(viz_split, fg_color=COLORS["entry"],
                                  corner_radius=8,
                                  border_width=1, border_color=COLORS["border2"],
                                  height=380)
    scatter_holder.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    scatter_holder.grid_propagate(False)
    _build_scatter(app, scatter_holder)

    legend = ctk.CTkScrollableFrame(
        viz_split, fg_color=COLORS["panel"],
        corner_radius=8, height=380,
        border_width=1, border_color=COLORS["border2"],
        scrollbar_fg_color=COLORS["panel"],
        scrollbar_button_color=COLORS["border"],
        scrollbar_button_hover_color=COLORS["accent3"],
    )
    legend.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    ctk.CTkLabel(legend, text="КЛАСТЕРЫ", font=font_label(),
                 text_color=COLORS["accent2"]).pack(anchor="w", padx=10, pady=(8, 4))

    app._cluster_legend_items = {}
    for seed in CLUSTER_SEEDS:
        _legend_item(app, legend, seed)

    return scroll


# ─────────────────────────────────────────────────────────────────────
# Scatter (matplotlib)
# ─────────────────────────────────────────────────────────────────────

def _build_scatter(app, holder):
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import math, random
    except ImportError:
        ctk.CTkLabel(holder, text="matplotlib не установлен.\npip install matplotlib",
                     font=font_base(), text_color=COLORS["muted"]).pack(expand=True)
        return

    fig = Figure(figsize=(6, 3.8), dpi=90)
    fig.patch.set_facecolor(COLORS["entry"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(COLORS["entry"])

    rnd = random.Random(1234)
    for _ in range(50):
        ax.scatter(rnd.random(), rnd.random(), s=6, color=COLORS["muted"], alpha=0.25)

    for seed in CLUSTER_SEEDS:
        n_pts = seed["n"] // 4
        xs, ys = [], []
        for _ in range(n_pts):
            ang = rnd.random() * 2 * math.pi
            rad = math.sqrt(rnd.random()) * seed["r"]
            xs.append(seed["cx"] + math.cos(ang) * rad)
            ys.append(seed["cy"] + math.sin(ang) * rad)
        ax.scatter(xs, ys, s=12, color=seed["color"], alpha=0.7, edgecolors="none")
        ax.scatter([seed["cx"]], [seed["cy"]], s=60, color=seed["color"],
                   edgecolors=COLORS["bg"], linewidths=1.5, zorder=10)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("UMAP-1", color=COLORS["muted"], fontsize=8)
    ax.set_ylabel("UMAP-2", color=COLORS["muted"], fontsize=8)
    ax.tick_params(colors=COLORS["muted"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(COLORS["border2"])
    ax.grid(True, color=COLORS["border2"], linewidth=0.5, alpha=0.5)
    fig.tight_layout(pad=0.5)

    canvas = FigureCanvasTkAgg(fig, master=holder)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)


def _legend_item(app, parent, seed):
    f = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=6)
    f.pack(fill="x", padx=8, pady=2)

    def on_click(_=None):
        app._cluster_selected = seed["id"]
        for sid, w in app._cluster_legend_items.items():
            w.configure(fg_color=(COLORS["select"] if sid == seed["id"]
                                  else "transparent"))

    f.bind("<Button-1>", on_click)
    inner = ctk.CTkFrame(f, fg_color="transparent")
    inner.pack(fill="x", padx=8, pady=6)
    inner.bind("<Button-1>", on_click)

    dot = ctk.CTkLabel(inner, text="●", font=font_md_bold(), text_color=seed["color"])
    dot.pack(side="left", padx=(0, 6))
    dot.bind("<Button-1>", on_click)

    name = ctk.CTkLabel(inner, text=seed["name"], font=font_sm(),
                        text_color=COLORS["fg"], anchor="w")
    name.pack(side="left", fill="x", expand=True)
    name.bind("<Button-1>", on_click)

    ctk.CTkLabel(inner, text=str(seed["n"]), font=font_mono(),
                 text_color=COLORS["muted"]).pack(side="right", padx=(6, 0))

    qual = "✓" if seed["q"] == "high" else "◐" if seed["q"] == "medium" else "✕"
    qcolor = (COLORS["success"] if seed["q"] == "high"
              else COLORS["warning"] if seed["q"] == "medium" else COLORS["error"])
    ctk.CTkLabel(inner, text=qual, font=font_md_bold(), text_color=qcolor).pack(
        side="right", padx=(6, 0))

    app._cluster_legend_items[seed["id"]] = f


def _sbert_models_list() -> list:
    try:
        from config.user_config import SBERT_MODELS_LIST
        return SBERT_MODELS_LIST
    except Exception:
        return ["ai-forever/ru-en-RoSBERTa", "ai-forever/rubert-tiny2"]
