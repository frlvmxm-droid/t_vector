# -*- coding: utf-8 -*-
"""
app_cluster_view_ctk.py — view-слой вкладки «Кластеризация» на CustomTkinter.

Подключение в app.py:
    from app_cluster_view_ctk import build_cluster_tab
    build_cluster_tab(self, parent_frame)

Scatter-плот рендерится через matplotlib (FigureCanvasTkAgg) — стилизован под
текущую палитру COLORS.

Ожидаемые методы у `app`:
    app.add_cluster_files()
    app.add_cluster_folder()
    app.start_cluster()
    app.export_cluster_results()
    app._auto_detect_cluster_params()
    app.cluster_algo, app.cluster_vec_mode, app.k_clusters  : tk.StringVar / IntVar
"""
from __future__ import annotations

import customtkinter as ctk

from ui_theme_ctk import (
    COLORS,
    font_label, font_sm, font_base, font_md_bold, font_lg_bold, font_mono,
)
from app_train_view_ctk import Card, Pill, _field_label, _separator


# Sample cluster data
CLUSTER_SEEDS = [
    {"id": 0, "name": "Не приходит SMS / push",        "color": "#1de9b6", "cx": 0.30, "cy": 0.32, "r": 0.10, "n": 412, "q": "high"},
    {"id": 1, "name": "Блокировка после оплаты",       "color": "#00c896", "cx": 0.62, "cy": 0.28, "r": 0.09, "n": 338, "q": "high"},
    {"id": 2, "name": "Перевод СБП — лимит превышен",  "color": "#7be3c6", "cx": 0.78, "cy": 0.55, "r": 0.08, "n": 287, "q": "medium"},
    {"id": 3, "name": "Реструктуризация кредита",      "color": "#f0b429", "cx": 0.20, "cy": 0.62, "r": 0.10, "n": 256, "q": "high"},
    {"id": 4, "name": "Курс валюты — претензия",       "color": "#e8884a", "cx": 0.45, "cy": 0.78, "r": 0.07, "n": 198, "q": "medium"},
    {"id": 5, "name": "Возврат товара / chargeback",   "color": "#9d7be8", "cx": 0.70, "cy": 0.80, "r": 0.08, "n": 176, "q": "medium"},
    {"id": 6, "name": "Жалоба на оператора",           "color": "#ff5252", "cx": 0.36, "cy": 0.18, "r": 0.06, "n": 121, "q": "low"},
    {"id": 7, "name": "Открытие депозита под % больше","color": "#1de9b6", "cx": 0.85, "cy": 0.18, "r": 0.06, "n":  98, "q": "high"},
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
    scroll = ctk.CTkScrollableFrame(parent, fg_color=COLORS["bg"])
    scroll.pack(fill="both", expand=True)

    app._cluster_selected = 0  # текущий выбранный кластер

    # ── 1. Файлы для кластеризации ──────────────────────────────────
    files_card = Card(scroll, title="Файлы для кластеризации",
                      subtitle="2 файла · 4 906 неразмеченных строк")
    files_card.pack(fill="x", padx=20, pady=(20, 12))

    btns = ctk.CTkFrame(files_card.body, fg_color="transparent")
    btns.pack(fill="x", pady=(0, 10))
    ctk.CTkButton(btns, text="+ Файл(ы)…", width=110,
                  command=getattr(app, "add_cluster_files", lambda: None)).pack(side="left", padx=(0, 6))
    ctk.CTkButton(btns, text="📁 Папка…", width=110,
                  command=getattr(app, "add_cluster_folder", lambda: None)).pack(side="left", padx=(0, 6))
    ctk.CTkButton(btns, text="✨ Определить параметры", width=200,
                  fg_color="transparent", border_width=1, border_color=COLORS["border2"],
                  command=getattr(app, "_auto_detect_cluster_params", lambda: None)
                  ).pack(side="left")

    # «Таблица» файлов
    files_box = ctk.CTkFrame(files_card.body, fg_color=COLORS["entry"],
                             corner_radius=8,
                             border_width=1, border_color=COLORS["border2"])
    files_box.pack(fill="x")
    for fname, rows, size in [
        ("необработанные_апрель.xlsx", "3 218", "1.4 МБ"),
        ("чат_бот_неотвеченные.csv",   "1 688", "0.6 МБ"),
    ]:
        r = ctk.CTkFrame(files_box, fg_color="transparent")
        r.pack(fill="x")
        ctk.CTkLabel(r, text="📄", font=font_base()).pack(side="left", padx=(12, 6), pady=6)
        ctk.CTkLabel(r, text=fname, font=font_mono(),
                     text_color=COLORS["fg"], anchor="w").pack(
            side="left", fill="x", expand=True, pady=6)
        ctk.CTkLabel(r, text=rows, font=font_mono(), width=60,
                     text_color=COLORS["muted"]).pack(side="left", padx=6)
        ctk.CTkLabel(r, text=size, font=font_mono(), width=60,
                     text_color=COLORS["muted"]).pack(side="left", padx=12)

    # ── 2. Алгоритм + Векторизация ──────────────────────────────────
    config_row = ctk.CTkFrame(scroll, fg_color="transparent")
    config_row.pack(fill="x", padx=20, pady=(0, 12))
    config_row.grid_columnconfigure((0, 1), weight=1, uniform="cfg")

    # Алгоритм
    algo_card = Card(config_row, title="Алгоритм")
    algo_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

    algo_var = ctk.StringVar(value="kmeans")
    for val, txt in [
        ("kmeans",       "MiniBatchKMeans · быстрый baseline"),
        ("hdbscan",      "HDBSCAN · плотностной, авто-K"),
        ("hierarchical", "Иерархическая · 2 уровня"),
        ("bertopic",     "BERTopic-подобный pipeline"),
        ("lda",          "LDA · тематическое моделирование"),
    ]:
        ctk.CTkRadioButton(algo_card.body, text=txt, value=val, variable=algo_var,
                           font=font_base()).pack(anchor="w", pady=2)
    app._cluster_algo_var = algo_var

    _separator(algo_card.body)

    k_row = ctk.CTkFrame(algo_card.body, fg_color="transparent")
    k_row.pack(fill="x")
    _field_label(k_row, "K (кластеров)")
    k_inner = ctk.CTkFrame(k_row, fg_color="transparent")
    k_inner.pack(fill="x")
    k_entry = ctk.CTkEntry(k_inner, width=70, font=font_base())
    k_entry.insert(0, "8")
    k_entry.pack(side="left", padx=(0, 12))
    auto_k = ctk.CTkCheckBox(k_inner, text="Авто-подбор K (метод локтя)",
                             font=font_base())
    auto_k.pack(side="left")

    # Векторизация
    vec_card = Card(config_row, title="Векторизация и шум")
    vec_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    _field_label(vec_card.body, "Векторизатор")
    vec_var = ctk.StringVar(value="combo")
    for val, txt in [
        ("tfidf",    "TF-IDF"),
        ("sbert",    "SBERT эмбеддинги"),
        ("combo",    "Комбо · TF-IDF + SBERT"),
        ("ensemble", "Ансамбль · TF-IDF + 2× SBERT"),
    ]:
        ctk.CTkRadioButton(vec_card.body, text=txt, value=val, variable=vec_var,
                           font=font_base()).pack(anchor="w", pady=2)

    _separator(vec_card.body)

    _field_label(vec_card.body, "Фильтрация")
    for txt in ["Стоп-слова", "Шумовые токены", "Шумовые фразы",
                "Игнорировать CHATBOT / SYSTEM", "c-TF-IDF ключевые слова"]:
        cb = ctk.CTkCheckBox(vec_card.body, text=txt, font=font_base())
        cb.select()
        cb.pack(anchor="w", pady=2)

    # ── 3. LLM-нейминг ──────────────────────────────────────────────
    llm_card = Card(scroll, title="LLM-нейминг и описания")
    llm_card.pack(fill="x", padx=20, pady=(0, 12))

    llm_grid = ctk.CTkFrame(llm_card.body, fg_color="transparent")
    llm_grid.pack(fill="x")
    llm_grid.grid_columnconfigure((0, 1), weight=1, uniform="llm")

    opts_col = ctk.CTkFrame(llm_grid, fg_color="transparent")
    opts_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    for txt, on in [
        ("Использовать LLM для названий кластеров", True),
        ("Генерировать обобщённое описание (cluster_reason)", True),
        ("Формировать описание без LLM (rule-based)", False),
        ("LLM-feedback: рекомендации по слиянию кластеров", True),
    ]:
        cb = ctk.CTkCheckBox(opts_col, text=txt, font=font_base())
        if on:
            cb.select()
        cb.pack(anchor="w", pady=3)

    prov_col = ctk.CTkFrame(llm_grid, fg_color="transparent")
    prov_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    _field_label(prov_col, "Провайдер")
    prov_seg = ctk.CTkSegmentedButton(
        prov_col, values=["Claude", "GPT", "GigaChat", "Qwen", "Ollama"])
    prov_seg.set("Claude")
    prov_seg.pack(fill="x", pady=(0, 10))

    fields_row = ctk.CTkFrame(prov_col, fg_color="transparent")
    fields_row.pack(fill="x")
    fields_row.grid_columnconfigure((0, 1), weight=1)
    m_col = ctk.CTkFrame(fields_row, fg_color="transparent")
    m_col.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    _field_label(m_col, "Модель")
    me = ctk.CTkEntry(m_col, font=font_mono())
    me.insert(0, "claude-sonnet-4-5")
    me.pack(fill="x")
    k_col = ctk.CTkFrame(fields_row, fg_color="transparent")
    k_col.grid(row=0, column=1, sticky="ew", padx=(4, 0))
    _field_label(k_col, "API key")
    ke = ctk.CTkEntry(k_col, font=font_mono(), show="•")
    ke.insert(0, "sk-ant-…")
    ke.pack(fill="x")

    # ── 4. Визуализация (scatter + легенда) ─────────────────────────
    viz_actions = ctk.CTkFrame(None, fg_color="transparent")
    ctk.CTkButton(viz_actions, text="📥 Экспорт .xlsx", width=130,
                  fg_color="transparent", border_width=1, border_color=COLORS["border2"],
                  command=getattr(app, "export_cluster_results", lambda: None)
                  ).pack(side="left", padx=(0, 6))
    ctk.CTkButton(viz_actions, text="▶  Запустить", width=120, height=36,
                  fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                  text_color="#04221a" if COLORS["bg"].startswith("#0") else "#ffffff",
                  font=font_md_bold(),
                  command=getattr(app, "start_cluster", lambda: None)).pack(side="left")

    viz_card = Card(scroll, title="Визуализация кластеров",
                    subtitle="UMAP 2D · 1 886 точек · 8 кластеров",
                    right=viz_actions)
    viz_card.pack(fill="x", padx=20, pady=(0, 12))

    viz_split = ctk.CTkFrame(viz_card.body, fg_color="transparent")
    viz_split.pack(fill="x")
    viz_split.grid_columnconfigure(0, weight=1)
    viz_split.grid_columnconfigure(1, weight=0, minsize=320)

    scatter_holder = ctk.CTkFrame(viz_split, fg_color=COLORS["entry"],
                                  corner_radius=8,
                                  border_width=1, border_color=COLORS["border2"],
                                  height=420)
    scatter_holder.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    scatter_holder.grid_propagate(False)
    _build_scatter(app, scatter_holder)

    # Легенда
    legend = ctk.CTkScrollableFrame(viz_split, fg_color=COLORS["panel"],
                                    corner_radius=8, height=420,
                                    border_width=1, border_color=COLORS["border2"])
    legend.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    ctk.CTkLabel(legend, text="КЛАСТЕРЫ", font=font_label(),
                 text_color=COLORS["accent2"]).pack(anchor="w", padx=10, pady=(8, 4))

    app._cluster_legend_items = {}
    for seed in CLUSTER_SEEDS:
        _legend_item(app, legend, seed)

    # ── 5. Деталь по выбранному кластеру ────────────────────────────
    detail_actions = ctk.CTkFrame(None, fg_color="transparent")
    ctk.CTkButton(detail_actions, text="✏️ Переименовать", width=130,
                  fg_color="transparent", border_width=1, border_color=COLORS["border2"]
                  ).pack(side="left", padx=(0, 6))
    ctk.CTkButton(detail_actions, text="✨ Через LLM", width=110,
                  fg_color="transparent", border_width=1, border_color=COLORS["border2"]
                  ).pack(side="left")

    detail_card = Card(scroll, title="Кластер #0",
                       subtitle="412 строк · качество: high · LLM-name готово",
                       right=detail_actions)
    detail_card.pack(fill="x", padx=20, pady=(0, 20))
    app._cluster_detail_card = detail_card

    _render_cluster_detail(app, detail_card.body, CLUSTER_SEEDS[0])

    return scroll


# ─────────────────────────────────────────────────────────────────────
# Scatter (matplotlib)
# ─────────────────────────────────────────────────────────────────────

def _build_scatter(app, holder):
    """Рендерит UMAP-scatter через matplotlib в стиле текущей темы."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import random
    except ImportError:
        ctk.CTkLabel(holder, text="matplotlib не установлен.\npip install matplotlib",
                     font=font_base(), text_color=COLORS["muted"]).pack(expand=True)
        return

    fig = Figure(figsize=(6, 4.2), dpi=90)
    fig.patch.set_facecolor(COLORS["entry"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(COLORS["entry"])

    # noise
    rnd = random.Random(1234)
    for _ in range(50):
        ax.scatter(rnd.random(), rnd.random(), s=6,
                   color=COLORS["muted"], alpha=0.25)

    # cluster points
    for seed in CLUSTER_SEEDS:
        n_pts = seed["n"] // 4
        xs, ys = [], []
        for _ in range(n_pts):
            import math
            ang = rnd.random() * 2 * math.pi
            rad = math.sqrt(rnd.random()) * seed["r"]
            xs.append(seed["cx"] + math.cos(ang) * rad)
            ys.append(seed["cy"] + math.sin(ang) * rad)
        ax.scatter(xs, ys, s=12, color=seed["color"], alpha=0.7,
                   edgecolors="none")
        # центр
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
    """Кликабельный элемент легенды."""
    f = ctk.CTkFrame(parent, fg_color="transparent",
                     corner_radius=6)
    f.pack(fill="x", padx=8, pady=2)

    def on_click(_=None):
        app._cluster_selected = seed["id"]
        # подсветка выбранного
        for sid, w in app._cluster_legend_items.items():
            w.configure(fg_color=(COLORS["select"] if sid == seed["id"] else "transparent"))
        # пере-render деталей
        for child in app._cluster_detail_card.body.winfo_children():
            child.destroy()
        _render_cluster_detail(app, app._cluster_detail_card.body, seed)

    f.bind("<Button-1>", on_click)
    inner = ctk.CTkFrame(f, fg_color="transparent")
    inner.pack(fill="x", padx=8, pady=6)
    inner.bind("<Button-1>", on_click)

    dot = ctk.CTkLabel(inner, text="●", font=font_md_bold(),
                       text_color=seed["color"])
    dot.pack(side="left", padx=(0, 6))
    dot.bind("<Button-1>", on_click)

    name = ctk.CTkLabel(inner, text=seed["name"], font=font_sm(),
                        text_color=COLORS["fg"], anchor="w")
    name.pack(side="left", fill="x", expand=True)
    name.bind("<Button-1>", on_click)

    cnt = ctk.CTkLabel(inner, text=str(seed["n"]), font=font_mono(),
                       text_color=COLORS["muted"])
    cnt.pack(side="right", padx=(6, 0))
    cnt.bind("<Button-1>", on_click)

    qual = "✓" if seed["q"] == "high" else "◐" if seed["q"] == "medium" else "✕"
    qcolor = COLORS["success"] if seed["q"] == "high" else COLORS["warning"] if seed["q"] == "medium" else COLORS["error"]
    ql = ctk.CTkLabel(inner, text=qual, font=font_md_bold(), text_color=qcolor)
    ql.pack(side="right", padx=(6, 0))
    ql.bind("<Button-1>", on_click)

    app._cluster_legend_items[seed["id"]] = f


def _render_cluster_detail(app, parent, seed):
    """Рендерит подробности по выбранному кластеру."""
    grid = ctk.CTkFrame(parent, fg_color="transparent")
    grid.pack(fill="both", expand=True)
    grid.grid_columnconfigure((0, 1), weight=1, uniform="d")

    # Левая колонка
    left = ctk.CTkFrame(grid, fg_color="transparent")
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

    _field_label(left, "LLM Название")
    ctk.CTkLabel(left, text=seed["name"], font=font_lg_bold(),
                 text_color=COLORS["fg"], anchor="w").pack(anchor="w", pady=(0, 12))

    _field_label(left, "LLM Описание")
    desc = ctk.CTkTextbox(left, height=90, font=font_base(),
                          fg_color=COLORS["panel2"], wrap="word")
    desc.insert("1.0",
                "Клиенты сообщают о невозможности получить SMS или push-уведомление "
                "с кодом подтверждения. Решение: проверка SMS-шлюза и переотправка.")
    desc.configure(state="disabled")
    desc.pack(fill="x", pady=(0, 12))

    _field_label(left, "Ключевые слова (c-TF-IDF)")
    kw_frame = ctk.CTkFrame(left, fg_color="transparent")
    kw_frame.pack(fill="x", anchor="w")
    for kw in CLUSTER_KEYWORDS.get(seed["id"], ["—"]):
        Pill(kw_frame, kw, "accent").pack(side="left", padx=(0, 4), pady=2)

    # Правая колонка
    right = ctk.CTkFrame(grid, fg_color="transparent")
    right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    _field_label(right, "Примеры обращений")
    examples = [
        "не приходит код для входа уже 20 минут",
        "жду смс с кодом авторизации, не приходит",
        "push-уведомления не приходят, как войти?",
    ]
    for i, ex in enumerate(examples):
        box = ctk.CTkFrame(right, fg_color=COLORS["panel2"], corner_radius=6,
                           border_width=1, border_color=COLORS["border2"])
        box.pack(fill="x", pady=4)
        meta = ctk.CTkFrame(box, fg_color="transparent")
        meta.pack(fill="x", padx=10, pady=(6, 0))
        ctk.CTkLabel(meta, text=f"#{102100 + seed['id']*10 + i}",
                     font=font_mono(), text_color=COLORS["muted"]).pack(side="left")
        ctk.CTkLabel(meta, text=f"cos sim 0.{92 - i*4}",
                     font=font_mono(), text_color=COLORS["accent2"]).pack(side="right")
        ctk.CTkLabel(box, text=f"«{ex}»", font=font_base(),
                     text_color=COLORS["fg"], anchor="w", wraplength=380,
                     justify="left").pack(fill="x", padx=10, pady=(2, 8))
