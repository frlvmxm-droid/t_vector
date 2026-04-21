# -*- coding: utf-8 -*-
"""
app_train_view_ctk.py — view-слой вкладки «Обучение» на CustomTkinter.

build_train_tab(app, parent) создаёт все виджеты, которые нужны TrainTabMixin:
  app.lb_train, app.train_log, app.btn_train, app.btn_train_stop,
  app.train_progress_pb, app.train_phase_lbl, app.train_speed_lbl, app.train_eta_lbl,
  app.cb_auto_profile, app.cb_sbert_model_combo, app.cb_sbert_device_combo,
  app.cb_setfit_model_combo, app.cb_calib_method.
Колонки Excel: app.desc_col / call_col / chat_col / label_col / summary_col /
  ans_short_col / ans_full_col — StringVar'ы уже созданы в app.__init__;
  view лишь создаёт ttk.Combobox, привязанные к ним.
"""
from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk

import customtkinter as ctk

from ui_theme_ctk import (
    COLORS,
    font_label, font_sm, font_base, font_md_bold, font_lg_bold, font_mono,
)
from ui_widgets_tk import Tooltip


# ─────────────────────────────────────────────────────────────────────
# Переиспользуемые компоненты
# ─────────────────────────────────────────────────────────────────────

class Card(ctk.CTkFrame):
    """Плоская секция Paper-стиля: ALL_CAPS заголовок без лишних рамок."""
    def __init__(self, parent, title: str | None = None, subtitle: str | None = None,
                 right=None, **kwargs):
        super().__init__(parent, corner_radius=0, fg_color="transparent", **kwargs)
        if title:
            header = ctk.CTkFrame(self, fg_color="transparent")
            header.pack(fill="x", pady=(4, 6))
            left = ctk.CTkFrame(header, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(left, text=title.upper(), font=font_label(),
                         text_color=COLORS["accent"]).pack(anchor="w")
            if subtitle:
                ctk.CTkLabel(left, text=subtitle, font=font_sm(),
                             text_color=COLORS["muted"]).pack(anchor="w", pady=(2, 0))
            if right is not None:
                right.pack(in_=header, side="right")
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=0, pady=(0, 10))


def Pill(parent, text: str, kind: str = "default") -> ctk.CTkLabel:
    """Маленький бейдж-пилюля."""
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
    """KPI-тайл: заголовок + значение + дельта."""
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
    Строит вкладку «Обучение» (Paper-стиль).
    Все виджеты, нужные TrainTabMixin, присваиваются на `app`.
    """
    scroll = ctk.CTkScrollableFrame(
        parent,
        fg_color=COLORS["bg"],
        scrollbar_fg_color=COLORS["panel"],
        scrollbar_button_color=COLORS["border"],
        scrollbar_button_hover_color=COLORS["accent3"],
    )
    scroll.pack(fill="both", expand=True)

    _PAD = dict(padx=28)   # горизонтальный отступ секций

    # ════════════════════════════════════════════════════════════════
    # 1. ФАЙЛЫ ОБУЧАЮЩЕЙ ВЫБОРКИ
    # ════════════════════════════════════════════════════════════════
    files_sec = Card(scroll, title="Файлы обучающей выборки")
    files_sec.pack(fill="x", **_PAD, pady=(20, 0))

    # ── кнопки управления ─────────────────────────────────────────
    fbtns = ctk.CTkFrame(files_sec.body, fg_color="transparent")
    fbtns.pack(fill="x", pady=(0, 8))

    def _ghost_btn(parent, text, cmd, tooltip=""):
        b = ctk.CTkButton(parent, text=text, height=30, width=0,
                          fg_color="transparent", border_width=1,
                          border_color=COLORS["border"], font=font_sm(),
                          text_color=COLORS["fg"], hover_color=COLORS["hover"],
                          command=cmd)
        b.pack(side="left", padx=(0, 6))
        if tooltip:
            Tooltip(b, tooltip)
        return b

    _btn_add = ctk.CTkButton(
        fbtns, text="+ Добавить…", height=30, width=130,
        fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
        font=font_sm(), text_color="#ffffff",
        command=getattr(app, "add_train_files", lambda: None))
    _btn_add.pack(side="left", padx=(0, 6))
    Tooltip(_btn_add, "Добавить Excel/CSV-файлы с размеченными примерами.")
    _ghost_btn(fbtns, "📁 Папка…",    getattr(app, "add_train_folder", lambda: None),
               "Рекурсивно добавить все .xlsx/.csv из папки.")
    _ghost_btn(fbtns, "📊 Статистика", getattr(app, "show_dataset_stats", lambda: None),
               "Распределение классов по файлам.")
    _ghost_btn(fbtns, "🗑 Удалить",    getattr(app, "remove_train_file", lambda: None),
               "Удалить выбранный файл из списка.")

    # ── treeview-таблица файлов ───────────────────────────────────
    _files_summary = tk.StringVar(value="Нет файлов")
    ctk.CTkLabel(files_sec.body, textvariable=_files_summary,
                 font=font_sm(), text_color=COLORS["muted"], anchor="w"
                 ).pack(anchor="w", pady=(0, 6))

    tree_host = ctk.CTkFrame(files_sec.body, fg_color=COLORS["panel"],
                             corner_radius=6, border_width=1,
                             border_color=COLORS["border"])
    tree_host.pack(fill="x")

    _style = ttk.Style()
    _style.configure("Files.Treeview",
                     background=COLORS["panel"], foreground=COLORS["fg"],
                     fieldbackground=COLORS["panel"], rowheight=28,
                     borderwidth=0, relief="flat")
    _style.configure("Files.Treeview.Heading",
                     background=COLORS["panel2"], foreground=COLORS["muted"],
                     font=("Segoe UI", 10, "bold"), relief="flat", borderwidth=0)
    _style.map("Files.Treeview", background=[("selected", COLORS["select"])],
               foreground=[("selected", COLORS["accent2"])])
    _style.layout("Files.Treeview", [("Files.Treeview.treearea", {"sticky": "nswe"})])

    _tree = ttk.Treeview(tree_host, style="Files.Treeview",
                         columns=("file", "rows", "cols", "size"),
                         show="headings", height=5, selectmode="extended")
    for col, heading, w, anchor in [
        ("file",  "ФАЙЛ",     380, "w"),
        ("rows",  "СТРОКИ",   80,  "e"),
        ("cols",  "КОЛОНКИ",  80,  "e"),
        ("size",  "РАЗМЕР",   90,  "e"),
    ]:
        _tree.heading(col, text=heading, anchor=anchor)
        _tree.column(col, width=w, anchor=anchor, stretch=(col == "file"))
    _tree_sb = ttk.Scrollbar(tree_host, orient="vertical", command=_tree.yview)
    _tree.configure(yscrollcommand=_tree_sb.set)
    _tree.pack(side="left", fill="both", expand=True, padx=1, pady=1)
    _tree_sb.pack(side="right", fill="y")

    app.lb_train = _FileListBox(_tree, _files_summary)

    # ════════════════════════════════════════════════════════════════
    # 2. ВЕКТОРИЗАЦИЯ И БАЗОВАЯ МОДЕЛЬ  (два столбца)
    # ════════════════════════════════════════════════════════════════
    vec_sec = Card(scroll, title="Векторизация и базовая модель")
    vec_sec.pack(fill="x", **_PAD, pady=0)

    two_col = ctk.CTkFrame(vec_sec.body, fg_color="transparent")
    two_col.pack(fill="x")
    two_col.grid_columnconfigure(0, weight=3)
    two_col.grid_columnconfigure(1, weight=2)

    # ── ЛЕВАЯ КОЛОНКА ─────────────────────────────────────────────
    left_col = ctk.CTkFrame(two_col, fg_color="transparent")
    left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 24))

    # БАЗОВАЯ МОДЕЛЬ — два кнопки-переключателя
    _field_label(left_col, "Базовая модель")
    _base_row = ctk.CTkFrame(left_col, fg_color="transparent")
    _base_row.pack(fill="x", pady=(0, 14))
    _base_mode = getattr(app, "base_model_mode", None) or tk.StringVar(value="scratch")
    if not hasattr(app, "base_model_mode"):
        app.base_model_mode = _base_mode

    def _base_btn(text, val):
        def _cmd():
            _base_mode.set(val)
            for v2, b2 in _base_btns.items():
                b2.configure(
                    fg_color=COLORS["accent"] if v2 == val else COLORS["panel2"],
                    text_color="#ffffff" if v2 == val else COLORS["muted"],
                )
        b = ctk.CTkButton(_base_row, text=text, height=30, width=0,
                          fg_color=COLORS["accent"] if val == _base_mode.get() else COLORS["panel2"],
                          text_color="#ffffff" if val == _base_mode.get() else COLORS["muted"],
                          hover_color=COLORS["accent2"], font=font_sm(),
                          command=_cmd, corner_radius=6)
        b.pack(side="left", padx=(0, 4))
        return b

    _base_btns: dict = {}
    _base_btns["scratch"]  = _base_btn("С нуля", "scratch")
    _base_btns["finetune"] = _base_btn("Дообучение от .joblib", "finetune")

    # ВЕКТОРИЗАТОР
    _field_label(left_col, "Векторизатор")
    _MODE_OPTIONS = [
        ("tfidf",  "TF-IDF (быстрый, baseline)",
         "Быстро, не требует GPU. Рекомендуется при 500+ примерах на класс."),
        ("sbert",  "SBERT — нейросетевые эмбеддинги",
         "Лучшее качество на GPU. Требует sentence-transformers."),
        ("hybrid", "Гибрид TF-IDF + SBERT (рекомендуется)",
         "Объединяет оба вида эмбеддингов. Требует GPU и sentence-transformers."),
        ("setfit", "Ансамбль · TF-IDF + 2× SBERT",
         "Нейросетевой few-shot классификатор. Эффективен при 8–64 примерах на класс."),
    ]
    _mode_var = getattr(app, "train_vec_mode", None)
    for _val, _label, _tip in _MODE_OPTIONS:
        _rb = ctk.CTkRadioButton(
            left_col, text=_label, value=_val, variable=_mode_var,
            font=font_sm(), text_color=COLORS["fg"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
            border_color=COLORS["border"],
        )
        _rb.pack(anchor="w", pady=2)
        Tooltip(_rb, _tip)

    # ── ПРАВАЯ КОЛОНКА ────────────────────────────────────────────
    right_col = ctk.CTkFrame(two_col, fg_color="transparent")
    right_col.grid(row=0, column=1, sticky="nsew")

    # ПАРАМЕТРЫ ОБУЧЕНИЯ
    _field_label(right_col, "Параметры обучения")
    _param_grid = ctk.CTkFrame(right_col, fg_color="transparent")
    _param_grid.pack(fill="x", pady=(0, 12))
    _param_grid.grid_columnconfigure((0, 1), weight=1)

    def _param_entry(parent, label, var, row, col):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=col, sticky="ew", padx=(0, 8) if col == 0 else (0, 0),
               pady=(0, 10))
        ctk.CTkLabel(f, text=label.upper(), font=font_label(),
                     text_color=COLORS["muted"], anchor="w").pack(anchor="w")
        ctk.CTkEntry(f, textvariable=var, height=34, font=font_base(),
                     fg_color=COLORS["entry"], border_color=COLORS["border"],
                     border_width=1, corner_radius=4).pack(fill="x", pady=(4, 0))

    _param_entry(_param_grid, "Test Split",   getattr(app, "test_size", None),     0, 0)
    _param_entry(_param_grid, "Регуляриз. C", getattr(app, "C", None),             0, 1)
    _param_entry(_param_grid, "Max Iter",     getattr(app, "max_iter", None),      1, 0)

    # N-Grams combined entry (word_ng_min + word_ng_max как строка)
    _ng_frame = ctk.CTkFrame(_param_grid, fg_color="transparent")
    _ng_frame.grid(row=1, column=1, sticky="ew", pady=(0, 10))
    ctk.CTkLabel(_ng_frame, text="N GRAMS TF-IDF", font=font_label(),
                 text_color=COLORS["muted"], anchor="w").pack(anchor="w")
    _ng_row = ctk.CTkFrame(_ng_frame, fg_color="transparent")
    _ng_row.pack(fill="x", pady=(4, 0))
    ctk.CTkEntry(_ng_row, textvariable=getattr(app, "word_ng_min", None),
                 width=44, height=34, font=font_base(),
                 fg_color=COLORS["entry"], border_color=COLORS["border"],
                 border_width=1, corner_radius=4).pack(side="left")
    ctk.CTkLabel(_ng_row, text="–", font=font_sm(),
                 text_color=COLORS["muted"]).pack(side="left", padx=4)
    ctk.CTkEntry(_ng_row, textvariable=getattr(app, "word_ng_max", None),
                 width=44, height=34, font=font_base(),
                 fg_color=COLORS["entry"], border_color=COLORS["border"],
                 border_width=1, corner_radius=4).pack(side="left")

    # ОПЦИИ
    _field_label(right_col, "Опции")
    _OPTIONS = [
        ("calib_method",          "CalibratedClassifierCV",         None,  True),
        ("class_weight_balanced", "Стратифицированный split",       None,  None),
        ("use_smote",             "ML-аугментация (SMOTE)",         None,  None),
    ]
    for _vname, _lbl, _tip, _is_str in _OPTIONS:
        _var = getattr(app, _vname, None)
        if _is_str:
            # calib_method — строка; показываем как чекбокс (не "none")
            _bool_proxy = tk.BooleanVar(value=(_var.get() != "none") if _var else True)
            def _on_calib(vname=_vname, proxy=_bool_proxy, strvar=_var):
                if strvar:
                    strvar.set("sigmoid" if proxy.get() else "none")
            _bool_proxy.trace_add("write", lambda *_, p=_bool_proxy, sv=_var:
                sv.set("sigmoid" if p.get() else "none") if sv else None)
            _cb = ctk.CTkCheckBox(right_col, text=_lbl, variable=_bool_proxy,
                                  font=font_sm(), text_color=COLORS["fg"],
                                  fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                                  border_color=COLORS["border"])
        else:
            _cb = ctk.CTkCheckBox(right_col, text=_lbl, variable=_var,
                                  font=font_sm(), text_color=COLORS["fg"],
                                  fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                                  border_color=COLORS["border"])
        _cb.pack(anchor="w", pady=2)

    # ════════════════════════════════════════════════════════════════
    # 3. КОЛОНКИ ДАННЫХ (свёртываемая секция)
    # ════════════════════════════════════════════════════════════════
    cols_container = ctk.CTkFrame(scroll, fg_color="transparent")
    cols_container.pack(fill="x", **_PAD, pady=0)

    ctk.CTkFrame(cols_container, height=1, fg_color=COLORS["border"]).pack(fill="x")
    _cols_hdr = ctk.CTkFrame(cols_container, fg_color="transparent")
    _cols_hdr.pack(fill="x", pady=(10, 6))
    ctk.CTkLabel(_cols_hdr, text="КОЛОНКИ ДАННЫХ", font=font_label(),
                 text_color=COLORS["accent"], anchor="w").pack(side="left")
    _cols_open = [False]
    _cols_toggle = ctk.CTkButton(
        _cols_hdr, text="▶ Развернуть", width=110, height=26,
        fg_color="transparent", border_width=1, border_color=COLORS["border"],
        font=font_sm(), text_color=COLORS["muted"], hover_color=COLORS["hover"],
    )
    _cols_toggle.pack(side="right")

    _cols_body = ctk.CTkFrame(cols_container, fg_color="transparent")

    def _toggle_cols():
        if _cols_open[0]:
            _cols_body.pack_forget()
            _cols_toggle.configure(text="▶ Развернуть")
        else:
            _cols_body.pack(fill="x", pady=(0, 8))
            _cols_toggle.configure(text="▼ Свернуть")
        _cols_open[0] = not _cols_open[0]

    _cols_toggle.configure(command=_toggle_cols)

    col_grid = ctk.CTkFrame(_cols_body, fg_color="transparent")
    col_grid.pack(fill="x")
    col_grid.grid_columnconfigure(1, weight=1)
    col_grid.grid_columnconfigure(3, weight=1)

    col_defs = [
        ("Описание:",      getattr(app, "desc_col",      None)),
        ("Текст звонка:",  getattr(app, "call_col",      None)),
        ("Текст чата:",    getattr(app, "chat_col",      None)),
        ("Label / причина:", getattr(app, "label_col",   None)),
        ("Суммаризация:",  getattr(app, "summary_col",   None)),
        ("Ответ краткий:", getattr(app, "ans_short_col", None)),
        ("Ответ полный:",  getattr(app, "ans_full_col",  None)),
    ]
    for i, (lbl, var) in enumerate(col_defs):
        r, c_off = divmod(i, 2)
        ctk.CTkLabel(col_grid, text=lbl, font=font_sm(),
                     text_color=COLORS["muted"], anchor="e").grid(
            row=r, column=c_off * 2, sticky="e", padx=(0, 4), pady=3)
        cb = ttk.Combobox(col_grid, textvariable=var,
                          state="readonly", width=30, values=[])
        cb.grid(row=r, column=c_off * 2 + 1, sticky="ew", padx=(0, 16), pady=3)

    # ── 2.5. Веса секций ────────────────────────────────────────────
    weights_card = Card(scroll, title="Веса секций",
                        subtitle="Влияние каждой колонки на классификацию")
    weights_card.pack(fill="x", padx=20, pady=(0, 12))

    _WEIGHT_SECTIONS = [
        ("w_desc",     "Описание"),
        ("w_client",   "Клиент"),
        ("w_operator", "Оператор"),
        ("w_summary",  "Резюме"),
        ("w_ans_short","Ответ кратко"),
        ("w_ans_full", "Ответ полно"),
    ]
    # (desc, client, operator, summary, ans_short, ans_full)
    _WEIGHT_PRESETS = {
        "Равномерные":       (1, 1, 1, 1, 1, 1),
        "Акцент: описание":  (3, 2, 1, 2, 1, 1),
        "Акцент: клиент":    (2, 3, 1, 2, 1, 1),
        "Без ответов":       (2, 3, 1, 2, 0, 0),
        "Только описание":   (3, 0, 0, 0, 0, 0),
    }

    _w_slider_disp: list = []  # (disp_var, slider, var_name)

    def _apply_weight_preset(name):
        vals = _WEIGHT_PRESETS.get(name)
        if not vals:
            return
        for (dv, sl, vname), val in zip(_w_slider_disp, vals):
            v = getattr(app, vname, None)
            if v:
                v.set(val)
                sl.set(val)
                dv.set(str(val))

    preset_row = ctk.CTkFrame(weights_card.body, fg_color="transparent")
    preset_row.pack(fill="x", pady=(0, 6))
    ctk.CTkLabel(preset_row, text="Пресет:", font=font_sm(),
                 text_color=COLORS["muted"], width=60).pack(side="left")
    _preset_cb = ctk.CTkComboBox(
        preset_row, values=list(_WEIGHT_PRESETS.keys()),
        command=_apply_weight_preset,
        state="readonly",
        fg_color=COLORS["panel2"], border_color=COLORS["border2"],
        button_color=COLORS["accent3"], button_hover_color=COLORS["accent"],
        text_color=COLORS["fg"], dropdown_fg_color=COLORS["panel2"],
        dropdown_text_color=COLORS["fg"], dropdown_hover_color=COLORS["hover"],
        font=font_sm(), width=200,
    )
    _preset_cb.set(list(_WEIGHT_PRESETS.keys())[0])
    _preset_cb.pack(side="left", padx=(6, 0))

    # Разворачиваемая детальная настройка
    _wdet_open = [False]
    _wdet_toggle = ctk.CTkButton(
        weights_card.body, text="▶  Развернуть детали", anchor="w", height=28,
        fg_color="transparent", font=font_sm(), text_color=COLORS["muted"],
        hover_color=COLORS["hover"],
    )
    _wdet_toggle.pack(fill="x")
    _wdet_frame = ctk.CTkFrame(weights_card.body, fg_color="transparent")

    for _wvar, _wlbl in _WEIGHT_SECTIONS:
        _var = getattr(app, _wvar, None)
        _wrow = ctk.CTkFrame(_wdet_frame, fg_color="transparent")
        _wrow.pack(fill="x", pady=2)
        ctk.CTkLabel(_wrow, text=_wlbl, width=120, anchor="w",
                     font=font_sm(), text_color=COLORS["muted"]).pack(side="left")
        _disp_var = tk.StringVar(value=str(_var.get() if _var else 1))
        _slider = ctk.CTkSlider(
            _wrow, from_=0, to=4, number_of_steps=4,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent2"], button_hover_color=COLORS["accent"],
        )
        if _var is not None:
            _slider.set(_var.get())
            def _on_weight(val, v=_var, dv=_disp_var):
                iv = int(round(val))
                v.set(iv)
                dv.set(str(iv))
            _slider.configure(command=_on_weight)
        _slider.pack(side="left", padx=(8, 6), fill="x", expand=True)
        ctk.CTkLabel(_wrow, textvariable=_disp_var, width=24,
                     font=font_mono(), text_color=COLORS["fg"]).pack(side="left")
        _w_slider_disp.append((_disp_var, _slider, _wvar))

    def _toggle_wdet():
        if _wdet_open[0]:
            _wdet_frame.pack_forget()
            _wdet_toggle.configure(text="▶  Развернуть детали")
        else:
            _wdet_frame.pack(fill="x", pady=(4, 0))
            _wdet_toggle.configure(text="▼  Свернуть")
        _wdet_open[0] = not _wdet_open[0]

    _wdet_toggle.configure(command=_toggle_wdet)

    # ── 3. Предобработка текста ─────────────────────────────────────
    mode_card = Card(scroll, title="Предобработка текста",
                     subtitle="Нормализация и фильтрация токенов")
    mode_card.pack(fill="x", padx=20, pady=(0, 12))
    _prep_row = ctk.CTkFrame(mode_card.body, fg_color="transparent")
    _prep_row.pack(fill="x", pady=(4, 0))

    _PREPROC_OPTS = [
        ("use_lemma",         "Лемматизация",
         "Приводит слова к начальной форме через pymorphy3/pymorphy2.\nУлучшает recall, особенно для глагольных форм."),
        ("use_stop_words",    "Стоп-слова (рус.)",
         "Удаляет частые незначимые слова (предлоги, союзы).\nУлучшает точность TF-IDF на ~2–5%."),
        ("use_noise_tokens",  "Шумовые токены",
         "Удаляет одиночные буквы, цифры и спецсимволы из текста."),
        ("use_noise_phrases", "Шумовые фразы",
         "Удаляет типовые шаблонные фразы банковского колл-центра."),
    ]
    for _vname, _lbl, _tip in _PREPROC_OPTS:
        _cb = ctk.CTkCheckBox(
            _prep_row, text=_lbl,
            variable=getattr(app, _vname, None),
            text_color=COLORS["fg"],
        )
        _cb.pack(side="left", padx=(0, 20))
        Tooltip(_cb, _tip)

    _btn_excl = ctk.CTkButton(
        mode_card.body, text="Редактировать исключения →", anchor="w",
        height=28, fg_color="transparent", border_width=1,
        border_color=COLORS["border2"], font=font_sm(), text_color=COLORS["muted"],
        hover_color=COLORS["hover"],
        command=lambda: _open_exclusions(app),
    )
    _btn_excl.pack(anchor="w", pady=(8, 0))
    Tooltip(_btn_excl,
            "Открыть редактор стоп-слов, шумовых токенов и шумовых фраз.\n"
            "Изменения сохраняются в ~/.hearsy/user_exclusions.json.")

    def _open_exclusions(app_ref):
        try:
            from app_dialogs_ctk import ExclusionsDialog
            ExclusionsDialog(app_ref).show()
        except Exception as _e:
            import tkinter.messagebox as mb
            mb.showerror("Ошибка", str(_e))

    # ── 4. Конфигурация модели (защищённые комбобоксы) ──────────────
    cfg_card = Card(scroll, title="Конфигурация модели")
    cfg_card.pack(fill="x", padx=20, pady=(0, 12))

    cfg_grid = ctk.CTkFrame(cfg_card.body, fg_color="transparent")
    cfg_grid.pack(fill="x")
    cfg_grid.grid_columnconfigure(1, weight=1)
    cfg_grid.grid_columnconfigure(3, weight=1)

    cfg_rows = [
        ("Auto profile:",   "cb_auto_profile",       "auto_profile",
         ["off", "smart", "strict"], "readonly"),
        ("SBERT модель:",   "cb_sbert_model_combo",  "sbert_model",
         _sbert_models_list(), "normal"),
        ("SBERT device:",   "cb_sbert_device_combo", "sbert_device",
         getattr(app, "gpu_device_values", ["auto", "cpu", "cuda"]), "readonly"),
        ("SetFit модель:",  "cb_setfit_model_combo", "setfit_model",
         _setfit_models_list(), "normal"),
        ("Калибровка:",     "cb_calib_method",       "calib_method",
         ["sigmoid", "isotonic", "temperature"], "readonly"),
    ]
    for row_i, (lbl, attr, var_name, vals, state) in enumerate(cfg_rows):
        r, c_off = divmod(row_i, 2)
        ctk.CTkLabel(cfg_grid, text=lbl, font=font_sm(),
                     text_color=COLORS["muted"], anchor="e").grid(
            row=r, column=c_off * 2, sticky="e", padx=(0, 4), pady=3)
        var = getattr(app, var_name, None)
        cb = ttk.Combobox(cfg_grid, textvariable=var, state=state, width=30, values=vals)
        cb.grid(row=r, column=c_off * 2 + 1, sticky="ew", padx=(0, 16), pady=3)
        setattr(app, attr, cb)

    # ── 4.5. Расширенные параметры (accordion) ──────────────────────
    # Контейнер всегда запакован — он держит позицию между cfg_card и run_card
    _adv_container = ctk.CTkFrame(scroll, fg_color="transparent")
    _adv_container.pack(fill="x", padx=20, pady=(0, 12))

    _adv_open = [False]
    _adv_toggle_btn = ctk.CTkButton(
        _adv_container, text="▶  Расширенные параметры", anchor="w", height=34,
        fg_color="transparent", border_width=1, border_color=COLORS["border2"],
        hover_color=COLORS["hover"], text_color=COLORS["muted"], font=font_sm(),
    )
    _adv_toggle_btn.pack(fill="x")

    # Рамка аккордиона — дочерний виджет контейнера, всегда остаётся на месте
    _adv_frame = ctk.CTkFrame(_adv_container, fg_color=COLORS["panel2"], corner_radius=10)

    def _toggle_adv():
        if _adv_open[0]:
            _adv_frame.pack_forget()
            _adv_toggle_btn.configure(text="▶  Расширенные параметры")
        else:
            _adv_frame.pack(fill="x", pady=(4, 0))
            _adv_toggle_btn.configure(text="▼  Расширенные параметры")
        _adv_open[0] = not _adv_open[0]

    _adv_toggle_btn.configure(command=_toggle_adv)

    _adv_body = ctk.CTkFrame(_adv_frame, fg_color="transparent")
    _adv_body.pack(fill="x", padx=18, pady=(12, 14))

    _ADV_OPTIONS = [
        ("use_optuna",           "Optuna авто-тюнинг гиперпараметров",
         "Автоматически подбирает C, char/word n-gram диапазоны через Optuna (30 trials)."),
        ("use_kfold_ensemble",   "K-fold ансамбль классификаторов",
         "Обучает K моделей на K-fold разбиении и усредняет предсказания."),
        ("use_confident_learning", "Confident Learning (очистка меток)",
         "Обнаруживает и исключает примеры с ошибочными метками через CleanLab."),
        ("use_label_smoothing",  "Label smoothing (мягкие метки)",
         "Заменяет жёсткие метки 0/1 на 0+ε/1−ε, снижает переобучение."),
        ("use_llm_augment",       "LLM-аугментация редких классов",
         "Использует LLM для генерации перефразировок примеров с < augment_min_samples образцов."),
        ("detect_near_dups",     "Near-dup дедупликация обучения",
         "Удаляет почти-дублирующиеся тексты (cosine ≥ near_dup_threshold) из train."),
        ("use_hierarchical",     "Иерархическая классификация",
         "Обучает двухуровневый классификатор: сначала группы, затем тонкие классы."),
    ]
    for _bvar, _blbl, _btip in _ADV_OPTIONS:
        _bv = getattr(app, _bvar, None)
        _sw = ctk.CTkSwitch(
            _adv_body, text=_blbl, variable=_bv,
            font=font_sm(), text_color=COLORS["fg"],
            progress_color=COLORS["accent"], button_color=COLORS["accent2"],
        )
        _sw.pack(anchor="w", pady=3)
        Tooltip(_sw, _btip)

        # H3: sub-row для K-fold K
        if _bvar == "use_kfold_ensemble":
            _kfold_row = ctk.CTkFrame(_adv_body, fg_color="transparent")
            _kfold_row.pack(anchor="w", padx=(28, 0), pady=(0, 2))
            ctk.CTkLabel(_kfold_row, text="K:", font=font_sm(),
                         text_color=COLORS["muted"], width=20).pack(side="left")
            ctk.CTkEntry(_kfold_row, textvariable=getattr(app, "kfold_k", None),
                         width=52, height=24, font=font_sm(),
                         fg_color=COLORS["entry"], border_color=COLORS["border2"],
                         border_width=1).pack(side="left", padx=(4, 0))
            ctk.CTkLabel(_kfold_row, text="(2–10)", font=font_sm(),
                         text_color=COLORS["muted"]).pack(side="left", padx=(4, 0))

        # H4: sub-row для label smoothing eps
        if _bvar == "use_label_smoothing":
            _ls_row = ctk.CTkFrame(_adv_body, fg_color="transparent")
            _ls_row.pack(anchor="w", fill="x", padx=(28, 0), pady=(0, 2))
            ctk.CTkLabel(_ls_row, text="α:", font=font_sm(),
                         text_color=COLORS["muted"], width=20).pack(side="left")
            _ls_eps_var = getattr(app, "label_smoothing_eps", None)
            _ls_disp = tk.StringVar(value=f"{_ls_eps_var.get():.2f}" if _ls_eps_var else "0.05")
            def _on_ls_eps(val, _dv=_ls_disp):
                _dv.set(f"{float(val):.2f}")
            _ls_sl = ctk.CTkSlider(
                _ls_row, from_=0.0, to=0.3, variable=_ls_eps_var,
                command=_on_ls_eps, height=18,
                progress_color=COLORS["accent"], button_color=COLORS["accent2"],
            )
            _ls_sl.pack(side="left", fill="x", expand=True, padx=(4, 4))
            ctk.CTkLabel(_ls_row, textvariable=_ls_disp, width=36,
                         font=font_sm(), text_color=COLORS["fg"]).pack(side="left")

    # ── 5. Запуск + прогресс + лог ──────────────────────────────────
    run_card = Card(scroll, title="Запуск обучения")
    run_card.pack(fill="x", padx=20, pady=(0, 20))

    actions_row = ctk.CTkFrame(run_card.body, fg_color="transparent")
    actions_row.pack(fill="x", pady=(0, 8))
    app.btn_train = ctk.CTkButton(
        actions_row, text="▶  Обучить / сохранить модель", width=240,
        fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
        font=font_md_bold(),
        command=getattr(app, "run_training", lambda: None),
    )
    app.btn_train.pack(side="left", padx=(0, 8))
    Tooltip(app.btn_train,
            "F5 или Ctrl+Enter — обучить модель на загруженных файлах.\n"
            "Модель сохраняется в папку models/ после обучения.")
    app.btn_train_stop = ctk.CTkButton(
        actions_row, text="⏹ Стоп", width=90,
        fg_color="transparent", border_width=1, border_color=COLORS["border2"],
        state="disabled",
        command=getattr(app, "_request_cancel", lambda: None),
    )
    app.btn_train_stop.pack(side="left")
    Tooltip(app.btn_train_stop, "Остановить обучение после текущего шага.")

    ph_row = ctk.CTkFrame(run_card.body, fg_color="transparent")
    ph_row.pack(fill="x", pady=(0, 4))
    app.train_phase_lbl = ctk.CTkLabel(
        ph_row,
        textvariable=getattr(app, "train_status", None),
        font=font_mono(), text_color=COLORS["accent2"], anchor="w",
    )
    app.train_phase_lbl.pack(side="left", fill="x", expand=True)
    ctk.CTkLabel(ph_row, textvariable=getattr(app, "train_pct", None),
                 font=font_mono(), text_color=COLORS["muted"]).pack(side="right")

    app.train_progress_pb = ctk.CTkProgressBar(run_card.body, height=8,
                                                progress_color=COLORS["accent"])
    app.train_progress_pb.set(0)
    app.train_progress_pb.pack(fill="x", pady=(0, 6))

    # Слежение за DoubleVar → обновление CTkProgressBar
    _train_progress_var = getattr(app, "train_progress", None)
    if _train_progress_var is not None:
        def _update_train_pb(*_):
            try:
                app.train_progress_pb.set(
                    min(1.0, _train_progress_var.get() / 100.0))
            except Exception:
                pass
        _train_progress_var.trace_add("write", _update_train_pb)

    se_row = ctk.CTkFrame(run_card.body, fg_color="transparent")
    se_row.pack(fill="x", pady=(0, 8))
    app.train_speed_lbl = ctk.CTkLabel(
        se_row, textvariable=getattr(app, "train_speed", None),
        font=font_sm(), text_color=COLORS["muted"], anchor="w",
    )
    app.train_speed_lbl.pack(side="left", padx=(0, 16))
    app.train_eta_lbl = ctk.CTkLabel(
        se_row, textvariable=getattr(app, "train_eta", None),
        font=font_sm(), text_color=COLORS["muted"], anchor="w",
    )
    app.train_eta_lbl.pack(side="left")

    app.train_log = ctk.CTkTextbox(
        run_card.body, height=220, font=font_mono(),
        fg_color=COLORS["bg"], text_color=COLORS["muted"],
        wrap="none", state="disabled",
    )
    app.train_log.pack(fill="x")

    return scroll


# ─────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────

def _sbert_models_list() -> list:
    try:
        from config.user_config import SBERT_MODELS_LIST
        return SBERT_MODELS_LIST
    except Exception:
        return ["ai-forever/ru-en-RoSBERTa", "ai-forever/rubert-tiny2"]


def _setfit_models_list() -> list:
    try:
        from config.user_config import SETFIT_MODELS_LIST
        return SETFIT_MODELS_LIST
    except Exception:
        return ["ai-forever/ru-en-RoSBERTa"]


def _field_label(parent, text: str):
    ctk.CTkLabel(parent, text=text.upper(), font=font_label(),
                 text_color=COLORS["muted"], anchor="w").pack(anchor="w", pady=(0, 4))


def _separator(parent):
    ctk.CTkFrame(parent, height=0, fg_color="transparent").pack(fill="x", pady=8)


class _FileListBox:
    """Обёртка над ttk.Treeview с API, совместимым с tk.Listbox (insert/delete/curselection)."""

    def __init__(self, tree: ttk.Treeview, summary_var: tk.StringVar):
        self._tree = tree
        self._items: list[str] = []
        self._summary = summary_var

    # ── Listbox-совместимый API ──────────────────────────────────────
    def insert(self, _pos, item: str) -> None:
        from pathlib import Path as _P
        self._items.append(item)
        fname = _P(item).name
        try:
            nb = _P(item).stat().st_size
            size = f"{nb/1_048_576:.1f} МБ" if nb >= 1_048_576 else f"{nb//1024} КБ"
        except Exception:
            size = "—"
        self._tree.insert("", "end", iid=str(len(self._items) - 1),
                          values=(fname, "—", "—", size))
        self._refresh_summary()

    def delete(self, *args) -> None:
        if len(args) == 2:          # delete(0, "end") — очистить всё
            self._tree.delete(*self._tree.get_children())
            self._items.clear()
        elif len(args) == 1:        # delete(i) — удалить по индексу
            idx = int(args[0])
            children = self._tree.get_children()
            if 0 <= idx < len(children):
                self._tree.delete(children[idx])
            if 0 <= idx < len(self._items):
                self._items.pop(idx)
            self._reindex()
        self._refresh_summary()

    def curselection(self) -> tuple:
        sel = self._tree.selection()
        if not sel:
            return ()
        children = self._tree.get_children()
        return tuple(children.index(s) for s in sel if s in children)

    def get(self, idx):
        return self._items[idx] if 0 <= idx < len(self._items) else ""

    def __len__(self) -> int:
        return len(self._items)

    # ── Внутренние хелперы ──────────────────────────────────────────
    def _reindex(self) -> None:
        for new_i, iid in enumerate(self._tree.get_children()):
            self._tree.item(iid, iid=str(new_i))

    def _refresh_summary(self) -> None:
        n = len(self._items)
        if n == 0:
            self._summary.set("Нет файлов")
        elif n == 1:
            self._summary.set("1 файл")
        elif n <= 4:
            self._summary.set(f"{n} файла")
        else:
            self._summary.set(f"{n} файлов")


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
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x")
    for i, c in enumerate(cells):
        ctk.CTkLabel(f, text=c, font=font_base() if i != 1 else font_mono(),
                     text_color=COLORS["fg"], anchor="w").grid(
            row=0, column=i, sticky="ew", padx=12, pady=6)
    for i in range(len(cells)):
        f.grid_columnconfigure(i, weight=1 if i == 1 else 0)
    ctk.CTkFrame(parent, height=1, fg_color=COLORS["border2"]).pack(fill="x")
