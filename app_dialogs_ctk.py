# -*- coding: utf-8 -*-
"""
app_dialogs_ctk.py — модальные диалоги приложения Hearsy на CustomTkinter.

Диалоги:
  ModalDialog   — базовый класс: CTkToplevel с заголовком и scrollable body
  HistoryDialog — история экспериментов (train / apply / cluster)
  ArtifactsDialog — браузер .joblib-артефактов
  SettingsDialog — зависимости Python + LLM-ключи
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import tkinter as tk
import tkinter.ttk as ttk
import customtkinter as ctk

from ui_theme_ctk import COLORS, font_label, font_sm, font_base, font_md_bold, font_mono
from app_train_view_ctk import Card


# ─────────────────────────────────────────────────────────────────────────────
# ModalDialog — базовый класс
# ─────────────────────────────────────────────────────────────────────────────

class ModalDialog(ctk.CTkToplevel):
    """
    Базовый тёмный CTkToplevel: заголовок + кнопка × + прокручиваемое тело.
    Подклассы заполняют self.body.
    """

    def __init__(self, parent, title: str, width: int = 900, height: int = 640):
        super().__init__(parent)
        self.title(title)
        self.configure(fg_color=COLORS["bg"])
        self.geometry(f"{width}x{height}")
        self.minsize(min(700, width), min(500, height))
        self.resizable(True, True)

        # Header
        hdr = ctk.CTkFrame(self, fg_color=COLORS["panel"], height=52, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text=title.upper(), font=font_md_bold(),
                     text_color=COLORS["fg"]).pack(side="left", padx=20, pady=14)
        ctk.CTkButton(
            hdr, text="✕", width=36, height=36,
            fg_color="transparent", hover_color=COLORS["hover"],
            text_color=COLORS["muted"], font=font_md_bold(),
            command=self.destroy,
        ).pack(side="right", padx=10, pady=8)

        ctk.CTkFrame(self, height=1, fg_color=COLORS["border2"]).pack(fill="x")

        self.body = ctk.CTkScrollableFrame(
            self, fg_color=COLORS["bg"],
            scrollbar_fg_color=COLORS["panel"],
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent3"],
        )
        self.body.pack(fill="both", expand=True)

    def show(self) -> None:
        self.grab_set()
        self.focus_set()
        self.lift()
        self.wait_window()


# ─────────────────────────────────────────────────────────────────────────────
# HistoryDialog
# ─────────────────────────────────────────────────────────────────────────────

class HistoryDialog(ModalDialog):
    """История экспериментов из experiment_log.py."""

    def __init__(self, parent):
        super().__init__(parent, "История экспериментов", width=1000, height=620)
        self._build()

    def _build(self) -> None:
        # Toolbar
        toolbar = ctk.CTkFrame(self.body, fg_color="transparent")
        toolbar.pack(fill="x", pady=(12, 8))
        ctk.CTkLabel(toolbar, text="Записи обучений, применений и кластеризаций",
                     font=font_sm(), text_color=COLORS["muted"]).pack(side="left")
        ctk.CTkButton(
            toolbar, text="↻ Обновить", width=100,
            fg_color="transparent", border_width=1, border_color=COLORS["border2"],
            font=font_sm(), command=self._load,
        ).pack(side="right")

        # Treeview
        tree_host = tk.Frame(self.body, bg=COLORS["bg"])
        tree_host.pack(fill="both", expand=True)

        cols = ("dt", "type", "status", "f1", "n_train", "n_test", "model")
        self._tree = ttk.Treeview(tree_host, columns=cols, show="headings", height=18)
        col_defs = [
            ("dt",      "Дата/Время",    160),
            ("type",    "Тип",            80),
            ("status",  "Статус",          90),
            ("f1",      "Macro F1",        80),
            ("n_train", "N train",         80),
            ("n_test",  "N test",          70),
            ("model",   "Модель",         300),
        ]
        for cid, heading, width in col_defs:
            self._tree.heading(cid, text=heading, anchor="w")
            self._tree.column(cid, width=width, anchor="w")

        vsb = ttk.Scrollbar(tree_host, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._tree.bind("<Double-Button-1>", self._on_row_dblclick)
        self._entries: list = []
        self._load()

    def _load(self) -> None:
        for row in self._tree.get_children():
            self._tree.delete(row)
        try:
            from experiment_log import read_experiments
            entries = read_experiments(last_n=200)
        except Exception:
            entries = []

        self._entries = list(reversed(entries[-200:]))
        for e in self._entries:
            dt     = str(e.get("timestamp", e.get("trained_at", "")))[:19]
            kind   = e.get("type", e.get("train_mode", "train"))
            status = "✅ OK" if not e.get("error") else "❌ ERR"
            f1_val = e.get("macro_f1") or (e.get("eval_metrics") or {}).get("macro_f1")
            f1     = f"{f1_val:.3f}" if f1_val else "—"
            ntr    = str(e.get("n_train", (e.get("eval_metrics") or {}).get("n_train", "—")))
            nte    = str(e.get("n_test",  (e.get("eval_metrics") or {}).get("n_test",  "—")))
            mdl    = str(e.get("model_file", e.get("model_path", e.get("model", "—"))))
            self._tree.insert("", "end", values=(dt, kind, status, f1, ntr, nte, mdl))

    def _on_row_dblclick(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        if idx >= len(self._entries):
            return
        entry = self._entries[idx]

        pop = ctk.CTkToplevel(self)
        pop.title("Детали эксперимента")
        pop.geometry("600x480")
        pop.configure(fg_color=COLORS["bg"])
        pop.grab_set()
        pop.focus_set()

        txt = ctk.CTkTextbox(pop, fg_color=COLORS["entry"], text_color=COLORS["fg"],
                             font=("Courier New", 12), wrap="word")
        txt.pack(fill="both", expand=True, padx=16, pady=16)
        import json as _json
        txt.insert("end", _json.dumps(entry, ensure_ascii=False, indent=2))
        txt.configure(state="disabled")

        ctk.CTkButton(pop, text="Закрыть", width=100,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                      command=pop.destroy).pack(pady=(0, 12))


# Backward-compat alias (старые вызовы _open_artifacts_dialog)
ArtifactsDialog = None  # will be set after ModelsDialog definition


# ─────────────────────────────────────────────────────────────────────────────
# ModelsDialog
# ─────────────────────────────────────────────────────────────────────────────

class ModelsDialog(ModalDialog):
    """
    Список SBERT-эмбеддинговых моделей из конфига: статус скачивания + кнопка.
    Статусы проверяются в фоновом потоке.
    """

    def __init__(self, parent):
        self._app = parent
        super().__init__(parent, "Модели", width=1020, height=640)
        # Заменяем generic scrollable body прямой компоновкой в диалоге
        self.body.pack_forget()
        self._build()

    def _build(self) -> None:
        # ── Toolbar ───────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", padx=20, pady=(12, 8))
        ctk.CTkLabel(
            toolbar,
            text="Эмбеддинговые модели (sentence-transformers / HuggingFace)",
            font=font_sm(), text_color=COLORS["muted"],
        ).pack(side="left")
        ctk.CTkButton(
            toolbar, text="↻ Обновить", width=100,
            fg_color="transparent", border_width=1, border_color=COLORS["border2"],
            font=font_sm(), command=self._reload_status,
        ).pack(side="right")

        # ── Заголовок таблицы ─────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=COLORS["panel2"], corner_radius=0)
        hdr.pack(fill="x")
        for _txt, _w in [("Модель", 280), ("Статус", 140),
                          ("Размер", 80), ("VRAM", 70), ("Описание", 0)]:
            ctk.CTkLabel(hdr, text=_txt.upper(), font=font_label(),
                         text_color=COLORS["muted"], width=_w, anchor="w").pack(
                side="left", padx=(12 if _w else 6, 0), pady=8)

        ctk.CTkFrame(self, height=1, fg_color=COLORS["border2"]).pack(fill="x")

        # ── Прокручиваемый список — занимает всё оставшееся место ────
        self._list_frame = ctk.CTkScrollableFrame(
            self, fg_color=COLORS["bg"],
            scrollbar_fg_color=COLORS["panel"],
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent3"],
        )
        self._list_frame.pack(fill="both", expand=True)

        self._model_rows: list = []
        self._build_rows()
        self.after(200, self._reload_status)

    def _build_rows(self) -> None:
        from config import SBERT_MODELS
        try:
            from app_deps import _MODEL_SIZES, _MODEL_VRAM
        except Exception:
            _MODEL_SIZES = {}
            _MODEL_VRAM = {}

        for model_id, desc in SBERT_MODELS.items():
            status_var = tk.StringVar(value="Проверка…")
            row = ctk.CTkFrame(self._list_frame, fg_color=COLORS["panel"],
                               corner_radius=0)
            row.pack(fill="x", pady=0)
            ctk.CTkFrame(self._list_frame, height=1,
                         fg_color=COLORS["border2"]).pack(fill="x")

            ctk.CTkLabel(row, text=model_id, width=280, anchor="w",
                         font=font_sm(), text_color=COLORS["fg"]).pack(
                side="left", padx=(12, 0), pady=10)

            status_lbl = ctk.CTkLabel(row, textvariable=status_var, width=140,
                                       anchor="w", font=font_sm(),
                                       text_color=COLORS["muted"])
            status_lbl.pack(side="left", padx=4)

            ctk.CTkLabel(row, text=_MODEL_SIZES.get(model_id, "—"), width=80,
                         anchor="w", font=font_mono(),
                         text_color=COLORS["muted"]).pack(side="left", padx=4)
            ctk.CTkLabel(row, text=_MODEL_VRAM.get(model_id, "—"), width=70,
                         anchor="w", font=font_mono(),
                         text_color=COLORS["muted"]).pack(side="left", padx=4)
            btn = ctk.CTkButton(
                row, text="Установить", width=130,
                fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                text_color="#ffffff", font=font_sm(),
                command=lambda mid=model_id, sv=status_var:
                    self._download(mid, sv),
            )
            btn.pack(side="right", padx=12, pady=8)

            ctk.CTkLabel(row, text=desc[:60], anchor="w",
                         font=font_sm(), text_color=COLORS["muted"]).pack(
                side="left", padx=4, fill="x", expand=True)

            self._model_rows.append({
                "model_id": model_id,
                "status_var": status_var,
                "status_lbl": status_lbl,
                "btn": btn,
            })

    def _reload_status(self) -> None:
        def _run():
            for row in self._model_rows:
                mid = row["model_id"]
                cached = getattr(self._app, "_is_model_cached",
                                 lambda _: False)(mid)
                sv, sl, btn = row["status_var"], row["status_lbl"], row["btn"]
                if cached:
                    self.after(0, lambda sv=sv: sv.set("✅ Установлена"))
                    self.after(0, lambda sl=sl: sl.configure(
                        text_color=COLORS["success"]))
                    self.after(0, lambda btn=btn: btn.configure(
                        text="Переустановить", state="normal",
                        fg_color="transparent",
                        border_width=1, border_color=COLORS["border"],
                        text_color=COLORS["muted"]))
                else:
                    self.after(0, lambda sv=sv: sv.set("❌ Не установлена"))
                    self.after(0, lambda sl=sl: sl.configure(
                        text_color=COLORS["error"]))
                    self.after(0, lambda btn=btn: btn.configure(
                        text="Установить", state="normal",
                        fg_color=COLORS["accent"],
                        border_width=0,
                        text_color="#ffffff"))

        threading.Thread(target=_run, daemon=True).start()

    def _download(self, model_id: str, status_var: tk.StringVar) -> None:
        dl_fn = getattr(self._app, "_download_sbert_model", None)
        if dl_fn is not None:
            dl_fn(model_id, status_var)
        else:
            status_var.set("⏳ Загрузка…")


# Backward-compat alias
ArtifactsDialog = ModelsDialog


# ─────────────────────────────────────────────────────────────────────────────
# ExclusionsDialog
# ─────────────────────────────────────────────────────────────────────────────

class ExclusionsDialog(ModalDialog):
    """
    Редактор пользовательских исключений: стоп-слова, шумовые токены, шумовые фразы.
    Загружает из config.exclusions.load_exclusions() и сохраняет через save_exclusions().
    """

    _SECTIONS = [
        ("stop_words",    "Стоп-слова",
         "Слова, которые исключаются из словаря TF-IDF (частые незначимые)"),
        ("noise_tokens",  "Шумовые токены",
         "Одиночные токены-шум (цифры, спецсимволы, буквы)"),
        ("noise_phrases", "Шумовые фразы",
         "Целые фразы, удаляемые до токенизации (шаблоны колл-центра)"),
    ]

    def __init__(self, parent):
        self._app = parent
        super().__init__(parent, "Редактор исключений", width=860, height=660)
        self._build()

    def _build(self) -> None:
        from config import load_exclusions
        self._data: dict = {k: list(v) for k, v in load_exclusions().items()}
        self._listboxes: dict = {}  # key → tk.Listbox
        self._entry_vars: dict = {}  # key → tk.StringVar

        seg = ctk.CTkSegmentedButton(
            self.body,
            values=[s[1] for s in self._SECTIONS],
            font=font_sm(),
            selected_color=COLORS["accent"],
            selected_hover_color=COLORS["accent2"],
            unselected_color=COLORS["panel2"],
            unselected_hover_color=COLORS["hover"],
            text_color=COLORS["fg"],
            fg_color=COLORS["panel2"],
            command=self._switch_section,
        )
        seg.pack(fill="x", padx=0, pady=(12, 10))
        seg.set(self._SECTIONS[0][1])

        self._section_frames: dict = {}
        for key, label, hint in self._SECTIONS:
            frame = ctk.CTkFrame(self.body, fg_color="transparent")
            ctk.CTkLabel(frame, text=hint, font=font_sm(),
                         text_color=COLORS["muted"]).pack(anchor="w", pady=(0, 6))
            self._build_section(frame, key)
            self._section_frames[key] = frame

        # Показываем первую секцию
        self._section_frames[self._SECTIONS[0][0]].pack(fill="both", expand=True)

        # Нижние кнопки
        btns = ctk.CTkFrame(self.body, fg_color="transparent")
        btns.pack(fill="x", pady=(10, 0), side="bottom")
        ctk.CTkButton(
            btns, text="Сохранить", width=120,
            fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
            font=font_sm(), command=self._save,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btns, text="Отмена", width=100,
            fg_color="transparent", border_width=1, border_color=COLORS["border2"],
            font=font_sm(), command=self.destroy,
        ).pack(side="left")
        self._status_var = tk.StringVar()
        ctk.CTkLabel(btns, textvariable=self._status_var,
                     font=font_sm(), text_color=COLORS["success"]).pack(
            side="right", padx=8)

    def _build_section(self, parent, key: str) -> None:
        items = self._data.get(key, [])

        # Listbox в тёмном стиле
        lb_host = ctk.CTkFrame(parent, fg_color=COLORS["entry"],
                                corner_radius=6, border_width=1,
                                border_color=COLORS["border2"])
        lb_host.pack(fill="both", expand=True, pady=(0, 6))

        lb = tk.Listbox(
            lb_host, bg=COLORS["entry"], fg=COLORS["fg"],
            selectbackground=COLORS["select"], selectforeground=COLORS["fg"],
            height=14, selectmode="extended", bd=0, highlightthickness=0,
            relief="flat", font=("Consolas", 11), activestyle="none",
        )
        vsb = tk.Scrollbar(lb_host, command=lb.yview, bg=COLORS["panel"],
                            troughcolor=COLORS["bg"])
        lb.configure(yscrollcommand=vsb.set)
        lb.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        vsb.pack(side="right", fill="y", pady=4)
        self._listboxes[key] = lb

        for item in sorted(items):
            lb.insert("end", item)

        # Строка добавления
        add_row = ctk.CTkFrame(parent, fg_color="transparent")
        add_row.pack(fill="x")
        ev = tk.StringVar()
        self._entry_vars[key] = ev
        entry = ctk.CTkEntry(add_row, textvariable=ev, font=font_mono(),
                              placeholder_text="Добавить новое…")
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        entry.bind("<Return>", lambda e, k=key: self._add_item(k))

        ctk.CTkButton(add_row, text="Добавить", width=100,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
                      font=font_sm(),
                      command=lambda k=key: self._add_item(k)).pack(side="left", padx=(0, 4))
        ctk.CTkButton(add_row, text="Удалить выбранные", width=150,
                      fg_color="transparent", border_width=1,
                      border_color=COLORS["border2"], font=font_sm(),
                      command=lambda k=key: self._delete_items(k)).pack(side="left")

    def _switch_section(self, label: str) -> None:
        key_map = {s[1]: s[0] for s in self._SECTIONS}
        key = key_map.get(label)
        if not key:
            return
        for k, f in self._section_frames.items():
            if k == key:
                f.pack(fill="both", expand=True)
            else:
                f.pack_forget()

    def _add_item(self, key: str) -> None:
        text = self._entry_vars[key].get().strip()
        if not text:
            return
        lb = self._listboxes[key]
        existing = list(lb.get(0, "end"))
        if text not in existing:
            lb.insert("end", text)
        self._entry_vars[key].set("")

    def _delete_items(self, key: str) -> None:
        lb = self._listboxes[key]
        for idx in reversed(lb.curselection()):
            lb.delete(idx)

    def _save(self) -> None:
        from config import save_exclusions
        new_data = {}
        for key, _label, _hint in self._SECTIONS:
            lb = self._listboxes[key]
            new_data[key] = list(lb.get(0, "end"))
        try:
            save_exclusions(new_data)
            # Обновить кэш в app
            excl = getattr(self._app, "_user_exclusions", None)
            if excl is not None:
                excl.update(new_data)
            sw = getattr(self._app, "_custom_stop_words", None)
            if sw is not None:
                sw.clear()
                sw.extend(new_data.get("stop_words", []))
            self._status_var.set("✅ Сохранено")
            self.after(2000, lambda: self._status_var.set(""))
        except Exception as e:
            self._status_var.set(f"❌ {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SettingsDialog
# ─────────────────────────────────────────────────────────────────────────────

class SettingsDialog(ModalDialog):
    """Диалог настроек: вкладки «Зависимости» и «LLM-ключи»."""

    def __init__(self, parent):
        self._app = parent
        super().__init__(parent, "Настройки", width=960, height=680)
        self._build()

    def _build(self) -> None:
        seg = ctk.CTkSegmentedButton(
            self.body,
            values=["Зависимости", "LLM-ключи"],
            font=font_sm(),
            selected_color=COLORS["accent"],
            selected_hover_color=COLORS["accent2"],
            unselected_color=COLORS["panel2"],
            unselected_hover_color=COLORS["hover"],
            text_color=COLORS["fg"],
            fg_color=COLORS["panel2"],
            command=self._switch_tab,
        )
        seg.pack(fill="x", padx=0, pady=(12, 14))
        seg.set("Зависимости")

        self._tab_deps = ctk.CTkFrame(self.body, fg_color="transparent")
        self._tab_llm  = ctk.CTkFrame(self.body, fg_color="transparent")

        self._build_deps_tab()
        self._build_llm_tab()

        self._tab_deps.pack(fill="both", expand=True)

    def _switch_tab(self, val: str) -> None:
        if val == "Зависимости":
            self._tab_llm.pack_forget()
            self._tab_deps.pack(fill="both", expand=True)
        else:
            self._tab_deps.pack_forget()
            self._tab_llm.pack(fill="both", expand=True)

    # ── Вкладка «Зависимости» ────────────────────────────────────────

    def _build_deps_tab(self) -> None:
        from app_deps import CORE_PACKAGES, OPTIONAL_PACKAGES

        # Summary strip
        summary_row = ctk.CTkFrame(self._tab_deps, fg_color=COLORS["panel2"],
                                    corner_radius=8)
        summary_row.pack(fill="x", pady=(0, 12))
        self._summary_lbl = ctk.CTkLabel(
            summary_row, text="Проверка статусов…",
            font=font_sm(), text_color=COLORS["muted"],
        )
        self._summary_lbl.pack(side="left", padx=16, pady=10)
        ctk.CTkButton(
            summary_row, text="↻ Проверить всё", width=130,
            fg_color="transparent", border_width=1, border_color=COLORS["border2"],
            font=font_sm(), command=self._check_all,
        ).pack(side="right", padx=8, pady=6)

        # Package rows
        self._pkg_rows: list = []
        deps_scroll = ctk.CTkScrollableFrame(
            self._tab_deps, fg_color="transparent", height=440,
            scrollbar_fg_color=COLORS["panel"],
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent3"],
        )
        deps_scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(deps_scroll, text="ОБЯЗАТЕЛЬНЫЕ", font=font_label(),
                     text_color=COLORS["muted"]).pack(anchor="w", pady=(0, 4))
        for pn, im, ia, desc in CORE_PACKAGES:
            self._make_pkg_row(deps_scroll, pn, im, ia, desc)

        ctk.CTkLabel(deps_scroll, text="ДОПОЛНИТЕЛЬНЫЕ", font=font_label(),
                     text_color=COLORS["muted"]).pack(anchor="w", pady=(12, 4))
        for pn, im, ia, desc in OPTIONAL_PACKAGES:
            self._make_pkg_row(deps_scroll, pn, im, ia, desc)

        self.after(200, self._check_all)

    def _make_pkg_row(self, parent, pip_name, import_name, install_args, desc) -> None:
        status_var = tk.StringVar(value="…")
        row = ctk.CTkFrame(parent, fg_color=COLORS["panel2"], corner_radius=6)
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=pip_name, width=180, anchor="w",
                     font=font_sm(), text_color=COLORS["fg"]).pack(
            side="left", padx=(10, 0), pady=4)
        status_lbl = ctk.CTkLabel(row, textvariable=status_var, width=140, anchor="w",
                                   font=font_sm(), text_color=COLORS["muted"])
        status_lbl.pack(side="left", padx=4)
        ctk.CTkLabel(row, text=desc[:55], anchor="w",
                     font=font_sm(), text_color=COLORS["muted"]).pack(side="left", padx=4)
        btn = ctk.CTkButton(
            row, text="Установить", width=100,
            fg_color=COLORS["accent3"], hover_color=COLORS["accent"],
            font=font_sm(),
            command=lambda pn=pip_name, ia=install_args, sv=status_var:
                self._install_pkg(pn, ia, sv),
        )
        btn.pack(side="right", padx=8, pady=4)
        self._pkg_rows.append({
            "pip_name": pip_name, "import_name": import_name,
            "install_args": install_args,
            "status_var": status_var, "status_lbl": status_lbl, "btn": btn,
        })

    def _check_all(self) -> None:
        def _run():
            ok_count = miss_count = outdated_count = 0
            try:
                import importlib.metadata as _ilm
                def _ver(name):
                    for n in (name, name.lower(), name.split()[0]):
                        try:
                            return _ilm.version(n)
                        except Exception:
                            pass
                    return None
            except ImportError:
                def _ver(_): return None

            def _req_ver(install_args):
                """Извлекает минимальную версию из install_args[0], напр. 'pandas>=2.1' → '2.1'."""
                import re
                spec = install_args[0] if install_args else ""
                m = re.search(r">=?([\d.]+)", spec)
                return m.group(1) if m else None

            def _is_outdated(installed, required):
                """Сравнивает версии через tuple из int-сегментов."""
                if not installed or not required:
                    return False
                try:
                    def _t(v): return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
                    return _t(installed) < _t(required)
                except Exception:
                    return False

            for row in self._pkg_rows:
                imp = row["import_name"]
                ok = importlib.util.find_spec(imp) is not None
                ver = _ver(row["pip_name"].split()[0]) if ok else None
                req = _req_ver(row.get("install_args", []))
                sv, sl, btn = row["status_var"], row["status_lbl"], row["btn"]
                if ok:
                    if _is_outdated(ver, req):
                        outdated_count += 1
                        self.after(0, lambda sv=sv, v=ver, r=req: sv.set(f"⚠️ {v} (нужна ≥{r})"))
                        self.after(0, lambda sl=sl: sl.configure(text_color=COLORS["warning"]))
                        self.after(0, lambda btn=btn: btn.configure(state="normal", text="Обновить"))
                    else:
                        ok_count += 1
                        self.after(0, lambda sv=sv, v=ver: sv.set(f"✅ {v or '?'}"))
                        self.after(0, lambda sl=sl: sl.configure(text_color=COLORS["success"]))
                        self.after(0, lambda btn=btn: btn.configure(state="disabled"))
                else:
                    miss_count += 1
                    self.after(0, lambda sv=sv: sv.set("❌ Не установлен"))
                    self.after(0, lambda sl=sl: sl.configure(text_color=COLORS["error"]))
                    self.after(0, lambda btn=btn: btn.configure(state="normal", text="Установить"))

            total = ok_count + outdated_count + miss_count
            txt = (f"Установлено: {ok_count} / {total}"
                   f"  ·  Устарело: {outdated_count}"
                   f"  ·  Отсутствует: {miss_count}")
            self.after(0, lambda: self._summary_lbl.configure(text=txt,
                text_color=(COLORS["success"] if miss_count == 0 and outdated_count == 0
                             else COLORS["warning"])))

        threading.Thread(target=_run, daemon=True).start()

    def _install_pkg(self, pip_name: str, install_args: list, status_var: tk.StringVar) -> None:
        def _run():
            try:
                status_var.set("⏳ Установка…")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install"] + install_args,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                status_var.set("✅ Установлен")
            except Exception as e:
                status_var.set(f"❌ {e}")
        threading.Thread(target=_run, daemon=True).start()

    # ── Вкладка «LLM-ключи» ─────────────────────────────────────────

    def _build_llm_tab(self) -> None:
        ctk.CTkLabel(self._tab_llm,
                     text="API-ключи хранятся в ~/.hearsy/credentials.enc (AES-256 Fernet)",
                     font=font_sm(), text_color=COLORS["muted"]).pack(
            anchor="w", pady=(0, 12))

        self._key_entries: dict = {}
        providers = [
            ("openai",    "OpenAI",    "sk-..."),
            ("anthropic", "Anthropic", "sk-ant-..."),
            ("yandex",    "YandexGPT", "yd-..."),
            ("gigachat",  "GigaChat",  "giga-..."),
        ]
        for key_id, label, placeholder in providers:
            card = ctk.CTkFrame(self._tab_llm, fg_color=COLORS["panel2"],
                                corner_radius=8)
            card.pack(fill="x", pady=4)
            ctk.CTkLabel(card, text=label, width=110, anchor="w",
                         font=font_md_bold(), text_color=COLORS["fg"]).pack(
                side="left", padx=14, pady=10)
            entry = ctk.CTkEntry(card, show="•", placeholder_text=placeholder,
                                  font=font_mono(), fg_color=COLORS["entry"],
                                  text_color=COLORS["fg"])
            entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self._key_entries[key_id] = entry

        self._load_keys()

        btn_row = ctk.CTkFrame(self._tab_llm, fg_color="transparent")
        btn_row.pack(fill="x", pady=(14, 0))
        ctk.CTkButton(
            btn_row, text="Сохранить", width=120,
            fg_color=COLORS["accent"], hover_color=COLORS["accent2"],
            font=font_sm(), command=self._save_keys,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row, text="Очистить всё", width=120,
            fg_color="transparent", border_width=1, border_color=COLORS["border2"],
            font=font_sm(), command=self._clear_keys,
        ).pack(side="left")

    def _cred_path(self) -> Path:
        return Path.home() / ".hearsy" / "credentials.enc"

    def _get_fernet(self):
        import os
        from cryptography.fernet import Fernet
        key_str = os.environ.get("LLM_SNAPSHOT_KEY", "")
        if key_str:
            return Fernet(key_str.encode())
        return None

    def _load_keys(self) -> None:
        try:
            import json
            f = self._get_fernet()
            p = self._cred_path()
            if f is None or not p.exists():
                return
            data = json.loads(f.decrypt(p.read_bytes()).decode())
            for key_id, entry in self._key_entries.items():
                val = data.get(key_id, "")
                if val:
                    entry.delete(0, "end")
                    entry.insert(0, val)
        except Exception:
            pass

    def _save_keys(self) -> None:
        try:
            import json
            f = self._get_fernet()
            if f is None:
                from tkinter import messagebox
                messagebox.showwarning("Ключ шифрования",
                    "Установите переменную окружения LLM_SNAPSHOT_KEY для шифрования.")
                return
            data = {k: e.get() for k, e in self._key_entries.items()}
            p = self._cred_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f.encrypt(json.dumps(data).encode()))
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Ошибка", str(e))

    def _clear_keys(self) -> None:
        for entry in self._key_entries.values():
            entry.delete(0, "end")
        try:
            self._cred_path().unlink(missing_ok=True)
        except Exception:
            pass
