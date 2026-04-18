# -*- coding: utf-8 -*-
"""Диалог «История обучений» — читает experiments.jsonl и показывает таблицу.

Вызов:
    from experiment_history_dialog import show_experiment_history
    show_experiment_history(parent_widget)
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from experiment_log import read_experiments

if TYPE_CHECKING:
    pass


def show_experiment_history(parent: tk.Widget, *, last_n: int = 50) -> None:
    """Открывает модальный диалог с историей экспериментов обучения."""
    records = read_experiments(last_n=last_n)

    dlg = tk.Toplevel(parent)
    dlg.title("История обучений")
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(True, True)
    dlg.geometry("980x460")

    # ── Заголовок ────────────────────────────────────────────────────────────
    hdr = ttk.Label(dlg,
                    text=f"Последние {last_n} запусков обучения  "
                         f"(~/.classification_tool/experiments.jsonl)",
                    font=("TkDefaultFont", 9),
                    foreground="#666666")
    hdr.pack(anchor="w", padx=12, pady=(8, 0))

    if not records:
        ttk.Label(dlg, text="История пуста — обучите хотя бы одну модель.",
                  font=("TkDefaultFont", 10)).pack(expand=True)
        ttk.Button(dlg, text="Закрыть", command=dlg.destroy).pack(pady=8)
        return

    # ── Treeview ─────────────────────────────────────────────────────────────
    cols = ("timestamp", "macro_f1", "accuracy", "n_train", "n_test",
            "train_mode", "C", "use_smote", "use_lemma", "model_file")
    col_labels = {
        "timestamp":  "Дата",
        "macro_f1":   "F1 macro",
        "accuracy":   "Accuracy",
        "n_train":    "Обучение",
        "n_test":     "Тест",
        "train_mode": "Режим",
        "C":          "C",
        "use_smote":  "SMOTE",
        "use_lemma":  "Лемма",
        "model_file": "Файл модели",
    }
    col_widths = {
        "timestamp": 140, "macro_f1": 75, "accuracy": 75,
        "n_train": 70, "n_test": 60, "train_mode": 70, "C": 55,
        "use_smote": 55, "use_lemma": 55, "model_file": 280,
    }

    frm = ttk.Frame(dlg)
    frm.pack(fill="both", expand=True, padx=8, pady=(4, 0))

    vsb = ttk.Scrollbar(frm, orient="vertical")
    hsb = ttk.Scrollbar(frm, orient="horizontal")
    tree = ttk.Treeview(frm, columns=cols, show="headings",
                        yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                        selectmode="browse")
    vsb.configure(command=tree.yview)
    hsb.configure(command=tree.xview)

    for col in cols:
        tree.heading(col, text=col_labels[col],
                     command=lambda c=col: _sort_column(tree, c, False))
        tree.column(col, width=col_widths[col], minwidth=50, anchor="w")

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    # ── Заполнение данными (новые сверху) ────────────────────────────────────
    for rec in reversed(records):
        params = rec.get("params") or {}
        model_short = rec.get("model_file", "")
        # Показываем только имя файла, без полного пути
        try:
            import pathlib
            model_short = pathlib.Path(model_short).name
        except Exception:
            pass

        f1_val = rec.get("macro_f1")
        f1_str = f"{f1_val:.4f}" if f1_val is not None else "—"
        acc_val = rec.get("accuracy")
        acc_str = f"{acc_val:.4f}" if acc_val is not None else "—"

        ts = str(rec.get("timestamp", ""))[:16]  # "2024-01-15T09:30"
        ts_pretty = ts.replace("T", "  ")

        row = (
            ts_pretty,
            f1_str,
            acc_str,
            str(rec.get("n_train") or "—"),
            str(rec.get("n_test") or "—"),
            str(params.get("train_mode") or "—"),
            str(params.get("C") or "—"),
            "✓" if params.get("use_smote") else "✗",
            "✓" if params.get("use_lemma") else "✗",
            model_short,
        )
        iid = tree.insert("", "end", values=row)

        # Подсвечиваем лучший F1 зелёным
        if f1_val is not None and f1_val >= 0.85:
            tree.item(iid, tags=("good",))
        elif f1_val is not None and f1_val < 0.65:
            tree.item(iid, tags=("warn",))

    tree.tag_configure("good", background="#e8f5e9")
    tree.tag_configure("warn", background="#fff3e0")

    # ── Строка статистики ─────────────────────────────────────────────────────
    f1_values = [r["macro_f1"] for r in records if r.get("macro_f1") is not None]
    if f1_values:
        stat = (f"Всего запусков: {len(records)}  |  "
                f"Лучший F1: {max(f1_values):.4f}  |  "
                f"Средний F1: {sum(f1_values)/len(f1_values):.4f}  |  "
                f"Последний F1: {f1_values[-1]:.4f}")
    else:
        stat = f"Всего запусков: {len(records)}"

    stat_lbl = ttk.Label(dlg, text=stat, font=("TkDefaultFont", 9), foreground="#444")
    stat_lbl.pack(anchor="w", padx=12, pady=(4, 0))

    # ── Кнопки ───────────────────────────────────────────────────────────────
    btn_frm = ttk.Frame(dlg)
    btn_frm.pack(fill="x", padx=8, pady=(6, 8))

    ttk.Button(btn_frm, text="Обновить",
               command=lambda: _refresh(tree, records, cols, last_n)).pack(side="left", padx=(0, 6))
    ttk.Button(btn_frm, text="Закрыть", command=dlg.destroy).pack(side="right")


def _sort_column(tree: ttk.Treeview, col: str, reverse: bool) -> None:
    """Сортировка по клику на заголовок столбца."""
    data = [(tree.set(iid, col), iid) for iid in tree.get_children("")]
    try:
        data.sort(key=lambda t: float(t[0].replace("—", "0").replace("✓", "1").replace("✗", "0")),
                  reverse=reverse)
    except ValueError:
        data.sort(key=lambda t: t[0], reverse=reverse)
    for idx, (_, iid) in enumerate(data):
        tree.move(iid, "", idx)
    tree.heading(col, command=lambda: _sort_column(tree, col, not reverse))


def _refresh(tree: ttk.Treeview, _records: list, cols: tuple, last_n: int) -> None:
    """Перечитывает лог и перерисовывает таблицу (кнопка «Обновить»)."""
    fresh = read_experiments(last_n=last_n)
    for iid in tree.get_children(""):
        tree.delete(iid)
    for rec in reversed(fresh):
        params = rec.get("params") or {}
        try:
            import pathlib
            model_short = pathlib.Path(rec.get("model_file", "")).name
        except Exception:
            model_short = str(rec.get("model_file", ""))
        f1_val = rec.get("macro_f1")
        row = (
            str(rec.get("timestamp", ""))[:16].replace("T", "  "),
            f"{f1_val:.4f}" if f1_val is not None else "—",
            f"{rec.get('accuracy'):.4f}" if rec.get("accuracy") is not None else "—",
            str(rec.get("n_train") or "—"),
            str(rec.get("n_test") or "—"),
            str(params.get("train_mode") or "—"),
            str(params.get("C") or "—"),
            "✓" if params.get("use_smote") else "✗",
            "✓" if params.get("use_lemma") else "✗",
            model_short,
        )
        tree.insert("", "end", values=row)
