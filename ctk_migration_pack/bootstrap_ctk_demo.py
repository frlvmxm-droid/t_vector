# -*- coding: utf-8 -*-
"""
bootstrap_ctk_demo.py — мини-приложение для проверки CustomTkinter UI.

Запуск:
    pip install customtkinter matplotlib
    python bootstrap_ctk_demo.py

Содержит:
    • Боковую навигацию (Обучение / Классификация / Кластеризация)
    • Все три переписанные вкладки
    • Переключатель темы (Dark Teal / Paper / Amber CRT) — наверху
"""
from __future__ import annotations

import customtkinter as ctk

from ui_theme_ctk import apply_theme, COLORS, PALETTES, font_label, font_md_bold
from app_train_view_ctk   import build_train_tab
from app_apply_view_ctk   import build_apply_tab
from app_cluster_view_ctk import build_cluster_tab


# Применяем тему ДО создания root
apply_theme("dark-teal")     # "dark-teal" | "paper" | "amber-crt"


class FakeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BankReasonTrainer · CustomTkinter demo")
        self.geometry("1380x860")
        self.configure(fg_color=COLORS["bg"])

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._current_tab = "train"
        self._build_sidebar()
        self._build_main()

    # ── Sidebar ─────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, fg_color=COLORS["bg"], width=230, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)

        # Бренд
        brand = ctk.CTkFrame(sb, fg_color="transparent")
        brand.pack(fill="x", padx=14, pady=(14, 12))
        ctk.CTkLabel(brand, text="BR", width=32, height=32,
                     font=font_md_bold(), fg_color=COLORS["accent"],
                     text_color="#ffffff", corner_radius=7).pack(side="left", padx=(0, 10))
        text = ctk.CTkFrame(brand, fg_color="transparent")
        text.pack(side="left")
        ctk.CTkLabel(text, text="BankReason", font=font_md_bold(),
                     text_color=COLORS["fg"]).pack(anchor="w")
        ctk.CTkLabel(text, text="TRAINER · RU", font=font_label(),
                     text_color=COLORS["muted"]).pack(anchor="w")

        ctk.CTkFrame(sb, height=1, fg_color=COLORS["border2"]).pack(fill="x", padx=14)

        # Навигация
        ctk.CTkLabel(sb, text="WORKFLOW", font=font_label(),
                     text_color=COLORS["muted"], anchor="w").pack(
            fill="x", padx=20, pady=(14, 4))

        self._nav_buttons = {}
        for key, label in [("train", "Обучение"),
                           ("apply", "Классификация"),
                           ("cluster", "Кластеризация")]:
            b = ctk.CTkButton(
                sb, text=label, anchor="w", height=36,
                fg_color=(COLORS["select"] if key == "train" else "transparent"),
                hover_color=COLORS["hover"],
                text_color=(COLORS["accent2"] if key == "train" else COLORS["muted"]),
                command=lambda k=key: self._switch_tab(k),
            )
            b.pack(fill="x", padx=10, pady=2)
            self._nav_buttons[key] = b

        # Theme switcher (внизу sidebar)
        ctk.CTkFrame(sb, fg_color="transparent").pack(fill="both", expand=True)  # spacer
        ctk.CTkFrame(sb, height=1, fg_color=COLORS["border2"]).pack(fill="x", padx=14)
        ctk.CTkLabel(sb, text="ТЕМА", font=font_label(),
                     text_color=COLORS["muted"], anchor="w").pack(
            fill="x", padx=20, pady=(10, 4))
        theme_seg = ctk.CTkSegmentedButton(
            sb, values=["Teal", "Paper", "CRT"],
            command=self._switch_theme,
        )
        theme_seg.set("Teal")
        theme_seg.pack(fill="x", padx=10, pady=(0, 14))

    def _switch_tab(self, key):
        self._current_tab = key
        for k, b in self._nav_buttons.items():
            b.configure(
                fg_color=(COLORS["select"] if k == key else "transparent"),
                text_color=(COLORS["accent2"] if k == key else COLORS["muted"]),
            )
        if key == "train":
            self._show_train()
        elif key == "apply":
            self._show_apply()
        elif key == "cluster":
            self._show_cluster()

    def _switch_theme(self, label):
        """Переключение темы требует перезапуска окна (CTk ограничение)."""
        mapping = {"Teal": "dark-teal", "Paper": "paper", "CRT": "amber-crt"}
        theme = mapping[label]
        apply_theme(theme)
        # Простейший способ — убить окно и пересоздать
        self.destroy()
        new_app = FakeApp()
        new_app._switch_tab(self._current_tab)
        new_app.mainloop()

    # ── Main pane ───────────────────────────────────────────────────
    def _build_main(self):
        self.main = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self.main, fg_color=COLORS["bg"], height=70,
                              corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        self._title = ctk.CTkLabel(header, text="Обучение модели",
                                   font=ctk.CTkFont(size=22, weight="bold"),
                                   text_color=COLORS["fg"])
        self._title.pack(side="left", padx=28, pady=(20, 0), anchor="sw")

        ctk.CTkFrame(self.main, height=1, fg_color=COLORS["border2"]).grid(
            row=0, column=0, sticky="sew")

        self.content = ctk.CTkFrame(self.main, fg_color=COLORS["bg"], corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self._show_train()

    def _clear(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _show_train(self):
        self._clear()
        self._title.configure(text="Обучение модели")
        build_train_tab(self, self.content)

    def _show_apply(self):
        self._clear()
        self._title.configure(text="Классификация")
        build_apply_tab(self, self.content)

    def _show_cluster(self):
        self._clear()
        self._title.configure(text="Кластеризация")
        build_cluster_tab(self, self.content)

    # ── Заглушки методов app ────────────────────────────────────────
    def add_train_files(self):           print("→ add_train_files()")
    def add_train_folder(self):          print("→ add_train_folder()")
    def show_dataset_stats(self):        print("→ show_dataset_stats()")
    def start_training(self):            print("→ start_training()")
    def pick_model(self):                print("→ pick_model()")
    def pick_apply_file(self):           print("→ pick_apply_file()")
    def start_apply(self):               print("→ start_apply()")
    def export_predictions(self):        print("→ export_predictions()")
    def add_cluster_files(self):         print("→ add_cluster_files()")
    def add_cluster_folder(self):        print("→ add_cluster_folder()")
    def start_cluster(self):             print("→ start_cluster()")
    def export_cluster_results(self):    print("→ export_cluster_results()")
    def _auto_detect_cluster_params(self): print("→ _auto_detect_cluster_params()")


if __name__ == "__main__":
    FakeApp().mainloop()
