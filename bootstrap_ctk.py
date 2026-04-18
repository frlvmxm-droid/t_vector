# -*- coding: utf-8 -*-
"""
bootstrap_ctk.py — CustomTkinter-версия главного окна BankReasonTrainer.

Это альтернативный entry-point, запускающий UI из ctk_migration_pack:
    python bootstrap_ctk.py

Старое окно доступно как прежде:
    python bootstrap_run.py   # или python app.py

Структура совпадает с bootstrap_ctk_demo.py (sidebar + контент), но view'ы
подключены из корневых модулей `app_*_view_ctk.py`. Для интеграции с реальной
бизнес-логикой (TrainTabMixin и т.д.) методы обработчиков можно постепенно
переносить сюда — CTk view'ы вызывают `getattr(app, name, lambda: None)`.
"""
from __future__ import annotations

import sys

import customtkinter as ctk

from ui_theme_ctk import apply_theme, COLORS, font_label, font_md_bold
from app_train_view_ctk   import build_train_tab
from app_apply_view_ctk   import build_apply_tab
from app_cluster_view_ctk import build_cluster_tab


apply_theme("dark-teal")


class BankReasonTrainerCTk(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BankReasonTrainer · CustomTkinter")
        self.geometry("1380x860")
        self.configure(fg_color=COLORS["bg"])

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._current_tab = "train"
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, fg_color=COLORS["bg"], width=230, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)

        brand = ctk.CTkFrame(sb, fg_color="transparent")
        brand.pack(fill="x", padx=14, pady=(14, 12))
        ctk.CTkLabel(
            brand, text="BR", width=32, height=32,
            font=font_md_bold(), fg_color=COLORS["accent"],
            text_color="#ffffff", corner_radius=7,
        ).pack(side="left", padx=(0, 10))
        text = ctk.CTkFrame(brand, fg_color="transparent")
        text.pack(side="left")
        ctk.CTkLabel(text, text="BankReason", font=font_md_bold(),
                     text_color=COLORS["fg"]).pack(anchor="w")
        ctk.CTkLabel(text, text="TRAINER · RU", font=font_label(),
                     text_color=COLORS["muted"]).pack(anchor="w")

        ctk.CTkFrame(sb, height=1, fg_color=COLORS["border2"]).pack(fill="x", padx=14)

        ctk.CTkLabel(sb, text="WORKFLOW", font=font_label(),
                     text_color=COLORS["muted"], anchor="w").pack(
            fill="x", padx=20, pady=(14, 4))

        self._nav_buttons: dict[str, ctk.CTkButton] = {}
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

        ctk.CTkFrame(sb, fg_color="transparent").pack(fill="both", expand=True)
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

    def _switch_tab(self, key: str) -> None:
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

    def _switch_theme(self, label: str) -> None:
        mapping = {"Teal": "dark-teal", "Paper": "paper", "CRT": "amber-crt"}
        theme = mapping[label]
        apply_theme(theme)
        self.destroy()
        new_app = BankReasonTrainerCTk()
        new_app._switch_tab(self._current_tab)
        new_app.mainloop()

    def _build_main(self) -> None:
        self.main = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.main, fg_color=COLORS["bg"], height=70,
                              corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        self._title = ctk.CTkLabel(
            header, text="Обучение модели",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS["fg"],
        )
        self._title.pack(side="left", padx=28, pady=(20, 0), anchor="sw")

        ctk.CTkFrame(self.main, height=1, fg_color=COLORS["border2"]).grid(
            row=0, column=0, sticky="sew")

        self.content = ctk.CTkFrame(self.main, fg_color=COLORS["bg"], corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self._show_train()

    def _clear(self) -> None:
        for w in self.content.winfo_children():
            w.destroy()

    def _show_train(self) -> None:
        self._clear()
        self._title.configure(text="Обучение модели")
        build_train_tab(self, self.content)

    def _show_apply(self) -> None:
        self._clear()
        self._title.configure(text="Классификация")
        build_apply_tab(self, self.content)

    def _show_cluster(self) -> None:
        self._clear()
        self._title.configure(text="Кластеризация")
        build_cluster_tab(self, self.content)

    def add_train_files(self) -> None:         print("[stub] add_train_files")
    def add_train_folder(self) -> None:        print("[stub] add_train_folder")
    def show_dataset_stats(self) -> None:      print("[stub] show_dataset_stats")
    def start_training(self) -> None:          print("[stub] start_training")
    def pick_model(self) -> None:              print("[stub] pick_model")
    def pick_apply_file(self) -> None:         print("[stub] pick_apply_file")
    def start_apply(self) -> None:             print("[stub] start_apply")
    def export_predictions(self) -> None:      print("[stub] export_predictions")
    def add_cluster_files(self) -> None:       print("[stub] add_cluster_files")
    def add_cluster_folder(self) -> None:      print("[stub] add_cluster_folder")
    def start_cluster(self) -> None:           print("[stub] start_cluster")
    def export_cluster_results(self) -> None:  print("[stub] export_cluster_results")
    def _auto_detect_cluster_params(self) -> None:
        print("[stub] _auto_detect_cluster_params")


def main() -> int:
    try:
        BankReasonTrainerCTk().mainloop()
    except Exception as exc:
        print(f"[ERROR] bootstrap_ctk: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
