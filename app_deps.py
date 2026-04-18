# -*- coding: utf-8 -*-
"""
app_deps.py — DepsTabMixin: вкладка «Зависимости».

Показывает:
  • Python-пакеты (core + optional) с версиями и кнопками установки
  • SBERT/DeBERTa модели с размерами и кнопками скачивания

Все установки и скачивания логируются в встроенный лог.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from typing import Dict, List, Tuple

import tkinter as tk
from tkinter import ttk

from config import SBERT_MODELS
from ml_core import SBERT_LOCAL_DIR
from config.ml_constants import hf_cache_key
from ui_theme import FG, ENTRY_BG, ACCENT, SUCCESS, ERROR, WARNING
from ui_widgets import Tooltip
from app_logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Описание пакетов: (pip_name, import_name, install_args, описание)
# ---------------------------------------------------------------------------

CORE_PACKAGES: List[Tuple[str, str, List[str], str]] = [
    ("pandas",       "pandas",   ["pandas>=2.1"],       "Чтение / запись Excel и CSV"),
    ("openpyxl",     "openpyxl", ["openpyxl>=3.1"],     "Запись Excel (.xlsx)"),
    ("scikit-learn", "sklearn",  ["scikit-learn>=1.4"], "ML: TF-IDF, LinearSVC, KMeans"),
    ("joblib",       "joblib",   ["joblib>=1.3"],        "Сохранение / загрузка моделей"),
    ("Pillow",       "PIL",      ["Pillow>=10.0"],       "Фоновое изображение GUI"),
    ("numpy",        "numpy",    ["numpy>=1.24"],        "Матричные операции"),
    ("psutil",       "psutil",   ["psutil>=5.9"],        "Определение RAM / CPU"),
]

OPTIONAL_PACKAGES: List[Tuple[str, str, List[str], str]] = [
    ("sentence-transformers", "sentence_transformers",
     ["sentence-transformers"],
     "SBERT / DeBERTa эмбеддинги — обязателен для SBERT-режима"),
    ("torch (CPU)", "torch",
     ["torch", "torchvision", "torchaudio"],
     "PyTorch CPU — нужен для sentence-transformers и T5"),
    ("transformers", "transformers",
     ["transformers>=4.40,<4.52"],
     "HuggingFace Transformers — нужен для T5-суммаризации"),
    ("pymorphy2", "pymorphy2",
     ["pymorphy2"],
     "Лемматизация русского текста (+4–10% F1)"),
    ("umap-learn", "umap",
     ["umap-learn"],
     "UMAP — снижение размерности для кластеризации"),
    ("hdbscan", "hdbscan",
     ["hdbscan"],
     "HDBSCAN — плотностная кластеризация"),
    ("setfit", "setfit",
     ["setfit>=0.9"],
     "SetFit — нейросетевой few-shot классификатор (HuggingFace) ⭐ +4–10% F1"),
]

# Актуальные размеры и рекомендации VRAM для моделей из таблицы сравнения.
_MODEL_SIZES: Dict[str, str] = {
    "intfloat/multilingual-e5-base":           "1.11 GB",
    "intfloat/multilingual-e5-large-instruct": "1.12 GB",
    "deepvk/USER-bge-m3":                      "1.44 GB",
    "ai-forever/ru-en-RoSBERTa":               "1.61 GB",
    "ai-forever/sbert_large_nlu_ru":           "1.71 GB",
    "cointegrated/rubert-tiny2":               "118 MB",
    "intfloat/multilingual-e5-large":          "2.24 GB",
    "BAAI/bge-m3":                             "2.27 GB",
    "deepvk/USER-base":                        "503 MB",
    "cointegrated/LaBSE-en-ru":                "516 MB",
    "deepvk/USER2-base":                       "596 MB",
    "deepvk/USER-large":                       "-",
    "deepvk/USER2-large":                      "-",
}

_MODEL_VRAM: Dict[str, str] = {
    "intfloat/multilingual-e5-base":           "4 GB",
    "intfloat/multilingual-e5-large-instruct": "6 GB",
    "deepvk/USER-bge-m3":                      "4 GB",
    "ai-forever/ru-en-RoSBERTa":               "4 GB",
    "ai-forever/sbert_large_nlu_ru":           "4 GB",
    "cointegrated/rubert-tiny2":               "1 GB",
    "intfloat/multilingual-e5-large":          "6 GB",
    "BAAI/bge-m3":                             "6 GB",
    "deepvk/USER-base":                        "2 GB",
    "cointegrated/LaBSE-en-ru":                "2 GB",
    "deepvk/USER2-base":                       "2 GB",
    "deepvk/USER-large":                       "-",
    "deepvk/USER2-large":                      "-",
}


class DepsTabMixin:
    """Методы вкладки «Зависимости»."""

    # ------------------------------------------------------------------ build

    def _check_and_install_deps(self):
        """Проверяет CORE-пакеты и предлагает установить недостающие.

        Для CTK-вкладки «Зависимости», где нет полноценной таблицы пакетов.
        Устанавливает в фоновом потоке через pip; сообщение о завершении —
        через messagebox.
        """
        import importlib.util as ilu
        from tkinter import messagebox as _mbox

        missing: List[Tuple[str, List[str]]] = []
        for pip_name, import_name, install_args, _desc in CORE_PACKAGES + OPTIONAL_PACKAGES:
            if ilu.find_spec(import_name) is None:
                missing.append((pip_name, install_args))

        if not missing:
            _mbox.showinfo("Зависимости",
                           "Все пакеты уже установлены ✅\n\n"
                           "Обязательные + дополнительные — всё на месте.")
            return

        names = "\n".join(f"  • {n}" for n, _ in missing)
        if not _mbox.askyesno(
            "Зависимости",
            f"Не установлено {len(missing)} пакет(ов):\n\n{names}\n\n"
            "Установить сейчас через pip?\n"
            "(После установки требуется перезапуск приложения.)"
        ):
            return

        def _run():
            ok, fail = 0, 0
            for pip_name, args in missing:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade"] + args,
                        capture_output=True, text=True, timeout=600,
                    )
                    if proc.returncode == 0:
                        ok += 1
                    else:
                        fail += 1
                        _log.warning("pip install %s failed: %s", pip_name,
                                     proc.stderr[:500])
                except Exception as e:
                    fail += 1
                    _log.warning("pip install %s raised: %s", pip_name, e)
            self.after(0, lambda: _mbox.showinfo(
                "Зависимости",
                f"Готово: установлено {ok}, ошибок {fail}.\n\n"
                f"Перезапустите приложение для применения изменений."
            ))

        threading.Thread(target=_run, daemon=True).start()

    def _build_deps_tab(self):
        tab = self.tab_deps

        # ── Sub-tab frames ─────────────────────────────────────────────────
        _s0 = ttk.Frame(tab)   # Пакеты
        _s1 = ttk.Frame(tab)   # Модели & Лог

        # ── Обязательные пакеты ────────────────────────────────────────────
        lf_core = ttk.LabelFrame(_s0, text="Обязательные пакеты", padding=10)
        lf_core.pack(fill="x", pady=(0, 10))
        self._deps_core_rows: List[Dict] = []
        self._build_pkg_section(lf_core, CORE_PACKAGES, self._deps_core_rows)

        # ── Дополнительные пакеты ──────────────────────────────────────────
        lf_opt = ttk.LabelFrame(_s0, text="Дополнительные пакеты (опционально)", padding=10)
        lf_opt.pack(fill="x", pady=(0, 10))
        self._deps_opt_rows: List[Dict] = []
        self._build_pkg_section(lf_opt, OPTIONAL_PACKAGES, self._deps_opt_rows)

        # Специальная строка: torch + CUDA
        ttk.Separator(lf_opt, orient="horizontal").pack(fill="x", pady=(8, 6))
        cuda_row = ttk.Frame(lf_opt, style="Card.TFrame")
        cuda_row.pack(fill="x")
        self._deps_cuda_status = tk.StringVar(value="")
        ttk.Label(cuda_row, text="torch + CUDA 12.4 (NVIDIA GPU)", width=28,
                  style="Card.TLabel").pack(side="left")
        self._deps_cuda_status_lbl = ttk.Label(cuda_row, textvariable=self._deps_cuda_status,
                                               style="Card.TLabel")
        self._deps_cuda_status_lbl.pack(side="left", padx=(8, 0))
        self._btn_cuda = ttk.Button(cuda_row, text="⚡ Установить torch+CUDA (cu124)",
                                    command=self._install_torch_cuda)
        self._btn_cuda.pack(side="left", padx=(12, 0))
        btn_cuda = self._btn_cuda
        Tooltip(btn_cuda,
                "Устанавливает PyTorch с CUDA 12.4 (NVIDIA RTX 30xx/40xx).\n"
                "Заменяет CPU-версию torch на GPU-сборку.\n\n"
                "Эквивалент:\n"
                "  pip install --force-reinstall torch torchvision torchaudio\n"
                "      --index-url https://download.pytorch.org/whl/cu124\n\n"
                "После установки — перезапустите приложение.")

        # ── SBERT / DeBERTa модели ────────────────────────────────────────
        lf_models = ttk.LabelFrame(
            _s1,
            text="SBERT / DeBERTa модели (скачиваются из HuggingFace Hub)",
            padding=10,
        )
        lf_models.pack(fill="x", pady=(0, 10))
        self._deps_model_rows: List[Dict] = []
        self._build_model_section(lf_models)

        # ── Кнопки управления ─────────────────────────────────────────────
        btn_bar = ttk.Frame(_s1)
        btn_bar.pack(fill="x", pady=(0, 6))
        btn_refresh = ttk.Button(btn_bar, text="↻ Обновить статус",
                                 command=self._refresh_deps_status)
        btn_refresh.pack(side="left")
        Tooltip(btn_refresh, "Повторно проверить, какие пакеты и модели установлены.\n"
                             "После установки пакета статус обновляется автоматически,\n"
                             "но IMPORT новых пакетов работает только после перезапуска.")
        ttk.Label(btn_bar,
                  text="  Перезапустите приложение после установки пакетов.",
                  style="Muted.TLabel").pack(side="left")

        # ── Лог ──────────────────────────────────────────────────────────
        ttk.Label(_s1, text="Лог установки / скачивания:", style="Card.TLabel").pack(
            anchor="w", pady=(4, 2))
        log_frame = ttk.Frame(_s1)
        log_frame.pack(fill="both", expand=True)
        vsb = ttk.Scrollbar(log_frame, orient="vertical")
        vsb.pack(side="right", fill="y")
        self.deps_log = tk.Text(
            log_frame, height=12, wrap="word",
            bg=ENTRY_BG, fg=FG, insertbackground=FG,
            relief="flat", padx=8, pady=6,
            font=("Courier New", 9),
            yscrollcommand=vsb.set,
            state="disabled",
        )
        vsb.configure(command=self.deps_log.yview)
        self.deps_log.pack(side="left", fill="both", expand=True)

        # ── Register bottom sub-tab strip for Deps tab ───────────────────────
        self._register_sub_tabs(
            3,
            ["Пакеты", "Модели & Лог"],
            [_s0, _s1],
        )

        # Первоначальная проверка статусов в фоне
        self.after(300, self._refresh_deps_status)

    # ---------------------------------------------------------------- builders

    def _build_pkg_section(self, parent, packages, rows_out):
        """Строит таблицу пакетов с кнопками установки."""
        # Заголовок
        hdr = ttk.Frame(parent, style="Card.TFrame")
        hdr.pack(fill="x")
        col_widths = [26, 20, 12, 46]
        for i, text in enumerate(["Пакет", "Статус", "Версия", "Назначение"]):
            ttk.Label(hdr, text=text, width=col_widths[i],
                      style="Card.TLabel").grid(row=0, column=i, sticky="w")
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(4, 6))

        for pip_name, import_name, install_args, desc in packages:
            row_frame = ttk.Frame(parent, style="Card.TFrame")
            row_frame.pack(fill="x", pady=2)

            status_var = tk.StringVar(value="Проверка…")
            version_var = tk.StringVar(value="—")

            ttk.Label(row_frame, text=pip_name, width=26,
                      style="Card.TLabel").grid(row=0, column=0, sticky="w")
            status_lbl = ttk.Label(row_frame, textvariable=status_var, width=20,
                                   style="Card.TLabel")
            status_lbl.grid(row=0, column=1, sticky="w")
            ttk.Label(row_frame, textvariable=version_var, width=12,
                      style="Card.Muted.TLabel").grid(row=0, column=2, sticky="w")
            ttk.Label(row_frame, text=desc, width=46,
                      style="Card.Muted.TLabel").grid(row=0, column=3, sticky="w")

            btn = ttk.Button(
                row_frame, text="Установить",
                command=lambda pn=pip_name, ia=install_args, sv=status_var, vv=version_var:
                    self._install_package(pn, ia, sv, vv),
            )
            btn.grid(row=0, column=4, sticky="w", padx=(10, 0))

            rows_out.append({
                "pip_name":    pip_name,
                "import_name": import_name,
                "status_var":  status_var,
                "version_var": version_var,
                "status_lbl":  status_lbl,
                "btn":         btn,
            })

    def _build_model_section(self, parent):
        """Строит таблицу SBERT-моделей с кнопками скачивания."""
        hdr = ttk.Frame(parent, style="Card.TFrame")
        hdr.pack(fill="x")
        col_widths = [36, 16, 10, 14, 52]
        for i, text in enumerate(
            ["Модель (HuggingFace ID)", "Статус", "Размер", "VRAM", "Краткое описание"]
        ):
            ttk.Label(hdr, text=text, width=col_widths[i],
                      style="Card.TLabel").grid(row=0, column=i, sticky="w")
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=(4, 6))

        for model_id, desc in SBERT_MODELS.items():
            row_frame = ttk.Frame(parent, style="Card.TFrame")
            row_frame.pack(fill="x", pady=2)

            status_var = tk.StringVar(value="Проверка…")
            size_str = _MODEL_SIZES.get(model_id, "~?? МБ")
            vram_str = _MODEL_VRAM.get(model_id, "-")
            short_desc = desc

            ttk.Label(row_frame, text=model_id, width=36,
                      style="Card.TLabel").grid(row=0, column=0, sticky="w")
            status_lbl = ttk.Label(row_frame, textvariable=status_var, width=16,
                                   style="Card.TLabel")
            status_lbl.grid(row=0, column=1, sticky="w")
            ttk.Label(row_frame, text=size_str, width=10,
                      style="Card.Muted.TLabel").grid(row=0, column=2, sticky="w")
            ttk.Label(row_frame, text=vram_str, width=14,
                      style="Card.Muted.TLabel").grid(row=0, column=3, sticky="w")
            lbl_desc = ttk.Label(row_frame, text=short_desc, width=52,
                                 style="Card.Muted.TLabel")
            lbl_desc.grid(row=0, column=4, sticky="w")
            Tooltip(lbl_desc, short_desc)

            btn = ttk.Button(
                row_frame, text="Скачать",
                command=lambda mid=model_id, sv=status_var:
                    self._download_sbert_model(mid, sv),
            )
            btn.grid(row=0, column=5, sticky="w", padx=(10, 0))

            self._deps_model_rows.append({
                "model_id":   model_id,
                "status_var": status_var,
                "status_lbl": status_lbl,
                "btn":        btn,
            })

    # ----------------------------------------------------------------- status

    def _refresh_deps_status(self):
        """Проверяет все пакеты и модели в фоновом потоке."""
        def _run():
            import importlib.util as ilu
            try:
                import importlib.metadata as ilm
                def _ver(pip_name):
                    # Pillow устанавливается как "Pillow", но metadata может быть "pillow"
                    for name in (pip_name, pip_name.lower(), pip_name.split()[0]):
                        try:
                            return ilm.version(name)
                        except Exception as _e:
                            _log.debug("_ver(%s): %s", name, _e)
                    return None
            except ImportError:
                def _ver(_):
                    return None

            for row in self._deps_core_rows + self._deps_opt_rows:
                # torch (CPU) → import_name = "torch"
                imp = row["import_name"]
                pname = row["pip_name"].split()[0]  # "torch (CPU)" → "torch"
                spec = ilu.find_spec(imp)
                ok = spec is not None
                ver = _ver(pname) if ok else None
                sv, vv, btn = row["status_var"], row["version_var"], row["btn"]
                lbl = row["status_lbl"]
                if ok:
                    self.after(0, lambda sv=sv: sv.set("✅ Установлен"))
                    self.after(0, lambda vv=vv, ver=ver: vv.set(ver or "?"))
                    self.after(0, lambda btn=btn: btn.configure(state="disabled"))
                    self.after(0, lambda lbl=lbl: lbl.configure(foreground=SUCCESS))
                else:
                    self.after(0, lambda sv=sv: sv.set("❌ Не установлен"))
                    self.after(0, lambda vv=vv: vv.set("—"))
                    self.after(0, lambda btn=btn: btn.configure(state="normal"))
                    self.after(0, lambda lbl=lbl: lbl.configure(foreground=ERROR))

            # torch+CUDA статус
            cuda_ok = False
            torch_spec = ilu.find_spec("torch")
            if torch_spec is not None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        cuda_ok = True
                        self.after(0, lambda: self._deps_cuda_status.set(
                            f"✅ CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}"))
                        self.after(0, lambda: self._deps_cuda_status_lbl.configure(
                            foreground=SUCCESS))
                        self.after(0, lambda: self._btn_cuda.configure(state="disabled"))
                    else:
                        self.after(0, lambda: self._deps_cuda_status.set(
                            "⚠️  torch установлен, но CUDA недоступна (CPU-версия)"))
                        self.after(0, lambda: self._deps_cuda_status_lbl.configure(
                            foreground=WARNING))
                        self.after(0, lambda: self._btn_cuda.configure(state="normal"))
                except (ImportError, RuntimeError, AttributeError) as _e:
                    _log.debug("torch CUDA probe failed: %s", _e)
                    self.after(0, lambda: self._deps_cuda_status.set(
                        "⚠️  torch установлен, CUDA не определена"))
                    self.after(0, lambda: self._deps_cuda_status_lbl.configure(
                        foreground=WARNING))
                    self.after(0, lambda: self._btn_cuda.configure(state="normal"))
            else:
                self.after(0, lambda: self._deps_cuda_status.set(
                    "❌ torch не установлен"))
                self.after(0, lambda: self._deps_cuda_status_lbl.configure(
                    foreground=ERROR))
                self.after(0, lambda: self._btn_cuda.configure(state="normal"))

            # Модели
            for row in self._deps_model_rows:
                mid = row["model_id"]
                cached = self._is_model_cached(mid)
                sv, btn, lbl = row["status_var"], row["btn"], row["status_lbl"]
                if cached:
                    self.after(0, lambda sv=sv: sv.set("✅ Скачана"))
                    self.after(0, lambda btn=btn: btn.configure(state="disabled"))
                    self.after(0, lambda lbl=lbl: lbl.configure(foreground=SUCCESS))
                else:
                    self.after(0, lambda sv=sv: sv.set("❌ Не скачана"))
                    self.after(0, lambda btn=btn: btn.configure(state="normal"))
                    self.after(0, lambda lbl=lbl: lbl.configure(foreground=ERROR))

        threading.Thread(target=_run, daemon=True).start()

    def _is_model_cached(self, model_id: str) -> bool:
        hf_key = hf_cache_key(model_id)
        local = SBERT_LOCAL_DIR / hf_key
        try:
            return local.exists() and any(local.iterdir())
        except OSError as _e:
            _log.debug("_is_model_cached(%s): %s", model_id, _e)
            return False

    # ------------------------------------------------------------------- log

    def _log_deps(self, msg: str):
        """Добавляет строку в лог зависимостей (thread-safe через after)."""
        def _append():
            self.deps_log.configure(state="normal")
            self.deps_log.insert("end", msg + "\n")
            self.deps_log.see("end")
            self.deps_log.configure(state="disabled")
        self.after(0, _append)

    # -------------------------------------------------------- package install

    def _install_package(self, pip_name: str, install_args: List[str],
                         status_var: tk.StringVar, version_var: tk.StringVar):
        """Устанавливает пакет через pip со стримингом вывода в лог."""
        self._log_deps(f"\n{'='*60}")
        self._log_deps(f"[pip] pip install --upgrade {' '.join(install_args)}")
        self.after(0, lambda: status_var.set("Устанавливаю…"))

        def _run():
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", "--upgrade"] + install_args,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log_deps(line)
                proc.wait(timeout=600)
                if proc.returncode == 0:
                    try:
                        import importlib.metadata as ilm
                        pname = pip_name.split()[0]
                        ver = ilm.version(pname)
                    except (ImportError, Exception) as _e:
                        _log.debug("post-install version(%s): %s", pip_name, _e)
                        ver = "?"
                    self.after(0, lambda: status_var.set("✅ Установлен"))
                    self.after(0, lambda v=ver: version_var.set(v))
                    self._log_deps(
                        f"[pip] ✅ {pip_name} установлен (v{ver}).\n"
                        f"[pip] Перезапустите приложение для применения изменений.")
                else:
                    self.after(0, lambda: status_var.set("❌ Ошибка"))
                    self._log_deps(f"[pip] ❌ Ошибка установки (код {proc.returncode})")
            except subprocess.TimeoutExpired:
                self.after(0, lambda: status_var.set("❌ Timeout"))
                self._log_deps(
                    f"[pip] ❌ Timeout. Выполните вручную:\n"
                    f"       pip install --upgrade {' '.join(install_args)}")
            except Exception as ex:
                self.after(0, lambda sv=status_var, e=str(ex): sv.set(f"❌ Ошибка"))
                self._log_deps(f"[pip] ❌ {ex}")

        threading.Thread(target=_run, daemon=True).start()

    def _install_torch_cuda(self):
        """Устанавливает torch+CUDA 12.4."""
        self._log_deps(f"\n{'='*60}")
        self._log_deps("[CUDA] pip install --force-reinstall torch torchvision torchaudio "
                       "--index-url https://download.pytorch.org/whl/cu124")
        self.after(0, lambda: self._deps_cuda_status.set("Устанавливаю torch+CUDA…"))

        def _run():
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install",
                     "--force-reinstall",
                     "torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cu124"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._log_deps(line)
                proc.wait(timeout=900)
                if proc.returncode == 0:
                    self.after(0, lambda: self._deps_cuda_status.set(
                        "✅ torch+CUDA установлен. Перезапустите приложение."))
                    self._log_deps("[CUDA] ✅ Установлено. Перезапустите приложение.")
                else:
                    self.after(0, lambda: self._deps_cuda_status.set(
                        f"❌ Ошибка (код {proc.returncode})"))
                    self._log_deps(f"[CUDA] ❌ Ошибка (код {proc.returncode})")
            except subprocess.TimeoutExpired:
                self.after(0, lambda: self._deps_cuda_status.set("❌ Timeout"))
                self._log_deps("[CUDA] ❌ Timeout. Выполните вручную:\n"
                               "  pip install --force-reinstall torch torchvision torchaudio "
                               "--index-url https://download.pytorch.org/whl/cu124")
            except Exception as ex:
                self.after(0, lambda: self._deps_cuda_status.set("❌ Ошибка"))
                self._log_deps(f"[CUDA] ❌ {ex}")

        threading.Thread(target=_run, daemon=True).start()

    # ---------------------------------------------------------- model download

    def _download_sbert_model(self, model_id: str, status_var: tk.StringVar):
        """Скачивает SBERT-модель из HuggingFace Hub."""
        import importlib.util as ilu
        if ilu.find_spec("sentence_transformers") is None:
            self.after(0, lambda: status_var.set("❌ Установите sentence-transformers"))
            self._log_deps(
                f"[SBERT] ❌ sentence-transformers не установлен.\n"
                f"[SBERT] Сначала установите его (кнопка выше), затем перезапустите приложение.")
            return

        self.after(0, lambda: status_var.set("Скачивание…"))
        self._log_deps(f"\n{'='*60}")
        self._log_deps(f"[SBERT] Скачиваю модель: {model_id}")
        self._log_deps(f"[SBERT] Папка кэша: {SBERT_LOCAL_DIR}")

        def _run():
            import pathlib
            import time as _time
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                self.after(0, lambda: status_var.set("❌ huggingface_hub не найден"))
                self._log_deps("[SBERT] ❌ huggingface_hub не найден. "
                               "Установите sentence-transformers.")
                return

            cache_path = pathlib.Path(str(SBERT_LOCAL_DIR))
            cache_path.mkdir(parents=True, exist_ok=True)

            # Поток-наблюдатель: каждые 5 сек логирует прогресс
            _stop = threading.Event()
            t0 = _time.time()

            def _watcher():
                while not _stop.wait(timeout=5):
                    try:
                        total_mb = sum(
                            f.stat().st_size
                            for f in cache_path.rglob("*") if f.is_file()
                        ) / 1024 / 1024
                        elapsed = int(_time.time() - t0)
                        self._log_deps(
                            f"[SBERT] Скачивание… {elapsed}с | ~{total_mb:.0f} МБ на диске")
                    except OSError as _e:
                        _log.debug("download progress watcher: %s", _e)

            wt = threading.Thread(target=_watcher, daemon=True)
            wt.start()
            try:
                snapshot_download(
                    model_id,
                    cache_dir=str(SBERT_LOCAL_DIR),
                    ignore_patterns=["*.h5", "*.ot", "flax_model*", "tf_model*",
                                     "rust_model*", "onnx*"],
                )
                self.after(0, lambda: status_var.set("✅ Скачана"))
                self._log_deps(
                    f"[SBERT] ✅ {model_id} — скачана и готова к использованию.")
            except Exception as ex:
                self.after(0, lambda: status_var.set("❌ Ошибка"))
                self._log_deps(f"[SBERT] ❌ {ex}")
            finally:
                _stop.set()

        threading.Thread(target=_run, daemon=True).start()
