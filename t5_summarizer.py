# -*- coding: utf-8 -*-
"""
t5_summarizer.py — T5-суммаризатор для русского текста.

Использует модель UrukHan/t5-russian-summarization (или другую T5-based).
Импорт transformers выполняется лениво — только при первом вызове load(),
чтобы не блокировать запуск приложения при отсутствии пакета.

Требования (устанавливаются вручную):
    pip install transformers torch
    # GPU (опционально):
    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Optional

# Папка для кэша моделей рядом с приложением
_APP_ROOT = Path(__file__).parent
T5_CACHE_DIR = _APP_ROOT / "t5_models"

DEFAULT_T5_MODEL = "UrukHan/t5-russian-summarization"


class T5RussianSummarizer:
    """
    Sklearn-style суммаризатор на базе T5 для русского текста.

    Параметры:
        model_name        — HuggingFace model id (по умолчанию UrukHan/t5-russian-summarization)
        max_input_length  — максимальное число токенов входного текста (truncation)
        max_target_length — максимальное число токенов суммаризации (выхода)
        batch_size        — размер батча при инференсе
        device            — "auto" | "cpu" | "cuda"
        log_cb            — callback(str) для текстовых сообщений прогресса
        progress_cb       — callback(float 0..1, str) для числового прогресса
        cache_dir         — папка локального кэша (по умолчанию t5_models/ рядом с app)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_T5_MODEL,
        max_input_length: int = 512,
        max_target_length: int = 128,
        batch_size: int = 4,
        device: str = "auto",
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.device = device
        self.log_cb = log_cb
        self.progress_cb = progress_cb
        self.cache_dir = str(cache_dir or T5_CACHE_DIR)

        # Модели загружаются лениво при первом вызове load() / summarize()
        self._tokenizer = None
        self._model = None
        self._device_obj = None
        self._load_lock = threading.Lock()

    # ---------------------------------------------------------------------- helpers

    def _log(self, msg: str) -> None:
        if self.log_cb:
            self.log_cb(msg)

    def _prog(self, frac: float, status: str) -> None:
        if self.progress_cb:
            self.progress_cb(max(0.0, min(1.0, frac)), status)

    # ---------------------------------------------------------------------- load

    def load(self) -> None:
        """Ленивая загрузка transformers и весов модели.

        Выбрасывает ImportError если transformers/torch не установлены,
        RuntimeError при сбое загрузки модели.
        Потокобезопасна: повторный вызов из другого потока во время загрузки
        заблокируется на lock и вернётся сразу после завершения первого.
        """
        if self._model is not None:
            return  # быстрый путь без блокировки

        with self._load_lock:
            if self._model is not None:
                return  # другой поток уже загрузил

            # — Проверяем зависимости —
            try:
                from transformers import T5ForConditionalGeneration, AutoTokenizer
                import torch
            except ImportError as e:
                raise ImportError(
                    "Для T5-суммаризации нужны пакеты transformers и torch.\n"
                    "Установите командой:\n"
                    "  pip install transformers torch\n"
                    f"(исходная ошибка: {e})"
                ) from e

            # — Определяем устройство —
            if self.device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                if dev == "cuda":
                    self._log(f"[T5] Авто-выбор → CUDA ({torch.cuda.get_device_name(0)})")
                else:
                    self._log("[T5] Авто-выбор → CPU (CUDA не обнаружена)")
            else:
                dev = self.device
                if dev.startswith("cuda:") and torch.cuda.is_available():
                    try:
                        _idx = int(dev.split(":")[1])
                        self._log(f"[T5] Устройство: {dev} ({torch.cuda.get_device_name(_idx)})")
                    except Exception:
                        pass

            # — Авто-масштаб батча под GPU VRAM (если не задан явно) —
            if dev.startswith("cuda") and self.batch_size <= 4:
                try:
                    from hw_profile import detect as _hw_detect
                    self.batch_size = _hw_detect().t5_batch
                    self._log(f"[T5] Авто-батч для GPU: {self.batch_size}")
                except Exception:
                    self.batch_size = 8

            self._log(f"[T5] Загружаю модель '{self.model_name}' на {dev}…")
            self._log(f"[T5] Кэш моделей: {self.cache_dir}")

            import os
            os.makedirs(self.cache_dir, exist_ok=True)

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
                _load_dtype = None
                if dev.startswith("cuda"):
                    try:
                        _load_dtype = torch.float16
                    except Exception:
                        pass
                self._model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=_load_dtype,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Не удалось загрузить T5-модель '{self.model_name}'.\n"
                    f"Проверьте название модели и интернет-соединение.\n"
                    f"(исходная ошибка: {exc})"
                ) from exc

            self._device_obj = torch.device(dev)
            self._model.to(self._device_obj)
            self._model.eval()
            self._log(f"[T5] Модель загружена ✅  устройство={dev}")

    # ---------------------------------------------------------------------- summarize

    def summarize(self, texts: List[str]) -> List[str]:
        """Суммаризует список текстов, возвращает список строк того же размера.

        Пустые строки возвращаются как есть (без запуска модели).
        Lazy-загружает модель при первом вызове.
        """
        import torch

        self.load()

        n = len(texts)
        if n == 0:
            return []

        results: List[str] = [""] * n
        # Индексы непустых текстов
        nonempty_idx = [i for i, t in enumerate(texts) if t and t.strip()]

        if not nonempty_idx:
            return results

        nonempty_texts = [texts[i] for i in nonempty_idx]
        bs = max(1, self.batch_size)
        n_ne = len(nonempty_texts)
        n_batches = (n_ne + bs - 1) // bs

        self._log(
            f"[T5] Суммаризую {n_ne} текстов из {n} "
            f"(батч={bs}, {n_batches} итераций)…"
        )

        summarized: List[str] = []

        with torch.no_grad():
            for b_start in range(0, n_ne, bs):
                batch = nonempty_texts[b_start : b_start + bs]

                encoded = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=self.max_input_length,
                    truncation=True,
                    padding=True,
                )
                input_ids = encoded["input_ids"].to(self._device_obj)
                attention_mask = encoded["attention_mask"].to(self._device_obj)

                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_target_length,
                    num_beams=4,
                    early_stopping=True,
                )

                decoded = self._tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                summarized.extend(decoded)

                done_batches = b_start // bs + 1
                frac = done_batches / n_batches
                self._prog(frac, f"T5: {b_start + len(batch)}/{n_ne}")

        # Расставляем результаты по исходным позициям
        for pos, orig_i in enumerate(nonempty_idx):
            results[orig_i] = summarized[pos] if pos < len(summarized) else ""

        self._log("[T5] Суммаризация завершена ✅")
        return results
