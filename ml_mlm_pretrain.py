# -*- coding: utf-8 -*-
"""
ml_mlm_pretrain — дообучение энкодера через Masked Language Modeling (MLM)
на доменном корпусе банковских диалогов перед основной классификацией.

Цель:
  Русские банковские тексты содержат специфическую лексику (досрочное погашение,
  переводоспособность, акцептование, инкассо, скоринг). Дообучение BERT/RoBERTa
  через MLM на unlabeled-корпусе адаптирует модель к этим словам ДО обучения
  классификационной головы, что даёт +2-4% macro F1 без новой разметки.

Как использовать:
  1. Подготовьте корпус текстов (List[str]) — можно брать все тексты обучающей
     выборки + дополнительные неразмеченные диалоги.
  2. Вызовите pretrain_mlm(), сохраните путь к дообученной модели.
  3. Передайте этот путь как model_name в SetFitClassifier / SBERTVectorizer.

Требования:
  pip install transformers>=4.35 datasets accelerate

Ограничения:
  • MLM требует GPU для разумного времени работы (CPU — очень медленно).
  • Минимальный полезный корпус: ~2 000 текстов, лучше 10 000+.
  • Для SetFit: дообучайте ту же модель, что выбрана как setfit_model.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from app_logger import get_logger

_log = get_logger(__name__)


def is_available() -> bool:
    """Проверяет наличие пакетов transformers и datasets."""
    try:
        import importlib.util
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("datasets") is not None
        )
    except Exception:
        return False


def pretrain_mlm(
    texts: List[str],
    model_name: str = "ai-forever/ru-en-RoSBERTa",
    output_dir: str = "mlm_pretrained",
    *,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 16,
    max_length: int = 256,
    mlm_probability: float = 0.15,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    fp16: bool = True,
    logging_steps: int = 50,
    save_steps: int = 500,
    seed: int = 42,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> str:
    """Дообучает encoder-модель через MLM на доменном корпусе.

    Возвращает путь к директории с дообученной моделью.
    Эту директорию можно передать как model_name в SetFitClassifier.

    Параметры:
        texts                 — список обучающих текстов (min ~2000)
        model_name            — HuggingFace repo id или локальный путь
        output_dir            — куда сохранить дообученную модель
        num_train_epochs      — эпохи MLM-дообучения (3–5 обычно достаточно)
        per_device_batch_size — батч (уменьшите до 8 при OOM)
        max_length            — длина токенизации (256 для диалогов)
        mlm_probability       — доля маскируемых токенов (стандарт: 0.15)
        fp16                  — mixed precision (только GPU)
        log_cb                — callback для логирования
        progress_cb           — callback(pct, status) для прогресс-бара
    """
    if not is_available():
        raise ImportError(
            "MLM дообучение требует пакетов: pip install transformers datasets accelerate\n"
            "После установки перезапустите приложение."
        )

    def _log_msg(msg: str) -> None:
        _log.info(msg)
        if log_cb:
            log_cb(msg)

    def _prog(pct: float, status: str) -> None:
        if progress_cb:
            progress_cb(pct, status)

    _log_msg(f"[MLM pretrain] Загрузка модели {model_name} …")
    _prog(5.0, f"MLM: загрузка {model_name}…")

    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset

    _n = len(texts)
    _log_msg(f"[MLM pretrain] Корпус: {_n} текстов | модель: {model_name}")
    if _n < 500:
        _log_msg(
            "⚠ [MLM pretrain] Корпус менее 500 текстов — эффект дообучения будет минимальным."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    _prog(15.0, "MLM: токенизация…")

    def _tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    dataset = Dataset.from_dict({"text": [str(t) for t in texts]})
    tokenized = dataset.map(_tokenize_fn, batched=True, remove_columns=["text"])
    _log_msg(f"[MLM pretrain] Токенизировано {len(tokenized)} примеров.")
    _prog(25.0, "MLM: подготовка тренера…")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=1,
        seed=seed,
        report_to="none",
        disable_tqdm=True,
    )

    _log_msg(f"[MLM pretrain] Обучение: {num_train_epochs} эпох | batch={per_device_batch_size}")
    _prog(30.0, f"MLM: обучение ({num_train_epochs} эпох)…")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized,
    )
    trainer.train()

    _log_msg(f"[MLM pretrain] Сохранение модели → {out_path}")
    _prog(95.0, "MLM: сохранение…")
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    _log_msg(f"[MLM pretrain] ✅ Готово! Используйте путь: {out_path}")
    _prog(100.0, "MLM: готово")
    return str(out_path)


def estimate_mlm_time_minutes(
    n_texts: int,
    epochs: int = 3,
    has_gpu: bool = True,
) -> Tuple[float, float]:
    """Оценочное время дообучения (мин). Возвращает (мин_оценка, макс_оценка)."""
    # Эмпирика: ~0.1 с/текст/эпоху на GPU, ~1.5 с/текст/эпоху на CPU
    secs_per_text_epoch = 0.1 if has_gpu else 1.5
    total_sec = n_texts * epochs * secs_per_text_epoch
    low = total_sec * 0.7 / 60.0
    high = total_sec * 1.5 / 60.0
    return round(low, 1), round(high, 1)
