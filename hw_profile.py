# -*- coding: utf-8 -*-
"""
hw_profile.py — определение характеристик ПК и расчёт рекомендуемых
параметров для LinearSVC-Calibrated.

Никогда не бросает исключений: при любой ошибке возвращает безопасные
значения по умолчанию (8 ГБ ОЗУ, 4 ядра, без GPU).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Низкоуровневые определялки
# ---------------------------------------------------------------------------

def _get_ram_gb() -> float:
    """Полный объём ОЗУ в ГБ."""
    # 1) psutil (самый надёжный, если установлен)
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    # 2) /proc/meminfo (Linux / WSL)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    # 3) ctypes / GlobalMemoryStatusEx (Windows без psutil)
    try:
        import ctypes

        class _MEMSTATEX(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        ms = _MEMSTATEX()
        ms.dwLength = ctypes.sizeof(ms)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    return 8.0   # безопасный fallback


def _get_gpu_vram_gb() -> Optional[float]:
    """VRAM первого GPU в ГБ, или None если GPU недоступен."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return None


def _get_gpu_name() -> Optional[str]:
    """Название первого GPU, или None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def _get_gpu_count() -> int:
    """Количество доступных CUDA-устройств."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _get_gpu_names() -> List[str]:
    """Список названий всех GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _get_gpu_compute_capability() -> tuple:
    """CUDA compute capability первого GPU как (major, minor), или (0, 0)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return (0, 0)
        props = torch.cuda.get_device_properties(0)
        return (props.major, props.minor)
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# Профиль и рекомендации
# ---------------------------------------------------------------------------

@dataclass
class HWProfile:
    ram_gb: float
    cpu_cores: int
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_compute_major: int = 0
    gpu_compute_minor: int = 0

    # --- Рекомендованные параметры ---

    @property
    def max_features(self) -> int:
        """Макс. признаков TF-IDF: главный потребитель RAM при обучении."""
        if self.ram_gb >= 32:   return 300_000
        if self.ram_gb >= 16:   return 200_000
        if self.ram_gb >= 8:    return 150_000
        if self.ram_gb >= 4:    return 100_000
        return 60_000

    @property
    def chunk(self) -> int:
        """Размер батча при классификации Excel (CHUNK).
        При каждом flush_chunk() в памяти живёт матрица chunk × max_features.
        """
        if self.ram_gb >= 32:   return 8_000
        if self.ram_gb >= 16:   return 6_000
        if self.ram_gb >= 8:    return 4_000
        if self.ram_gb >= 4:    return 2_000
        return 1_000

    @property
    def sbert_batch(self) -> int:
        """Размер батча SBERTVectorizer.encode().
        На GPU ограничен VRAM; на CPU — выигрыш от параллелизма меньше.
        """
        vram = self.gpu_vram_gb
        if vram is None:          # CPU-режим
            if self.ram_gb >= 16: return 64
            if self.ram_gb >= 8:  return 32
            return 16
        if vram >= 38:  return 2048   # A100 40GB (~39.6 GiB) / A100 80GB / H100
        if vram >= 24:  return 1024   # RTX 3090 / 4090 / A6000
        if vram >= 16:  return 512    # A4000 / RTX 4080
        if vram >= 12:  return 256
        if vram >= 8:   return 128
        if vram >= 4:   return 64
        return 32

    @property
    def t5_batch(self) -> int:
        """Размер батча T5 инференса (суммаризация).
        На GPU ограничен VRAM; на CPU — батч маленький (OOM-риск при beam search).
        """
        vram = self.gpu_vram_gb
        if vram is None:      # CPU-режим
            return 4
        if vram >= 38:  return 256    # A100 40GB (~39.6 GiB) / A100 80GB / H100
        if vram >= 24:  return 128    # RTX 3090 / 4090
        if vram >= 16:  return 64
        if vram >= 8:   return 32
        if vram >= 4:   return 16
        return 4

    @property
    def kmeans_batch(self) -> int:
        """batch_size для MiniBatchKMeans."""
        if self.ram_gb >= 32:   return 8_192
        if self.ram_gb >= 16:   return 4_096
        if self.ram_gb >= 8:    return 2_048
        return 1_024

    @property
    def setfit_batch(self) -> int:
        """Batch size для контрастного обучения SetFit (≠ inference batch).

        Расчитано совместно с max_length=128 (ml_setfit.py) + gradient_checkpointing.
        DeBERTa-v3 attention ∝ seq²: при seq=128 потребление в 16× меньше чем при seq=512.

        Расчёт для A100 40GB, batch=64, max_length=128:
          • Параметры fp16 (86M): 172 MB
          • AdamW optimizer states fp32: 1 032 MB
          • Gradients fp16: 172 MB
          • Dynamic per layer (grad checkpoint): ~80 MB
          • Итого: ~1.5 GB / 39.7 GiB (4%)
        """
        vram = self.gpu_vram_gb
        if vram is None:
            return 2
        if vram >= 38:  return 64   # A100 40GB (~39.6 GiB) — ~2.5 GB при max_length=128
        if vram >= 24:  return 32   # RTX 3090 / 4090 / A6000
        if vram >= 12:  return 16   # A4000 / RTX 4080
        if vram >= 8:   return 8
        if vram >= 4:   return 4
        return 2

    @property
    def transformer_batch(self) -> int:
        """Per-device batch size для дообучения трансформера (Fine-tune)."""
        vram = self.gpu_vram_gb
        if vram is None:
            return 4
        if vram >= 38:  return 64     # A100 40GB (~39.6 GiB) / A100 80GB / H100
        if vram >= 24:  return 32     # RTX 3090 / 4090
        if vram >= 12:  return 16
        if vram >= 8:   return 8
        if vram >= 4:   return 4
        return 2

    @property
    def n_jobs_cv(self) -> int:
        """Число параллельных джобов для cross_val_score.

        На машинах с ≥32 ГБ ОЗУ используем все ядра (-1).
        На меньших машинах ограничиваем половиной ядер, чтобы не исчерпать
        оперативную память при параллельном fit каждого фолда.
        """
        if self.ram_gb >= 32:
            return -1
        return max(1, self.cpu_cores // 2)

    @property
    def gpu_devices(self) -> List[str]:
        """Список явных CUDA-устройств: ['cuda:0', 'cuda:1', ...]"""
        return [f"cuda:{i}" for i in range(self.gpu_count)]

    @property
    def gpu_supports_bf16(self) -> bool:
        """True для GPU с Ampere+ (A100, H100, RTX 30xx/40xx): compute capability >= 8.0."""
        return self.gpu_compute_major >= 8

    @property
    def gpu_supports_tf32(self) -> bool:
        """True для GPU с Ampere+: TF32 ускоряет matmul ~20-30% бесплатно."""
        return self.gpu_compute_major >= 8

    def summary(self) -> str:
        """Однострочное описание железа для отображения в UI."""
        if self.gpu_count >= 2:
            _gn = self.gpu_names[0] if self.gpu_names else (self.gpu_name or "GPU")
            _vr = f"{self.gpu_vram_gb:.0f}" if self.gpu_vram_gb is not None else "?"
            gpu_part = f"  |  GPU: {self.gpu_count}× {_gn} ({_vr} ГБ × {self.gpu_count})"
        elif self.gpu_name and self.gpu_vram_gb is not None:
            gpu_part = f"  |  GPU: {self.gpu_name} ({self.gpu_vram_gb:.0f} ГБ VRAM)"
        else:
            gpu_part = "  |  GPU: нет / не определён"
        return f"ОЗУ: {self.ram_gb:.0f} ГБ  |  CPU: {self.cpu_cores} ядер{gpu_part}"

    def rec_summary(self) -> str:
        """Краткий текст рекомендаций для тултипа."""
        lines = [
            f"Рекомендуемые значения для вашего ПК ({self.ram_gb:.0f} ГБ ОЗУ):",
            f"  • Макс. признаков TF-IDF : {self.max_features:,}",
            f"  • Батч классификации     : {self.chunk:,} строк",
            f"  • Батч SBERT encode      : {self.sbert_batch}",
            f"  • Батч T5 инференс       : {self.t5_batch}",
            f"  • Батч KMeans            : {self.kmeans_batch:,}",
            f"  • Батч SetFit обучение   : {self.setfit_batch}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

_FALLBACK = HWProfile(ram_gb=8.0, cpu_cores=4, gpu_name=None, gpu_vram_gb=None,
                      gpu_count=0, gpu_names=[], gpu_compute_major=0, gpu_compute_minor=0)


def detect() -> HWProfile:
    """Определяет характеристики ПК. Никогда не бросает исключения."""
    try:
        _cc = _get_gpu_compute_capability()
        return HWProfile(
            ram_gb=_get_ram_gb(),
            cpu_cores=os.cpu_count() or 4,
            gpu_name=_get_gpu_name(),
            gpu_vram_gb=_get_gpu_vram_gb(),
            gpu_count=_get_gpu_count(),
            gpu_names=_get_gpu_names(),
            gpu_compute_major=_cc[0],
            gpu_compute_minor=_cc[1],
        )
    except Exception as e:
        import warnings
        warnings.warn(f"hw_profile.detect() failed ({e}), using 8 GB RAM fallback", stacklevel=2)
        return _FALLBACK


def tune_runtime_by_input_size(
    *,
    input_bytes: int,
    chunk: int,
    sbert_batch: int,
    kmeans_batch: int,
) -> dict:
    """Подстраивает runtime-параметры под размер входного файла.

    Это дополняет HWProfile (железо) фактором объёма входных данных.
    """
    size_gb = max(0.0, float(input_bytes) / (1024 ** 3))
    factor = 1.0
    if size_gb >= 2.0:
        factor = 0.4
    elif size_gb >= 1.0:
        factor = 0.6
    elif size_gb >= 0.5:
        factor = 0.8
    return {
        "chunk": max(500, int(chunk * factor)),
        "sbert_batch": max(8, int(sbert_batch * factor)),
        "kmeans_batch": max(256, int(kmeans_batch * factor)),
        "input_size_gb": round(size_gb, 3),
    }
