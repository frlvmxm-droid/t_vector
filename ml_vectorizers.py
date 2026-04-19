# -*- coding: utf-8 -*-
"""
ml_vectorizers — векторайзеры и фабрики признаков.

Содержит:
  _BuiltinsPatch             — контекстный менеджер patching builtins
  SBERTVectorizer            — Sentence Transformers векторайзер
  PerFieldSBERTVectorizer    — per-field SBERT
  DeBERTaVectorizer          — DeBERTa векторайзер
  make_neural_vectorizer     — фабрика нейросетевых векторайзеров
  find_sbert_in_pipeline     — поиск SBERT в Pipeline
  find_setfit_classifier     — поиск SetFit классификатора
  PhraseRemover, Lemmatizer  — текстовые препроцессоры
  MetaFeatureExtractor       — числовые мета-признаки диалога
  PerFieldVectorizer         — per-field TF-IDF
  make_hybrid_vectorizer     — фабрика гибридного TF-IDF пайплайна
"""
from __future__ import annotations

import math
import re as _re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from constants import APP_ROOT, RUSSIAN_STOP_WORDS, NOISE_PHRASES, NOISE_TOKENS
from config.ml_constants import (
    hf_cache_key,
    SBERT_IMPORT_MAX_RETRIES,
)
from app_logger import get_logger
import ml_compat

_log = get_logger(__name__)

# Папка для локального хранения скачанных SBERT-моделей
SBERT_LOCAL_DIR = APP_ROOT / "sbert_models"


# ---------------------------------------------------------------------------
# pymorphy2 compat-shim для Python 3.13: inspect.getargspec удалён, а
# pymorphy2 продолжает его вызывать. Мы подменяем его совместимым адаптером.
# Патч идемпотентный и защищён блокировкой — корректен при конкурентных
# импортах pymorphy2 из разных потоков.
# ---------------------------------------------------------------------------

import threading as _threading

_GETARGSPEC_SHIM_LOCK = _threading.Lock()
_GETARGSPEC_SHIM_INSTALLED = False


def _install_getargspec_shim() -> None:
    """Устанавливает inspect.getargspec как совместимый адаптер getfullargspec.

    Применяется один раз на процесс. После установки НЕ откатывается —
    pymorphy2 хранит ссылки на MorphAnalyzer, который может вызывать
    getargspec в любой момент. Возврат к NotImplementedError приведёт к
    краху pymorphy2 на отложенных операциях.
    """
    global _GETARGSPEC_SHIM_INSTALLED
    if _GETARGSPEC_SHIM_INSTALLED:
        return
    with _GETARGSPEC_SHIM_LOCK:
        if _GETARGSPEC_SHIM_INSTALLED:
            return
        import inspect
        if not hasattr(inspect, "_getargspec_orig"):
            inspect._getargspec_orig = getattr(inspect, "getargspec", None)

            def _compat_getargspec(func):
                fs = inspect.getfullargspec(func)
                return fs.args, fs.varargs, fs.varkw, fs.defaults

            inspect.getargspec = _compat_getargspec
        _GETARGSPEC_SHIM_INSTALLED = True


def _run_sbert_bootstrap_patches() -> None:
    """Применяет compat-патчи в bootstrap-контексте SBERT (не на import модуля)."""
    ml_compat.apply_early_compat_patches()

class _BuiltinsPatch:
    """Контекстный менеджер — инжектирует имена в builtins и гарантированно убирает их.

    Используется в SBERTVectorizer._ensure_model() для обхода NameError в Python 3.13,
    когда transformers-модули используют torch/nn в аннотациях без явного импорта.

    Пример::

        with _BuiltinsPatch() as bp:
            bp.inject('torch')
            bp.inject('nn')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(...)
        # builtins очищены гарантированно, даже при исключении
    """

    def __init__(self):
        self._injected: list = []

    # ------------------------------------------------------------------
    def inject(self, name: str) -> bool:
        """Ищет *name* в torch-пакете и добавляет в builtins.

        Возвращает True при успехе. Имя запоминается для удаления в __exit__.
        Если имя уже есть в builtins — ничего не делает и возвращает True.
        """
        import builtins as _b
        if hasattr(_b, name):
            return True
        import sys as _s
        import types as _tp
        if name in _s.modules:
            setattr(_b, name, _s.modules[name])
            self._injected.append(name)
            return True
        try:
            import importlib as _il
            _direct = _il.import_module(name)
            setattr(_b, name, _direct)
            self._injected.append(name)
            return True
        except ImportError:
            pass  # прямой import не найден — пробуем источники из torch.*
            _sources = []
            try:
                import torch as _t
                _sources.append(_t)
            except ImportError:
                pass  # torch недоступен — остаются только уже импортированные модули/заглушка
            try:
                import torch.nn as _tnn; _sources.append(_tnn)
            except ImportError:
                pass  # подмодуль недоступен — собираем то, что есть
            try:
                import torch.optim as _topt; _sources.append(_topt)
            except ImportError:
                pass  # подмодуль недоступен — собираем то, что есть
            try:
                import torch.optim.lr_scheduler as _tlrs; _sources.append(_tlrs)
            except ImportError:
                pass  # подмодуль недоступен — собираем то, что есть
            for _src in _sources:
                if hasattr(_src, name):
                    setattr(_b, name, getattr(_src, name))
                    self._injected.append(name)
                    return True
        except AttributeError:
            pass  # API torch изменился — идём к SimpleNamespace fallback (ImportError уже обработан выше)
        # Крайний случай: заглушка SimpleNamespace.
        # ПРЕДУПРЕЖДЕНИЕ: если transformers действительно использует это имя
        # не как аннотацию типа, а как реальный объект — будет AttributeError.
        import types as _tp
        _log.warning(
            "[ml_core] builtins.%s не найден — инжектирован SimpleNamespace-stub. "
            "Если transformers упадёт с AttributeError — обновите torch.",
            name,
        )
        setattr(_b, name, _tp.SimpleNamespace())
        self._injected.append(name)
        return True

    # ------------------------------------------------------------------
    def __enter__(self) -> "_BuiltinsPatch":
        return self

    def __exit__(self, *_exc_info) -> bool:
        """Удаляет все инжектированные имена из builtins."""
        try:
            import builtins as _b
            for _name in self._injected:
                if hasattr(_b, _name):
                    delattr(_b, _name)
        except (AttributeError, TypeError):
            pass  # builtins уже не содержит этого атрибута — ничего очищать не нужно
        return False  # не подавляем исключение


# ---------------------------------------------------------------------------

class SBERTVectorizer:
    """
    Sklearn-совместимый векторайзер на основе Sentence Transformers.

    fit()       — проверяет кэш, при необходимости скачивает, загружает в память.
    transform() — кодирует тексты батчами с прогрессом через log_cb / progress_cb.

    Сериализация (ВАЖНО):
      • В .joblib сохраняются ТОЛЬКО имя модели, batch_size, device, prefix-
        таблица. Сами веса (~400 MB – 2 GB) НЕ пиклятся — это осознанное
        архитектурное решение, иначе bundle будет неподъёмным.
      • При загрузке из .joblib SBERTVectorizer подгружает веса из локального
        кэша HuggingFace (SBERT_LOCAL_DIR / cache_dir). Это означает жёсткое
        требование к deployment-окружению:
          1) либо кэш должен быть физически доступен (bind-mount, hf-cache dir),
          2) либо при первом transform() нужен интернет-доступ к huggingface.co
             для повторной загрузки.
        Если этого не обеспечено, transform() упадёт с OSError / ConnectionError.
      • В airgapped-деплоях рекомендуется заранее выполнить
        `SentenceTransformer(model_name).save(...)` и выставить cache_dir
        на локальную папку.

    Параметры:
        log_cb      — callback(str): вызывается с текстовыми строками прогресса
        progress_cb — callback(float, str): вызывается с (%, статус) как у ui_prog
    """

    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny2",
        batch_size: int = 32,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cache_dir: Optional[Any] = None,
        device: str = "auto",
        progress_range: Tuple[float, float] = (78.0, 90.0),
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.log_cb = log_cb
        self.progress_cb = progress_cb
        # cache_dir: где хранятся модели локально; по умолчанию sbert_models/ рядом с приложением
        self.cache_dir = str(cache_dir or SBERT_LOCAL_DIR)
        # device: "auto" (авто-выбор), "cpu", "cuda" / "gpu"
        self.device = device
        # progress_range: (start%, end%) — диапазон прогресс-бара при кодировании.
        # Передаётся снаружи, чтобы не захардкоживать 78..90% для разных сценариев
        # (обучение, кластеризация, apply).
        self.progress_range = progress_range
        self._model = None  # не сериализуем

    # --- sklearn-compatible interface ---

    def fit(self, X: List[str], y: Any = None) -> "SBERTVectorizer":
        self._ensure_model()
        return self

    # Словарь task-prefix для моделей, которые требуют специального префикса.
    # USER/USER2 → "clustering: ", E5 → "query: ", instruct → развёрнутая инструкция.
    # BGE-M3 и deepvk/USER-bge-m3 не требуют префикса для dense retrieval/clustering.
    _MODEL_TASK_PREFIXES: dict = {
        # deepvk/USER — DeBERTa-based, обучены с prefix-инструкциями
        "deepvk/USER-base":                        "clustering: ",
        "deepvk/USER-large":                       "clustering: ",
        "deepvk/USER2-base":                       "clustering: ",
        "deepvk/USER2-large":                      "clustering: ",
        # RoSBERTa обучена с инструкциями
        "ai-forever/ru-en-RoSBERTa":              "clustering: ",
        # Multilingual E5 — нужен prefix "query: " для encoding
        "intfloat/multilingual-e5-small":          "query: ",
        "intfloat/multilingual-e5-base":           "query: ",
        "intfloat/multilingual-e5-large":          "query: ",
        # E5-instruct — развёрнутая инструкция для тематической кластеризации
        "intfloat/multilingual-e5-large-instruct": "Identify the topic or theme based on the text: ",
        # BGE-M3 — без prefix для симметричного кодирования (clustering)
        # "BAAI/bge-m3": "",  # No prefix needed
        # "deepvk/USER-bge-m3": "",  # No prefix needed
    }

    def _get_task_prefix(self) -> str:
        """Возвращает prefix для текущей модели (пустую строку, если prefix не нужен)."""
        return self._MODEL_TASK_PREFIXES.get(self.model_name, "")

    def transform_stream(self, X: List[str]) -> Iterator[np.ndarray]:
        """Потоковое кодирование: возвращает эмбеддинги по батчам."""
        self._ensure_model()
        prefix = self._get_task_prefix()
        texts = [
            (prefix + str(t)) if (t is not None and prefix) else (str(t) if t is not None else "")
            for t in X
        ]
        n = len(texts)
        bs = self.batch_size
        n_batches = (n + bs - 1) // bs

        _prefix_note = f" (prefix='{prefix}')" if prefix else ""
        self._log(f"[SBERT] Кодирую {n} текстов батчами по {bs} ({n_batches} батч(ей)){_prefix_note}…")

        for i in range(0, n, bs):
            batch = texts[i : i + bs]
            emb = self._model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            yield emb

            done_batches = i // bs + 1
            pct_done = done_batches / n_batches
            if self.progress_cb:
                p0, p1 = self.progress_range
                pct_ui = p0 + pct_done * (p1 - p0)
                self.progress_cb(pct_ui, f"SBERT: {i + len(batch)}/{n} текстов")

        self._log(f"[SBERT] Кодирование завершено ✅")

    def transform_iter(
        self,
        X: List[str],
        writer_cb: Optional[Callable[[np.ndarray, int], None]] = None,
    ) -> Iterator[np.ndarray]:
        """Итератор батчей эмбеддингов с optional writer callback."""
        offset = 0
        for emb in self.transform_stream(X):
            if writer_cb is not None:
                writer_cb(emb, offset)
            yield emb
            offset += emb.shape[0]

    def transform(self, X: List[str], stream: bool = False) -> Union[np.ndarray, Iterator[np.ndarray]]:
        if stream:
            return self.transform_iter(X)
        n = len(X)
        out = None
        offset = 0
        for emb in self.transform_iter(X):
            if out is None:
                out = np.empty((n, emb.shape[1]), dtype=emb.dtype)
            out[offset : offset + emb.shape[0], :] = emb
            offset += emb.shape[0]
        if out is None:
            return np.empty((0, 0), dtype=np.float32)
        return out

    def fit_transform(self, X: List[str], y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # --- internal ---

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)

    def _patch_torch_and_packaging(self) -> None:
        """Патч 1+2: torch.__version__ и packaging.version.parse для Python 3.13.

        torch.__version__ может быть None на некоторых сборках, что ломает
        packaging.version.parse() при импорте transformers. Патч идемпотентен
        (проверяет _none_patched флаг перед применением).
        """
        try:
            import sys as _sys
            _torch_mod = _sys.modules.get("torch")
            if _torch_mod is not None and getattr(_torch_mod, "__version__", None) is None:
                # transformers requires >= 2.4 — "0.0.0" would make is_torch_available()
                # return False and leave AutoModel as None → TypeError on __call__.
                try:
                    import importlib.metadata as _imeta
                    _torch_mod.__version__ = _imeta.version("torch")
                except (ImportError, Exception):  # PackageNotFoundError varies by Python version
                    _torch_mod.__version__ = "2.4.0"
            from packaging import version as _pv
            if not getattr(_pv, "_none_patched", False):
                _orig_parse = _pv.parse
                def _safe_parse(v, *_a, **_kw):
                    return _orig_parse("0.0.0" if v is None else v, *_a, **_kw)
                _pv.parse = _safe_parse
                _OrigVer = _pv.Version
                class _SafeVersion(_OrigVer):
                    def __init__(self, version):
                        super().__init__("0.0.0" if version is None else version)
                _pv.Version = _SafeVersion
                _pv._none_patched = True
                _log.warning(
                    "[ml_core] packaging.version.parse/Version глобально пропатчены "
                    "(torch.__version__ == None — см. SBERTVectorizer._patch_torch_and_packaging). "
                    "Патч применяется один раз и идемпотентен.",
                )
        except (ImportError, AttributeError) as _patch_e:
            _log.debug("_patch_torch_and_packaging: non-critical patch failed: %s", _patch_e)

    def _load_sentence_transformer(self, bp: "_BuiltinsPatch") -> type:
        """Патч 3: загружает SentenceTransformer с retry при NameError (Python 3.13).

        Некоторые файлы transformers используют torch/nn/LRScheduler в аннотациях
        без импорта — Python 3.13 падает с NameError. Стратегия: retry-цикл с
        автоинжекцией каждого недостающего имени в builtins через _BuiltinsPatch.
        bp — активный контекстный менеджер; очистка builtins выполняется в его __exit__.
        """
        def _clear_broken_tf_cache() -> None:
            """Удаляет сломанные transformers/sentence_transformers из sys.modules.

            Нужно если is_torch_available() был закэширован как False
            (из-за torch.__version__==None при первом импорте).
            """
            import sys as _s
            for _k in list(_s.modules.keys()):
                if _k.startswith('sentence_transformers.') or _k == 'sentence_transformers':
                    _s.modules.pop(_k, None)
                elif _k.startswith('transformers.') or _k == 'transformers':
                    _s.modules.pop(_k, None)

        bp.inject('torch')
        bp.inject('nn')

        _SentenceTransformer = None
        for _attempt in range(SBERT_IMPORT_MAX_RETRIES):
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer
                break
            except ModuleNotFoundError as _mnfe:
                if 'sentence_transformers' in str(_mnfe) or 'sentence-transformers' in str(_mnfe):
                    raise ImportError(
                        "Пакет sentence-transformers не установлен.\n"
                        "Установите: pip install sentence-transformers\n"
                        "Или нажмите кнопку «Установить» в разделе SBERT."
                    )
                raise ImportError(
                    f"Ошибка импорта зависимости sentence-transformers: {_mnfe}\n"
                    "Попробуйте: pip install sentence-transformers transformers --upgrade"
                )
            except NameError as _ne:
                _m = _re.search(r"name '(\w+)' is not defined", str(_ne))
                if not _m:
                    raise ImportError(f"NameError при импорте sentence-transformers: {_ne}")
                bp.inject(_m.group(1))
                _clear_broken_tf_cache()
            except ImportError as _ie:
                raise ImportError(
                    f"Ошибка загрузки sentence-transformers: {_ie}\n"
                    "Попробуйте: pip install sentence-transformers transformers --upgrade"
                )
        else:
            raise ImportError(
                f"Не удалось загрузить sentence-transformers после {SBERT_IMPORT_MAX_RETRIES} попыток.\n"
                "Попробуйте: pip install transformers --upgrade"
            )
        return _SentenceTransformer

    def _resolve_device(self) -> Optional[str]:
        """Определяет аргумент device для SentenceTransformer по self.device.

        Возвращает: "cpu", "cuda", "cuda:N", или None (авто-выбор SentenceTransformer).
        """
        _dev = getattr(self, "device", "auto")
        # Явный cuda:N (напр. "cuda:0", "cuda:1") — прокидываем напрямую
        if isinstance(_dev, str) and _dev.startswith("cuda:"):
            try:
                import torch
                _idx = int(_dev.split(":")[1])
                if torch.cuda.is_available() and _idx < torch.cuda.device_count():
                    self._log(f"[SBERT] Устройство: {_dev} — {torch.cuda.get_device_name(_idx)}")
                    return _dev
                self._log(f"[SBERT] ⚠️ {_dev} недоступен — переключаюсь на CPU")
            except (ImportError, RuntimeError, AttributeError) as _e:
                _log.debug("SBERT device probe %r failed: %s", _dev, _e)
            return "cpu"
        if _dev == "cpu":
            self._log("[SBERT] Устройство: CPU (задано явно)")
            return "cpu"
        if _dev in ("cuda", "gpu"):
            try:
                import torch
                if torch.cuda.is_available():
                    self._log(f"[SBERT] Устройство: CUDA — {torch.cuda.get_device_name(0)}")
                    return "cuda"
                self._log("[SBERT] ⚠️ CUDA запрошена, но недоступна — переключаюсь на CPU")
                return "cpu"
            except Exception as _e:
                self._log(f"[SBERT] ⚠️ Ошибка инициализации torch ({_e}) — используется CPU")
                return "cpu"
        # "auto": SentenceTransformer сам выбирает устройство (CUDA → CPU)
        try:
            import torch
            if torch.cuda.is_available():
                self._log(f"[SBERT] Авто-выбор → CUDA ({torch.cuda.get_device_name(0)})")
            else:
                self._log("[SBERT] Авто-выбор → CPU (CUDA не обнаружена)")
        except (ImportError, RuntimeError, AttributeError):
            self._log("[SBERT] Устройство: авто-выбор (SentenceTransformer)")
        return None  # None → SentenceTransformer сам выбирает

    def _load_model_with_tf_patch(self, ST_class: type, device: Optional[str]) -> None:
        """Патч 4: исправляет is_torch_available если transformers ошибочно считает
        torch недоступным (кэш из-за torch.__version__==None), затем создаёт модель.

        Builtins очищает _BuiltinsPatch-контекстный менеджер в _ensure_model (не здесь) —
        _LazyModule внутри SentenceTransformer делает повторные импорты transformers.models.*,
        которые тоже нуждаются в инжектированных именах.
        """
        try:
            import transformers.utils.import_utils as _tfu
            # 4a: Прямой патч is_torch_available
            if not _tfu.is_torch_available():
                _tfu.is_torch_available = lambda: True
                try:
                    import transformers.utils as _tu
                    _tu.is_torch_available = lambda: True
                except ImportError as _e:
                    _log.debug("transformers.utils patch skipped: %s", _e)
                try:
                    import transformers as _tf_mod
                    if hasattr(_tf_mod, 'is_torch_available'):
                        _tf_mod.is_torch_available = lambda: True
                except ImportError as _e:
                    _log.debug("transformers root patch skipped: %s", _e)
            # 4b: Патч BACKENDS_MAPPING
            _bm = getattr(_tfu, 'BACKENDS_MAPPING', None)
            if _bm is not None and 'torch' in _bm:
                _entry = _bm['torch']
                _check_fn = _entry[0] if isinstance(_entry, (list, tuple)) else _entry
                if callable(_check_fn) and not _check_fn():
                    _bm['torch'] = (lambda: True,) + (
                        tuple(_entry[1:]) if isinstance(_entry, (list, tuple)) else ()
                    )
        except (ImportError, AttributeError) as _tf_patch_e:
            _log.debug("_load_model_with_tf_patch: non-critical transformers patch failed: %s", _tf_patch_e)

        self._model = ST_class(self.model_name, cache_folder=self.cache_dir, device=device)
        self._log(f"[SBERT] Модель загружена ✅  (dim={self._model.get_sentence_embedding_dimension()})")
        if self.progress_cb:
            self.progress_cb(76.0, "SBERT готов, обучаю классификатор…")

    def _ensure_model(self):
        if self._model is not None:
            return

        _run_sbert_bootstrap_patches()
        self._patch_torch_and_packaging()

        # _BuiltinsPatch остаётся активным на всё время загрузки модели:
        # _LazyModule внутри SentenceTransformer делает повторные импорты
        # transformers.models.*, которые тоже нуждаются в инжектированных именах.
        # Cleanup гарантирован через __exit__ даже при исключении.
        with _BuiltinsPatch() as _bp:
            SentenceTransformer = self._load_sentence_transformer(_bp)

            # --- 1. Проверить кэш HuggingFace ---
            cached = self._check_cache()
            if cached:
                self._log(f"[SBERT] Модель найдена в кэше ✅")
            else:
                self._log(f"[SBERT] Модель не найдена локально. Скачиваю из HuggingFace Hub…")
                self._log(f"[SBERT] Модель: {self.model_name}")
                self._log(f"[SBERT] Это может занять несколько минут (зависит от размера и скорости интернета).")
                if self.progress_cb:
                    self.progress_cb(51.0, f"Скачивание {self.model_name}…")
                self._download_model()
                self._log(f"[SBERT] Скачивание завершено ✅")

            # --- 2. Загрузить в память ---
            self._log(f"[SBERT] Загружаю модель в память…")
            if self.progress_cb:
                self.progress_cb(64.0, "SBERT: загрузка в память…")

            device = self._resolve_device()
            self._load_model_with_tf_patch(SentenceTransformer, device)

    def _check_cache(self) -> bool:
        """Возвращает True, если модель уже скачана в sbert_models/ или HuggingFace-кэш."""
        import pathlib
        # HuggingFace хранит модели в {cache_dir}/models--{org}--{model}/
        hf_key = hf_cache_key(self.model_name)
        local = pathlib.Path(self.cache_dir) / hf_key
        if local.exists() and any(local.iterdir()):
            return True
        # Проверка глобального кэша HuggingFace (fallback)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(self.model_name, cache_dir=self.cache_dir, local_files_only=True)
            return True
        except EnvironmentError:
            # Нормальная ситуация: модель не в кэше (LocalEntryNotFoundError → EnvironmentError)
            return False
        except Exception as ex:
            # Неожиданная ошибка (права доступа, диск переполнен и т.д.) — логируем
            self._log(f"[SBERT] ⚠️ Ошибка при проверке кэша: {ex}")
            return False

    def _download_model(self):
        """Скачивает модель из HuggingFace Hub в sbert_models/ папку проекта."""
        import pathlib, threading, time as _time
        cache_path = pathlib.Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Фоновый поток: каждые 5 сек логирует суммарный объём скачанного
        _stop = threading.Event()
        def _progress_watcher():
            t0 = _time.time()
            while not _stop.wait(timeout=5):
                try:
                    total_mb = sum(
                        f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
                    ) / 1024 / 1024
                    elapsed = int(_time.time() - t0)
                    self._log(f"[SBERT] Скачивание… {elapsed}с | ~{total_mb:.0f} МБ на диске")
                except OSError:
                    pass  # файл ещё создаётся — ожидаемо в начале скачивания

        watcher = threading.Thread(target=_progress_watcher, daemon=True)
        watcher.start()
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                self.model_name,
                cache_dir=self.cache_dir,
                ignore_patterns=["*.h5", "*.ot", "flax_model*", "tf_model*",
                                  "rust_model*", "onnx*"],
            )
        except (OSError, ValueError, RuntimeError) as e:
            # Expected failure modes:
            #   OSError — network / disk / permission / connection reset
            #   ValueError — malformed repo id or missing required files
            #   RuntimeError — HfHubHTTPError (404 / 401 / 403) subclasses this
            # SentenceTransformer will retry the download on its own below,
            # so we record the reason and fall through.
            self._log(f"[SBERT] ⚠️ Ошибка при скачивании ({type(e).__name__}): {e}")
            self._log(f"[SBERT] Попытка загрузить напрямую через SentenceTransformer…")
        finally:
            _stop.set()

    # --- serialization: не сохраняем веса модели в joblib ---

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["log_cb"] = None
        state["progress_cb"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None
        self.log_cb = None
        self.progress_cb = None
        # Совместимость со старыми моделями без поля device
        if not hasattr(self, "device"):
            self.device = "auto"

    @staticmethod
    def is_available() -> bool:
        """True если sentence-transformers установлен И успешно импортируется.

        Использует find_spec для обнаружения пакета (без импорта), затем
        пробует импорт. Это позволяет различить «не установлен» и «DLL-конфликт».
        """
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            return False          # точно не установлен
        _run_sbert_bootstrap_patches()
        # Перед первым импортом sentence_transformers/transformers:
        # если torch.__version__ всё ещё None (не был исправлен ранее),
        # берём версию из метаданных pip — иначе transformers напечатает
        # "Disabling PyTorch because PyTorch >= 2.4 is required but found None".
        import sys as _sys_ia
        _torch_ia = _sys_ia.modules.get("torch")
        if _torch_ia is not None and getattr(_torch_ia, "__version__", None) is None:
            try:
                import importlib.metadata as _imeta_ia
                _torch_ia.__version__ = _imeta_ia.version("torch")
            except (ImportError, Exception):  # PackageNotFoundError varies by Python version
                _torch_ia.__version__ = "2.4.0"
            # Сбрасываем LRU-кэш is_torch_available, если transformers уже загружен
            try:
                import transformers.utils.import_utils as _tfu_ia
                if hasattr(_tfu_ia.is_torch_available, "cache_clear"):
                    _tfu_ia.is_torch_available.cache_clear()
            except (ImportError, AttributeError) as _e:
                _log.debug("is_torch_available cache_clear skipped: %s", _e)
        # Подавляем предупреждение transformers о версии torch на время первого импорта
        import logging as _logging_ia
        _tf_logger = _logging_ia.getLogger("transformers.utils.import_utils")
        _orig_level = _tf_logger.level
        _tf_logger.setLevel(_logging_ia.ERROR)
        try:
            import sentence_transformers  # noqa: F401
            return True
        except (ImportError, OSError, RuntimeError):
            return False          # установлен, но import падает (DLL-конфликт и т.п.)
        finally:
            _tf_logger.setLevel(_orig_level)

    @staticmethod
    def is_installed() -> bool:
        """True если пакет sentence-transformers физически присутствует в окружении
        (даже если import не работает из-за DLL-конфликта torchvision)."""
        import importlib.util
        return importlib.util.find_spec("sentence_transformers") is not None

    @staticmethod
    def is_cuda_available() -> bool:
        """Проверяет, доступна ли CUDA (GPU через PyTorch)."""
        try:
            import torch
            return torch.cuda.is_available()
        except (ImportError, RuntimeError, AttributeError):
            return False


# ---------------------------------------------------------------------------
# PerFieldSBERTVectorizer — Комбо C
# ---------------------------------------------------------------------------

class PerFieldSBERTVectorizer(BaseEstimator, TransformerMixin):
    """
    SBERT-векторайзер с раздельным кодированием секций (DESC, CLIENT, OPERATOR, …).

    Вместо кодирования одного склеенного текста с тегами [DESC] / [CLIENT]
    каждая секция кодируется независимо одной и той же моделью SBERT.
    Полученные эмбеддинги L2-нормируются, умножаются на вес поля и
    конкатенируются в итоговый вектор.

    Преимущество перед SBERTVectorizer:
      • Теги [DESC], [CLIENT] больше не засоряют эмбеддинг служебным шумом.
      • Каждое поле получает отдельный вектор — модель различает сигналы
        DESC / CLIENT / SUMMARY так же, как PerFieldVectorizer для TF-IDF.
      • Пустые поля → нулевой вектор, не мешают другим секциям.

    Output dim = embedding_dim × n_active_fields.

    Сериализация: веса модели НЕ сохраняются в .joblib (как в SBERTVectorizer).
    """

    _FIELDS: List[Tuple[str, str]] = [
        ("w_desc",         "DESC"),
        ("w_client",       "CLIENT"),
        ("w_operator",     "OPERATOR"),
        ("w_summary",      "SUMMARY"),
        ("w_answer_short", "ANSWER_SHORT"),
        ("w_answer_full",  "ANSWER_FULL"),
    ]
    _TAG_RE = _re.compile(r'^\[([A-Z_]+)\]$')

    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny2",
        base_weights: Optional[Dict[str, int]] = None,
        batch_size: int = 32,
        normalize: bool = True,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cache_dir: Optional[Any] = None,
        device: str = "auto",
        progress_range: Tuple[float, float] = (78.0, 90.0),
    ):
        self.model_name = model_name
        self.base_weights = base_weights or {}
        self.batch_size = batch_size
        self.normalize = normalize
        self.log_cb = log_cb
        self.progress_cb = progress_cb
        self.cache_dir = cache_dir
        self.device = device
        self.progress_range = progress_range
        # Заполняется при fit()
        self._active: List[Tuple[str, str, int]] = []
        self._embedding_dim: Optional[int] = None
        self._sbert: Optional[SBERTVectorizer] = None  # не сериализуется

    # ------------------------------------------------------------------

    def _get_active_fields(self) -> List[Tuple[str, str, int]]:
        active = [
            (wk, tag, int(self.base_weights.get(wk, 0)))
            for wk, tag in self._FIELDS
            if int(self.base_weights.get(wk, 0)) > 0
        ]
        return active if active else [(wk, tag, 1) for wk, tag in self._FIELDS]

    def _parse_fields(self, text: str) -> Dict[str, str]:
        """Разбивает feature-текст на секции по тегам [TAG].
        Берёт только первое вхождение каждого тега (повторы для весов игнорируются).
        """
        result: Dict[str, str] = {}
        current_tag: Optional[str] = None
        buf: List[str] = []
        for line in text.splitlines():
            m = self._TAG_RE.match(line.strip())
            if m:
                tag = m.group(1)
                if tag == "CHANNEL":
                    current_tag = None; buf = []; continue
                if tag in result:
                    current_tag = None; buf = []; continue
                if current_tag and current_tag not in result:
                    result[current_tag] = "\n".join(buf).strip()
                current_tag = tag; buf = []
            elif current_tag:
                buf.append(line)
        if current_tag and current_tag not in result:
            result[current_tag] = "\n".join(buf).strip()
        return result

    def _ensure_sbert(self) -> None:
        """Инициализирует SBERTVectorizer и загружает модель (если ещё не загружена)."""
        if self._sbert is None:
            self._sbert = SBERTVectorizer(
                model_name=self.model_name,
                batch_size=self.batch_size,
                log_cb=self.log_cb,
                progress_cb=self.progress_cb,
                cache_dir=str(self.cache_dir or SBERT_LOCAL_DIR),
                device=self.device,
                progress_range=self.progress_range,
            )
        else:
            # Восстанавливаем колбэки (сбрасываются при десериализации)
            self._sbert.log_cb = self.log_cb
            self._sbert.progress_cb = self.progress_cb
        self._sbert._ensure_model()
        if self._embedding_dim is None:
            self._embedding_dim = self._sbert._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: List[str], y: Any = None) -> "PerFieldSBERTVectorizer":
        self._active = self._get_active_fields()
        self._ensure_sbert()
        self.n_features_in_ = self._embedding_dim * len(self._active)
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        self._ensure_sbert()
        n = len(X)
        dim = self._embedding_dim

        X_list = [str(t) if t is not None else "" for t in X]
        parsed_rows = [self._parse_fields(text) for text in X_list]

        n_fields = len(self._active)
        p0, p1 = self.progress_range
        step = (p1 - p0) / max(n_fields, 1)

        field_matrices: List[np.ndarray] = []
        for field_idx, (wk, tag, weight) in enumerate(self._active):
            field_texts = [row.get(tag, "") for row in parsed_rows]

            non_empty = [i for i, t in enumerate(field_texts) if t.strip()]
            if not non_empty:
                field_matrices.append(np.zeros((n, dim), dtype=np.float32))
                continue

            if self.log_cb:
                self.log_cb(
                    f"[PerFieldSBERT] Поле {tag} ({field_idx + 1}/{n_fields}): "
                    f"{len(non_empty)}/{n} документов…"
                )

            # Кодируем только непустые тексты
            self._sbert.log_cb = self.log_cb
            self._sbert.progress_cb = self.progress_cb
            self._sbert.progress_range = (
                p0 + field_idx * step,
                p0 + (field_idx + 1) * step,
            )
            texts_to_encode = [field_texts[i] for i in non_empty]
            encoded = self._sbert.transform(texts_to_encode).astype(np.float32)

            # L2-нормализация
            if self.normalize:
                norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                encoded /= norms

            # Применяем вес поля
            if weight != 1:
                encoded *= float(weight)

            # Заполняем финальную матрицу (пустые строки остаются нулями)
            field_embs = np.zeros((n, dim), dtype=np.float32)
            for out_idx, orig_idx in enumerate(non_empty):
                field_embs[orig_idx] = encoded[out_idx]

            field_matrices.append(field_embs)

        stacked = np.hstack(field_matrices)

        # Финальная L2-норма после применения per-field весов. Без этого
        # шага поле с весом w=3 раздувает норму конкатенированного вектора в
        # √(w²) раз относительно полей с w=1 → косинусное сравнение между
        # документами с разной комбинацией непустых секций теряет смысл
        # (LinearSVC и KMeans перестают быть инвариантны к магнитуде).
        if self.normalize:
            stacked_norms = np.linalg.norm(stacked, axis=1, keepdims=True)
            stacked_norms = np.where(stacked_norms > 0, stacked_norms, 1.0)
            stacked = stacked / stacked_norms

        return stacked

    def fit_transform(self, X: List[str], y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Сериализация: не сохраняем веса модели в joblib
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sbert"] = None
        state["log_cb"] = None
        state["progress_cb"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._sbert = None
        self.log_cb = None
        self.progress_cb = None
        if not hasattr(self, "device"):
            self.device = "auto"
        if not hasattr(self, "normalize"):
            self.normalize = True


# ---------------------------------------------------------------------------
# DeBERTa Vectorizer (опциональный, требует transformers + torch)
# ---------------------------------------------------------------------------

# Набор идентификаторов моделей DeBERTa-v2 / DeBERTa-v3.
# Используется фабрикой make_neural_vectorizer() для выбора нужного векторайзера.
DEBERTA_MODEL_IDS: frozenset = frozenset({
    # DeBERTa-v2 (Microsoft, English, очень высокое качество NLU)
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge",
    # DeBERTa-v3 (Microsoft, многоязычный, SOTA энкодер 2024+)
    "microsoft/deberta-v3-xsmall",
    "microsoft/deberta-v3-small",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-xlarge",
    # Русские модели на базе DeBERTa
    "deepvk/deberta-v1-base",
})


class DeBERTaVectorizer:
    """
    Sklearn-совместимый векторайзер на базе DeBERTa-v2 / DeBERTa-v3
    через библиотеку transformers.

    Использует mean pooling токен-эмбеддингов с учётом attention mask —
    стандартный и наиболее точный способ получить sentence-level вектор
    из энкодерной модели без специального пулинг-слоя.

    Почему DeBERTa для кластеризации:
      • DeBERTa-v2/v3 — SOTA на GLUE/SuperGLUE, превосходит BERT/RoBERTa.
      • Disentangled attention (позиционные и контентные матрицы раздельно)
        даёт более точное понимание семантики.
      • v3 обучен на 2TB+ текстов с replaced token detection (ELECTRA-like).
      • Хорошо работает на русских текстах через многоязычный режим (mDeBERTa).

    Параметры:
        model_name      — HuggingFace model id (DeBERTa-v2 или v3)
        batch_size      — размер батча (DeBERTa крупный, рекомендуется 8-16)
        max_length      — максимальная длина токенов (512 для v2/v3)
        normalize       — L2-нормализация эмбеддингов (True рекомендуется для KMeans/HDBSCAN)
        log_cb          — callback(str): прогресс-лог строками
        progress_cb     — callback(float, str): прогресс (%, статус)
        cache_dir       — локальный кэш (по умолчанию sbert_models/)
        device          — "auto" / "cpu" / "cuda"
        progress_range  — (start%, end%) диапазон в общей шкале прогресса
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        batch_size: int = 16,
        max_length: int = 512,
        normalize: bool = True,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cache_dir: Optional[Any] = None,
        device: str = "auto",
        progress_range: Tuple[float, float] = (20.0, 90.0),
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.log_cb = log_cb
        self.progress_cb = progress_cb
        self.cache_dir = str(cache_dir or SBERT_LOCAL_DIR)
        self.device = device
        self.progress_range = progress_range
        self._tokenizer = None
        self._model = None
        self._device_obj = None  # torch.device

    # --- sklearn-compatible interface ---

    def fit(self, X: List[str], y: Any = None) -> "DeBERTaVectorizer":
        self._ensure_model()
        return self

    def transform(self, X: List[str]) -> np.ndarray:
        self._ensure_model()
        import torch

        texts = [str(t) if t is not None else "" for t in X]
        n = len(texts)
        bs = self.batch_size
        n_batches = (n + bs - 1) // bs
        out = None
        offset = 0

        self._log(
            f"[DeBERTa] Кодирую {n} текстов батчами по {bs} ({n_batches} батч(ей))…"
        )

        for i in range(0, n, bs):
            batch_texts = texts[i : i + bs]

            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device_obj) for k, v in encoded.items()}

            with torch.no_grad():
                output = self._model(**encoded)

            # Mean pooling с учётом attention mask
            token_emb = output.last_hidden_state  # (B, L, H)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            pooled = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            pooled_np = pooled.cpu().numpy()
            if out is None:
                out = np.empty((n, pooled_np.shape[1]), dtype=pooled_np.dtype)
            out[offset : offset + pooled_np.shape[0], :] = pooled_np
            offset += pooled_np.shape[0]

            done_batches = i // bs + 1
            pct_done = done_batches / n_batches
            if self.progress_cb:
                p0, p1 = self.progress_range
                pct_ui = p0 + pct_done * (p1 - p0)
                self.progress_cb(pct_ui, f"DeBERTa: {i + len(batch_texts)}/{n} текстов")

        self._log("[DeBERTa] Кодирование завершено ✅")
        if out is None:
            return np.empty((0, 0), dtype=np.float32)
        return out

    def fit_transform(self, X: List[str], y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # --- internal ---

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Пакет transformers или torch не установлен.\n"
                "Установите: pip install transformers torch"
            )

        self._log(f"[DeBERTa] Загружаю модель {self.model_name}…")
        if self.progress_cb:
            self.progress_cb(15.0, f"DeBERTa: загрузка {self.model_name}…")

        # Выбор устройства
        _dev = getattr(self, "device", "auto")
        if _dev == "cpu":
            self._device_obj = torch.device("cpu")
            self._log("[DeBERTa] Устройство: CPU (задано явно)")
        elif _dev in ("cuda", "gpu"):
            if torch.cuda.is_available():
                self._device_obj = torch.device("cuda")
                self._log(f"[DeBERTa] Устройство: CUDA — {torch.cuda.get_device_name(0)}")
            else:
                self._device_obj = torch.device("cpu")
                self._log("[DeBERTa] ⚠️ CUDA запрошена, но недоступна — CPU")
        else:
            if torch.cuda.is_available():
                self._device_obj = torch.device("cuda")
                self._log(f"[DeBERTa] Авто-выбор → CUDA ({torch.cuda.get_device_name(0)})")
            else:
                self._device_obj = torch.device("cpu")
                self._log("[DeBERTa] Авто-выбор → CPU")

        cache = self.cache_dir
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache)
        self._model = AutoModel.from_pretrained(self.model_name, cache_dir=cache)
        self._model.eval()
        self._model.to(self._device_obj)

        # Размер эмбеддинга
        _hdim = self._model.config.hidden_size
        self._log(f"[DeBERTa] Модель загружена ✅  (hidden_size={_hdim})")
        if self.progress_cb:
            self.progress_cb(20.0, "DeBERTa готов, кодирую тексты…")

    # --- serialization: не сохраняем веса модели в joblib ---

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_model"] = None
        state["_device_obj"] = None
        state["log_cb"] = None
        state["progress_cb"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = None
        self._model = None
        self._device_obj = None
        self.log_cb = None
        self.progress_cb = None
        if not hasattr(self, "device"):
            self.device = "auto"
        if not hasattr(self, "normalize"):
            self.normalize = True

    @staticmethod
    def is_available() -> bool:
        """True если transformers и torch установлены."""
        import importlib.util
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    @staticmethod
    def is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except (ImportError, RuntimeError, AttributeError):
            return False


# ---------------------------------------------------------------------------
# Фабрика нейронных векторайзеров
# ---------------------------------------------------------------------------

def make_neural_vectorizer(
    model_name: str,
    batch_size: int = 32,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cache_dir: Optional[Any] = None,
    device: str = "auto",
    progress_range: Tuple[float, float] = (20.0, 90.0),
    per_field: bool = False,
    base_weights: Optional[Dict[str, int]] = None,
):
    """
    Фабрика: возвращает нейронный векторайзер по model_name и режиму кодирования.

    Логика выбора:
      • model_name в DEBERTA_MODEL_IDS → DeBERTaVectorizer (per_field не поддерживается)
      • per_field=True  → PerFieldSBERTVectorizer (Combo C: раздельное кодирование секций)
      • per_field=False → SBERTVectorizer (Combo B: весь текст одним вектором)

    per_field=True требует base_weights={'w_desc': N, 'w_client': N, …} для определения
    активных полей. Если base_weights не передан — используются все поля с весом 1.

    Параметры batch_size:
      • DeBERTa-v3-base:  16 (рекомендуется)
      • DeBERTa-v3-large:  8 (крупная модель, память)
      • SBERT/E5/BGE:     32 (стандарт)
    """
    if model_name in DEBERTA_MODEL_IDS:
        _bs = batch_size if batch_size != 32 else 16  # DeBERTa крупнее → меньший батч
        return DeBERTaVectorizer(
            model_name=model_name,
            batch_size=_bs,
            normalize=True,
            log_cb=log_cb,
            progress_cb=progress_cb,
            cache_dir=cache_dir,
            device=device,
            progress_range=progress_range,
        )
    if per_field:
        return PerFieldSBERTVectorizer(
            model_name=model_name,
            base_weights=base_weights or {},
            batch_size=batch_size,
            normalize=True,
            log_cb=log_cb,
            progress_cb=progress_cb,
            cache_dir=cache_dir,
            device=device,
            progress_range=progress_range,
        )
    return SBERTVectorizer(
        model_name=model_name,
        batch_size=batch_size,
        log_cb=log_cb,
        progress_cb=progress_cb,
        cache_dir=cache_dir,
        device=device,
        progress_range=progress_range,
    )


def find_sbert_in_pipeline(pipe) -> "Optional[SBERTVectorizer]":
    """
    Ищет SBERTVectorizer или PerFieldSBERTVectorizer в sklearn Pipeline или FeatureUnion.

    Проверяет:
      1. named_steps["features"] — прямой SBERTVectorizer / PerFieldSBERTVectorizer
      2. transformer_list внутри FeatureUnion в named_steps["features"]

    Возвращает первый найденный экземпляр или None.
    """
    _SBERT_TYPES = (SBERTVectorizer, PerFieldSBERTVectorizer)

    features_step = None
    if hasattr(pipe, "named_steps"):
        features_step = pipe.named_steps.get("features")
    elif hasattr(pipe, "transformer_list"):
        features_step = pipe

    if features_step is None:
        return None
    if isinstance(features_step, _SBERT_TYPES):
        return features_step
    if hasattr(features_step, "transformer_list"):
        for _nm, _t in features_step.transformer_list:
            if isinstance(_t, _SBERT_TYPES):
                return _t
            # SBERT может быть вложен в sub-pipeline внутри FeatureUnion (гибрид)
            if hasattr(_t, "named_steps"):
                for _sub in _t.named_steps.values():
                    if isinstance(_sub, _SBERT_TYPES):
                        return _sub
    return None


def find_setfit_classifier(pipe) -> "Optional[Any]":
    """
    Проверяет, является ли pipe SetFitClassifier (не обёрнутым в sklearn Pipeline).

    SetFit-модели сохраняются в joblib напрямую (без Pipeline-обёртки),
    поэтому функция просто проверяет тип объекта.

    Возвращает объект если это SetFitClassifier, иначе None.
    """
    try:
        from ml_setfit import SetFitClassifier
        if isinstance(pipe, SetFitClassifier):
            return pipe
    except ImportError:
        pass
    return None


# Re-export для удобства импорта из app_train.py / app_apply.py
def _import_train_model_setfit():
    from ml_setfit import train_model_setfit
    return train_model_setfit


class PhraseRemover(BaseEstimator, TransformerMixin):
    """
    Sklearn-совместимый трансформер: удаляет шаблонные фразы regex-заменой до токенизации.

    Принимает список фраз (строк), строит единое регулярное выражение и заменяет
    все совпадения пробелом. Поддерживает joblib-сериализацию через __getstate__/__setstate__:
    скомпилированный паттерн не сохраняется, восстанавливается при первом use.
    """

    def __init__(self, phrases: List[str]):
        self.phrases: List[str] = [p.strip() for p in phrases if p.strip()]
        self._pattern: Optional[_re.Pattern] = None
        self._build()

    def _build(self) -> None:
        if not self.phrases:
            self._pattern = None
            return
        escaped = sorted((_re.escape(p) for p in self.phrases), key=len, reverse=True)
        self._pattern = _re.compile(r"(?:" + "|".join(escaped) + r")", _re.IGNORECASE)

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> List[str]:
        if self._pattern is None:
            return [str(t) if t is not None else "" for t in X]
        return [self._pattern.sub(" ", str(t) if t is not None else "") for t in X]

    def fit_transform(self, X, y=None) -> List[str]:
        return self.fit(X, y).transform(X)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pattern"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build()


class Lemmatizer(BaseEstimator, TransformerMixin):
    """
    Sklearn-совместимый трансформер: лемматизация русского текста.

    Использует pymorphy3 (предпочтительно) или pymorphy2 как fallback.
    pymorphy3 — поддерживаемый форк pymorphy2, совместим с Python 3.12+
    и не требует compat-патча inspect.getargspec.

    Алгоритм:
      1. Разбивает текст на токены (слова ≥ 2 символов из рус/лат алфавита).
      2. Для каждого токена берёт наиболее вероятный лемм.
      3. Нерусские токены (секционные маркеры [TAG], числа) сохраняются как есть.
      4. Собирает токены обратно с пробелами (порядок слов не важен для TF-IDF).

    Преимущество для русского:
      «снял», «снимаю», «снятие», «снять» → всё «снять»
      «банке», «банку», «банком», «банков» → всё «банк»

    include_pos=True — добавляет POS-тег: «снять_VERB», «стоит_VERB» vs «стоит_ADJF».

    Если ни pymorphy3, ни pymorphy2 не установлены — трансформер работает
    как pass-through (no-op). Поддерживает joblib-сериализацию.
    """

    _TOKEN_RE = _re.compile(r'[А-ЯЁа-яёA-Za-z]{2,}')
    _SECTION_RE = _re.compile(r'^\[[A-Z_]+\]$')

    def __init__(self, include_pos: bool = False):
        # include_pos=True — после леммы добавляет «_POS» (напр. «снять_VERB»).
        # Различает части речи для омонимов (стоит_VERB / стоит_ADJF).
        self._morph = None
        self._available = None
        self._backend: str = ""   # "pymorphy3" | "pymorphy2" | ""
        self.include_pos = bool(include_pos)

    def _get_morph(self):
        if self._available is False:
            return None
        if self._morph is not None:
            return self._morph
        # Пробуем pymorphy3 первым — поддерживаемый форк, Python 3.12+ совместим.
        try:
            import pymorphy3
            self._morph = pymorphy3.MorphAnalyzer()
            self._available = True
            self._backend = "pymorphy3"
            return self._morph
        except ImportError:
            pass
        except Exception:
            pass
        # Fallback: pymorphy2 (legacy; требует compat-патча на Python 3.13).
        try:
            _install_getargspec_shim()
            import pymorphy2
            self._morph = pymorphy2.MorphAnalyzer()
            self._available = True
            self._backend = "pymorphy2"
        except ImportError:
            self._available = False
        except Exception:
            self._available = False
        return self._morph

    def fit(self, X, y=None):
        # Форсируем инициализацию при fit, чтобы is_active_ был известен
        # до начала трансформации и мог быть проверен снаружи (app.py).
        self._get_morph()
        self.is_active_: bool = bool(self._available)
        self.backend_: str = self._backend   # "pymorphy3" | "pymorphy2" | ""
        return self

    def _lemmatize_text(self, text: str) -> str:
        morph = self._get_morph()
        if morph is None:
            return text
        lines_out: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            # Секционные маркеры ([DESC], «…») — оставляем без изменений
            if self._SECTION_RE.match(stripped) or stripped in ("…", ""):
                lines_out.append(line)
                continue
            # Лемматизируем токены, не-токены оставляем
            result: List[str] = []
            last_end = 0
            for m in self._TOKEN_RE.finditer(line):
                result.append(line[last_end:m.start()])   # не-токен (пробелы, пунктуация)
                token = m.group()
                parsed = morph.parse(token)
                if parsed:
                    lemma = parsed[0].normal_form
                    if getattr(self, "include_pos", False):
                        _pos = getattr(parsed[0].tag, "POS", None)
                        if _pos:
                            lemma = f"{lemma}_{_pos}"
                else:
                    lemma = token.lower()
                result.append(lemma)
                last_end = m.end()
            result.append(line[last_end:])
            lines_out.append("".join(result))
        return "\n".join(lines_out)

    def transform(self, X) -> List[str]:
        return [self._lemmatize_text(str(t) if t is not None else "") for t in X]

    def fit_transform(self, X, y=None) -> List[str]:
        return self.fit(X, y).transform(X)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_morph"] = None      # MorphAnalyzer не сериализуем (ни v2, ни v3)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # _morph будет восстановлен при первом вызове _get_morph()


class MetaFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Извлекает 15 числовых мета-признаков из структурированного feature-текста.

    Работает на выходе build_feature_text(): парсит секции [TAG] и возвращает
    числовую матрицу (n_samples, 15), ортогональную TF-IDF признакам.

    Признаки:
       0  has_desc         — есть секция [DESC]        (0/1)
       1  has_client       — есть секция [CLIENT]      (0/1)
       2  has_operator     — есть секция [OPERATOR]    (0/1)
       3  has_summary      — есть секция [SUMMARY]     (0/1)
       4  has_ans_short    — есть секция [ANSWER_SHORT](0/1)
       5  has_ans_full     — есть секция [ANSWER_FULL] (0/1)
       6  is_call          — канал call или call+chat   (0/1)
       7  is_chat          — канал chat или call+chat   (0/1)
       8  desc_len         — log1p(слов в DESC)
       9  client_len       — log1p(слов в CLIENT)
      10  oper_len         — log1p(слов в OPERATOR)
      11  n_client_lines   — число реплик клиента (utterances)
      12  n_oper_lines     — число реплик оператора
      13  client_share     — client_words/(client_words+oper_words) ∈ [0..1]
      14  total_len        — log1p(общее число слов в тексте)
      15  n_fields_present — кол-во непустых полей из 6 (0–6); помогает SVM
                             понять степень заполненности строки

    Преимущество: TF-IDF не видит структуру диалога (кто говорит больше,
    какие поля заполнены). MetaFeatureExtractor добавляет именно этот сигнал.
    Работает совместно с PerFieldVectorizer через sklearn FeatureUnion.
    """

    _TAGS: Dict[str, str] = {
        "channel":   "[CHANNEL]",
        "desc":      "[DESC]",
        "client":    "[CLIENT]",
        "operator":  "[OPERATOR]",
        "summary":   "[SUMMARY]",
        "ans_short": "[ANSWER_SHORT]",
        "ans_full":  "[ANSWER_FULL]",
    }

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X) -> np.ndarray:
        rows = [self._extract(str(t) if t is not None else "") for t in X]
        return np.array(rows, dtype=np.float32)

    def _extract(self, text: str) -> List[float]:
        # --- Парсинг секций ---
        sections: Dict[str, List[str]] = {}
        current_tag: Optional[str] = None
        current_lines: List[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            matched = False
            for tag_key, tag_prefix in self._TAGS.items():
                if line == tag_prefix or line.startswith(tag_prefix):
                    if current_tag is not None:
                        sections[current_tag] = current_lines
                    current_tag = tag_key
                    current_lines = []
                    after = line[len(tag_prefix):].strip()
                    if after:
                        current_lines.append(after)
                    matched = True
                    break
            if not matched and current_tag is not None and line:
                current_lines.append(line)

        if current_tag is not None:
            sections[current_tag] = current_lines

        # --- Канал ---
        channel_text = " ".join(sections.get("channel", [])).lower()
        is_call = int("call" in channel_text)
        is_chat = int("chat" in channel_text)

        # --- Наличие секций ---
        has_desc      = int(bool(sections.get("desc")))
        has_client    = int(bool(sections.get("client")))
        has_operator  = int(bool(sections.get("operator")))
        has_summary   = int(bool(sections.get("summary")))
        has_ans_short = int(bool(sections.get("ans_short")))
        has_ans_full  = int(bool(sections.get("ans_full")))

        # --- Длины ---
        def _wc(lines: List[str]) -> int:
            return sum(len(l.split()) for l in lines)

        desc_words   = _wc(sections.get("desc", []))
        client_words = _wc(sections.get("client", []))
        oper_words   = _wc(sections.get("operator", []))

        n_client_lines = len(sections.get("client", []))
        n_oper_lines   = len(sections.get("operator", []))

        all_words = sum(_wc(v) for v in sections.values())

        # --- Доля клиента в диалоге ---
        total_dialog = client_words + oper_words
        client_share = (client_words / total_dialog) if total_dialog > 0 else 0.5

        n_fields_present = (
            has_desc + has_client + has_operator
            + has_summary + has_ans_short + has_ans_full
        )

        return [
            float(has_desc),
            float(has_client),
            float(has_operator),
            float(has_summary),
            float(has_ans_short),
            float(has_ans_full),
            float(is_call),
            float(is_chat),
            math.log1p(desc_words),
            math.log1p(client_words),
            math.log1p(oper_words),
            float(n_client_lines),
            float(n_oper_lines),
            client_share,
            math.log1p(all_words),
            float(n_fields_present),
        ]


class PerFieldVectorizer(BaseEstimator, TransformerMixin):
    """
    Раздельный TF-IDF для каждого поля (DESC, CLIENT, OPERATOR, SUMMARY, …).

    Отличие от единого гибридного TF-IDF:
    - Каждое поле имеет независимый словарь → слова не конкурируют между полями.
    - Веса применяются как скалярный множитель нормированной матрицы,
      а не как повторение блоков текста.  Это устраняет артефакт надутого TF
      (с субстрочным масштабированием sublinear_tf=True эффект умеренный,
      но всё же заметный при w=3).
    - Поля, пустые для конкретной строки (auto_profile → w=0), дают
      нулевой вектор без потери других полей.

    Совместимость: joblib-сериализация полная (не содержит непикаемых объектов).
    """

    # Пары (ключ_веса, [TAG]) — в порядке сборки feature-текста.
    _FIELDS: List[Tuple[str, str]] = [
        ("w_desc",         "DESC"),
        ("w_client",       "CLIENT"),
        ("w_operator",     "OPERATOR"),
        ("w_summary",      "SUMMARY"),
        ("w_answer_short", "ANSWER_SHORT"),
        ("w_answer_full",  "ANSWER_FULL"),
    ]

    # Скомпилированный паттерн для разбора секций — хранится как константа класса.
    _SPLIT_RE = _re.compile(
        r'\[(?:CHANNEL|DESC|CLIENT|OPERATOR|SUMMARY|ANSWER_SHORT|ANSWER_FULL)\]',
        _re.MULTILINE,
    )
    _TAG_RE = _re.compile(
        r'^\[([A-Z_]+)\]$'
    )

    def __init__(
        self,
        base_weights: Dict[str, int],
        char_ng: Tuple[int, int] = (2, 7),
        word_ng: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int = 300_000,
        sublinear_tf: bool = True,
        stop_words: Optional[List[str]] = None,
        normalize_output: bool = True,
    ):
        self.base_weights = base_weights
        self.char_ng = char_ng
        self.word_ng = word_ng
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.stop_words: List[str] = stop_words or []
        self.normalize_output = normalize_output
        # Заполняется при fit()
        self._active: List[Tuple[str, str, int]] = []       # (wkey, tag, weight)
        self._char_vecs: Dict[str, TfidfVectorizer] = {}
        self._word_vecs: Dict[str, TfidfVectorizer] = {}

    # ------------------------------------------------------------------

    def _parse_fields(self, text: str) -> Dict[str, str]:
        """Извлекает ПЕРВОЕ вхождение каждого поля из combined feature-text.

        build_feature_text может повторять блоки для весов (legacy) — берём
        только первое, чтобы не раздувать TF.
        """
        result: Dict[str, str] = {}
        lines = text.splitlines()
        current_tag: Optional[str] = None
        buf: List[str] = []

        for line in lines:
            m = self._TAG_RE.match(line.strip())
            if m:
                tag = m.group(1)
                if tag == "CHANNEL":
                    current_tag = None   # канал не векторизуем отдельно
                    buf = []
                    continue
                if tag in result:
                    # Второе вхождение того же тега = повтор для веса → пропускаем
                    current_tag = None
                    buf = []
                else:
                    if current_tag and current_tag not in result:
                        result[current_tag] = "\n".join(buf).strip()
                    current_tag = tag
                    buf = []
            else:
                if current_tag:
                    buf.append(line)

        if current_tag and current_tag not in result:
            result[current_tag] = "\n".join(buf).strip()

        return result

    def _make_tfidf_pair(
        self, max_char: int, max_word: int
    ) -> Tuple[TfidfVectorizer, TfidfVectorizer]:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ng,
            min_df=self.min_df,
            max_features=max(500, max_char),
            sublinear_tf=self.sublinear_tf,
        )
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.word_ng,
            min_df=1,
            max_features=max(200, max_word),
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words or None,
        )
        return char_vec, word_vec

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        import scipy.sparse as sp

        # Определяем активные поля и суммарный вес для пропорционального бюджета
        active = [
            (wk, tag, int(self.base_weights.get(wk, 0)))
            for wk, tag in self._FIELDS
            if int(self.base_weights.get(wk, 0)) > 0
        ]
        if not active:
            # Нет ни одного ненулевого поля — аварийный fallback
            active = [(wk, tag, 1) for wk, tag in self._FIELDS]
        self._active = active

        total_w = max(1, sum(w for _, _, w in active))

        # Собираем тексты по полям за один проход
        field_texts: Dict[str, List[str]] = {tag: [] for _, tag, _ in active}
        for text in X:
            parsed = self._parse_fields(str(text) if text is not None else "")
            for _, tag, _ in active:
                field_texts[tag].append(parsed.get(tag, ""))

        # Распределяем бюджет признаков пропорционально весам.
        # Проблема: max(floor, proportional) может превысить max_features на N_fields × floor.
        # Решение: вычислить «сырые» аллокации, затем масштабировать вниз если сумма > max_features.
        char_budget = self.max_features
        word_budget = max(1, self.max_features // 3)
        _MIN_CHAR, _MIN_WORD = 200, 100     # абсолютные минимумы на поле

        raw_char = {tag: max(_MIN_CHAR, int(char_budget * w / total_w)) for _, tag, w in active}
        raw_word = {tag: max(_MIN_WORD, int(word_budget * w / total_w)) for _, tag, w in active}

        # Масштабируем вниз если сумма превысила бюджет
        sum_char = sum(raw_char.values())
        if sum_char > char_budget:
            scale = char_budget / sum_char
            raw_char = {tag: max(_MIN_CHAR, int(v * scale)) for tag, v in raw_char.items()}
        sum_word = sum(raw_word.values())
        if sum_word > word_budget:
            scale = word_budget / sum_word
            raw_word = {tag: max(_MIN_WORD, int(v * scale)) for tag, v in raw_word.items()}

        # Обучаем пары (char + word) TF-IDF для каждого активного поля.
        # Поля, у которых все тексты пустые, исключаем из _active.
        self._char_vecs = {}
        self._word_vecs = {}
        confirmed_active: List[Tuple[str, str, int]] = []
        for wk, tag, w in active:
            texts_for_tag = field_texts[tag]
            if not any(t.strip() for t in texts_for_tag):
                # Нет ни одного непустого документа — пропускаем поле
                continue
            max_char = raw_char[tag]
            max_word = raw_word[tag]
            cv, wv = self._make_tfidf_pair(max_char, max_word)
            try:
                cv.fit(texts_for_tag)
                wv.fit(texts_for_tag)
            except ValueError:
                # empty vocabulary — например, все тексты состоят из стоп-слов
                continue
            self._char_vecs[tag] = cv
            self._word_vecs[tag] = wv
            confirmed_active.append((wk, tag, w))
        self._active = confirmed_active
        # Маркер для sklearn check_is_fitted (ищет атрибуты оканчивающиеся на '_')
        self.n_features_in_ = sum(
            cv.vocabulary_.__len__() + wv.vocabulary_.__len__()
            for cv, wv in zip(self._char_vecs.values(), self._word_vecs.values())
        )
        return self

    def transform(self, X):
        import scipy.sparse as sp

        X_list = [str(t) if t is not None else "" for t in X]

        # Парсим каждый документ ровно один раз и кэшируем все поля.
        # Без этого _parse_fields вызывался бы N_active_fields раз на строку.
        parsed_rows = [self._parse_fields(text) for text in X_list]

        matrices = []
        for wk, tag, w in self._active:
            field_texts = [p.get(tag, "") for p in parsed_rows]
            mat_char = self._char_vecs[tag].transform(field_texts)
            mat_word = self._word_vecs[tag].transform(field_texts)
            mat = sp.hstack([mat_char, mat_word], format="csr")
            if w != 1:
                mat = mat.multiply(float(w))
            matrices.append(mat)

        matrices = [m for m in matrices if m.shape[1] > 0]
        if not matrices:
            return sp.csr_matrix((len(X_list), self.n_features_in_))
        result = sp.hstack(matrices, format="csr")
        # Нормализация по строкам: делает частичные строки (2–3 поля из 6)
        # сопоставимыми по масштабу с полными. Без нормализации при пропуске
        # 3 из 6 полей L2-норма вектора уменьшается, сдвигая SVM-решение к интерцепту.
        if getattr(self, "normalize_output", True):
            from sklearn.preprocessing import normalize as _normalize
            result = _normalize(result, norm="l2", copy=False)
        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def make_hybrid_vectorizer(
    char_ng: Tuple[int, int] = (2, 9),
    word_ng: Tuple[int, int] = (1, 3),
    min_df: int = 3,
    max_features: int = 150_000,
    sublinear_tf: bool = True,
    use_stop_words: bool = False,
    extra_stop_words: Optional[List[str]] = None,
    extra_noise_tokens: Optional[List[str]] = None,
    extra_noise_phrases: Optional[List[str]] = None,
    extra_phrases: Optional[List[str]] = None,
    use_noise_tokens: bool = True,
    use_noise_phrases: bool = True,
    # --- per-field режим ---
    use_per_field: bool = True,
    base_weights: Optional[Dict[str, int]] = None,
    # --- Латентный семантический анализ (SVD / LSA) ---
    use_svd: bool = True,
    svd_components: int = 200,
    # --- Лемматизация (pymorphy2) ---
    use_lemma: bool = True,
    use_pos_tags: bool = False,
    # --- Числовые мета-признаки ---
    use_meta: bool = False,
) -> Any:
    """
    Создаёт пайплайн: PhraseRemover → векторайзер.

    use_per_field=True (по умолчанию):
        Каждое поле (DESC, CLIENT, …) получает отдельный TF-IDF.
        Веса применяются как скалярный множитель нормированной матрицы.
        Требует base_weights: {'w_desc': N, 'w_client': N, …}.

    use_per_field=False (legacy):
        Единый char_wb + word TF-IDF на весь собранный текст.
        Веса реализованы через повторение блоков в build_feature_text.

    use_meta=True:
        Добавляет MetaFeatureExtractor через FeatureUnion поверх текстового
        пайплайна. Извлекает 15 числовых признаков структуры диалога
        (наличие секций, длины, соотношения клиент/оператор, канал).
        Возвращает FeatureUnion вместо Pipeline.

    PhraseRemover убирает шаблонные фразы до токенизации.
    NOISE_TOKENS добавляются в стоп-слова word-TF-IDF.
    """
    # --- стоп-слова для word-TF-IDF ---
    # Всё приводим к lower: TF-IDF с lowercase=True сравнивает с уже строчными токенами
    sw_base        = RUSSIAN_STOP_WORDS if use_stop_words else set()
    sw_extra       = {w.lower() for w in (extra_stop_words or [])}
    sw_noise       = NOISE_TOKENS if use_noise_tokens else set()
    sw_user_tokens = {t.lower() for t in (extra_noise_tokens or [])}
    effective_stop_words: List[str] = sorted(sw_base | sw_noise | sw_extra | sw_user_tokens)

    # --- фразы для PhraseRemover ---
    noise_ph  = NOISE_PHRASES if use_noise_phrases else []
    user_ph   = [p.lower() for p in (extra_noise_phrases or []) if p.strip()]
    # Пользовательские фразы идут первыми — матчатся до встроенных коротких
    all_phrases = user_ph + list(noise_ph) + (list(extra_phrases) if extra_phrases else [])
    phrase_remover = PhraseRemover(all_phrases)

    if use_per_field and base_weights:
        tfidf: Any = PerFieldVectorizer(
            base_weights=base_weights,
            char_ng=char_ng,
            word_ng=word_ng,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            stop_words=effective_stop_words,
        )
    else:
        # Legacy: единый гибридный TF-IDF
        vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ng,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        vec_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ng,
            min_df=min_df,
            max_features=max(50_000, max_features // 2),
            sublinear_tf=sublinear_tf,
            stop_words=effective_stop_words,
        )
        tfidf = FeatureUnion([("char", vec_char), ("word", vec_word)])

    steps: list = [("phrase_remover", phrase_remover)]
    if use_lemma:
        steps.append(("lemmatizer", Lemmatizer(include_pos=bool(use_pos_tags))))
    steps.append(("tfidf", tfidf))

    if use_svd:
        # TruncatedSVD (LSA): снижает шум за счёт латентного семантического
        # пространства. После SVD необходима L2-нормализация (cosine нормировка),
        # т.к. LinearSVC предполагает нормированные признаки.
        n = max(50, min(svd_components, 3000))
        steps.append(("svd", TruncatedSVD(n_components=n, random_state=42)))
        steps.append(("svd_norm", Normalizer(copy=False)))

    text_pipeline = Pipeline(steps)

    if use_meta:
        # FeatureUnion объединяет текстовый пайплайн (sparse/dense TF-IDF)
        # с числовыми мета-признаками структуры диалога (15 float).
        # StandardScaler нормализует мета-признаки (разные шкалы: бинарные,
        # log1p длин, доли) перед подачей в LinearSVC.
        # Оба трансформера получают одинаковые входные тексты X.
        meta_pipeline = Pipeline([
            ("extract", MetaFeatureExtractor()),
            ("scale", StandardScaler()),
        ])
        return FeatureUnion([
            ("text", text_pipeline),
            ("meta", meta_pipeline),
        ])

    return text_pipeline
