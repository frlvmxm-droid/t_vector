"""
ml_setfit.py — SetFitClassifier: sklearn-совместимый классификатор на базе SetFit.

SetFit (HuggingFace) обучает контрастную классификационную голову поверх
замороженного SBERT-энкодера. Отлично работает на few-shot сценариях
(50–2000 примеров на класс) и стабильно превосходит LinearSVC+TF-IDF
на задачах семантической классификации текста.

Требования: pip install setfit>=0.9

VRAM-пороги авто-настройки можно переопределить через env-переменные:
    BRT_SETFIT_MAX_TRAIN_OVERRIDE  — принудительный cap размера датасета
    BRT_SETFIT_MAX_PAIRS_OVERRIDE  — принудительный cap кол-ва пар
"""
from __future__ import annotations

import gc
import importlib.util
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config.ml_constants import SETFIT_VRAM_PROFILES

# Снижает фрагментацию GPU-памяти; рекомендовано самим PyTorch при OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# VRAM-профили: дефолты определены в config.ml_constants.SETFIT_VRAM_PROFILES;
# env-override-ы (см. docstring) применяются поверх. Локальный алиас
# `VRAM_PROFILES` оставлен для обратной совместимости — тесты и внешние
# утилиты могут monkeypatch-ить его.
# ---------------------------------------------------------------------------

VRAM_PROFILES: tuple[tuple[float, int, int], ...] = SETFIT_VRAM_PROFILES


def _pick_vram_profile(vram_gb: float) -> tuple[int, int]:
    """Возвращает (max_train, max_pairs) для заданного объёма VRAM."""
    override_train = os.environ.get("BRT_SETFIT_MAX_TRAIN_OVERRIDE")
    override_pairs = os.environ.get("BRT_SETFIT_MAX_PAIRS_OVERRIDE")
    for thr, max_train, max_pairs in VRAM_PROFILES:
        if vram_gb >= thr:
            _train = int(override_train) if override_train else max_train
            _pairs = int(override_pairs) if override_pairs else max_pairs
            return _train, _pairs
    return VRAM_PROFILES[-1][1], VRAM_PROFILES[-1][2]


# ---------------------------------------------------------------------------
# VRAM cleanup helper — удалён дубликат из fit() (3 места).
# ---------------------------------------------------------------------------

def _cuda_cleanup(*, synchronize: bool = True) -> None:
    """Best-effort освобождение CUDA-кэша. Никогда не поднимает исключение."""
    try:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            if synchronize:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):  # noqa: BLE001 — best-effort only
        pass


def _is_setfit_config_missing_error(exc: BaseException) -> bool:
    """True если исключение — отсутствие config_setfit.json (нужен fallback на ST)."""
    # HF/huggingface_hub не гарантирует единого типа исключения для missing entry:
    # ловим EntryNotFoundError и OSError/RepositoryNotFoundError, плюс substring
    # как последнюю линию защиты для совместимости с разными версиями hub.
    exc_name = type(exc).__name__
    if exc_name in ("EntryNotFoundError", "RepositoryNotFoundError"):
        return True
    msg = str(exc)
    return any(k in msg for k in ("Entry Not Found", "config_setfit", "404"))


# ---------------------------------------------------------------------------
# SetFitClassifier
# ---------------------------------------------------------------------------

class SetFitClassifier:
    """
    Sklearn-API совместимый классификатор на базе HuggingFace SetFit.

    Особенности сериализации:
      • PyTorch-веса НЕ сохраняются в joblib — это нарушит pickle/unpickle.
      • При fit() веса сохраняются через model.save_pretrained(local_path).
      • В joblib сохраняется путь _local_model_path и метаданные.
      • При загрузке joblib модель подгружается из _local_model_path lazily
        при первом вызове predict() / predict_proba().

    Параметры:
        model_name       — HuggingFace repo id (напр. "deepvk/USER2-base")
        num_iterations   — кол-во контрастных пар на класс (SetFit)
        num_epochs       — эпохи обучения классификационной головы
        batch_size       — размер батча контрастного обучения
        fp16             — использовать mixed-precision (только GPU)
        device           — "auto" | "cpu" | "cuda"
        cache_dir        — путь к папке кеша (None = sbert_models/ рядом с app.py)
        log_cb           — callback(str) для текстового лога
        progress_cb      — callback(float, str) для прогресс-бара
        progress_range   — диапазон (start%, end%) прогресс-бара
    """

    def __init__(
        self,
        model_name: str = "deepvk/USER2-base",
        num_iterations: int = 20,
        num_epochs: int = 3,
        batch_size: int = 8,
        fp16: bool = True,
        max_length: int = 256,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        progress_range: Tuple[float, float] = (50.0, 92.0),
    ):
        self.model_name = model_name
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.fp16 = fp16
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self.log_cb = log_cb
        self.progress_cb = progress_cb
        self.progress_range = progress_range

        self._model: Any = None
        self._local_model_path: Optional[str] = None
        self.classes_: Optional[List[str]] = None

    # ------------------------------------------------------------------ utils

    @staticmethod
    def is_available() -> bool:
        """Проверяет наличие пакета setfit."""
        return importlib.util.find_spec("setfit") is not None

    def _log(self, msg: str) -> None:
        if self.log_cb:
            self.log_cb(msg)

    def _prog(self, pct: float, status: str) -> None:
        if self.progress_cb:
            lo, hi = self.progress_range
            mapped = lo + (hi - lo) * pct / 100.0
            self.progress_cb(mapped, status)

    def _resolve_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _default_cache_dir(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir)
        # Рядом с этим файлом (корень приложения)
        return Path(__file__).parent / "sbert_models"

    # ------------------------------------------------------------------ fit

    def fit(self, X: List[str], y: List[str]) -> "SetFitClassifier":
        """
        Обучает SetFit-классификатор на текстах X с метками y.

        1. Загружает SBERT-энкодер model_name
        2. Проводит контрастное обучение (num_iterations пар × num_epochs)
        3. Обучает классификационную голову (логистическая регрессия)
        4. Сохраняет веса через save_pretrained() в sbert_models/setfit_*/
        """
        if not self.is_available():
            raise ImportError(
                "Пакет setfit не установлен. Выполните:\n"
                "  pip install setfit>=0.9"
            )

        # Патч совместимости: default_logdir убрана из transformers >= 4.40
        import transformers.training_args as _ta
        if not hasattr(_ta, "default_logdir"):
            import datetime
            def _default_logdir() -> str:
                current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
                return os.path.join("runs", current_time)
            _ta.default_logdir = _default_logdir

        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Освобождаем память от предыдущей модели (если была) перед загрузкой новой —
        # без этого повторный вызов fit() накапливает модели на GPU и ведёт к OOM.
        if self._model is not None:
            try:
                self._model.to("cpu")
            except (RuntimeError, AttributeError):  # noqa: BLE001 — best-effort VRAM release
                pass
            del self._model
            self._model = None
            _cuda_cleanup()

        self.classes_ = sorted(set(y))
        device = self._resolve_device()

        _cuda_cleanup()

        self._log(f"[SetFit] Загрузка энкодера «{self.model_name}» на {device}…")
        self._prog(0.0, f"[SetFit] Загрузка {self.model_name}…")

        hf_kwargs: Dict[str, Any] = {}
        if self.cache_dir:
            hf_kwargs["cache_dir"] = self.cache_dir

        try:
            model = SetFitModel.from_pretrained(
                self.model_name,
                **hf_kwargs,
            )
        except Exception as _load_err:
            if _is_setfit_config_missing_error(_load_err):
                # Модель — обычный SentenceTransformer без SetFit-чекпоинта.
                # Создаём SetFitModel вручную поверх тела SentenceTransformer.
                self._log(
                    f"[SetFit] config_setfit.json не найден — "
                    f"инициализируем SetFitModel из SentenceTransformer…"
                )
                from sentence_transformers import SentenceTransformer as _ST
                _st_kwargs: Dict[str, Any] = {}
                if self.cache_dir:
                    _st_kwargs["cache_folder"] = self.cache_dir
                _st_body = _ST(self.model_name, device=device, **_st_kwargs)
                from sklearn.linear_model import LogisticRegression as _LR
                _head = _LR(max_iter=100, C=1.0, class_weight="balanced")
                model = SetFitModel(model_body=_st_body, model_head=_head, labels=self.classes_)
            else:
                raise
        model.to(device)

        # Страховка: в setfit >= 1.0 model_head = None если не передан явно при
        # инициализации через конструктор (не from_pretrained).
        if getattr(model, "model_head", None) is None:
            from sklearn.linear_model import LogisticRegression as _LR2
            model.model_head = _LR2(max_iter=100, C=1.0, class_weight="balanced")
            self._log("[SetFit] model_head создан вручную (LogisticRegression)")

        # Ограничиваем длину последовательности: bank text короткий (50-150 токенов),
        # а attention O(seq²) — 128 vs 512 = 16× экономия памяти.
        try:
            _body = getattr(model, "model_body", None)
            if _body is not None and hasattr(_body, "max_seq_length"):
                _body.max_seq_length = self.max_length
                self._log(f"[SetFit] max_seq_length → {self.max_length}")
        except Exception:
            pass

        # Gradient checkpointing: снижает пик VRAM за счёт пересчёта активаций
        if device == "cuda":
            try:
                _body = getattr(model, "model_body", None) or getattr(model, "body", None)
                if _body is not None:
                    _inner = getattr(_body, "_modules", {})
                    # SentenceTransformer хранит трансформер внутри
                    for _sub in _inner.values():
                        _auto = getattr(_sub, "auto_model", None)
                        if _auto is not None and hasattr(_auto, "gradient_checkpointing_enable"):
                            _auto.gradient_checkpointing_enable()
                            self._log("[SetFit] Gradient checkpointing включён")
                            break
            except Exception:
                pass

        self._log(
            f"[SetFit] Обучение: {len(X)} примеров | "
            f"{len(self.classes_)} классов | "
            f"iterations={self.num_iterations} | epochs={self.num_epochs} | "
            f"batch={self.batch_size} | device={device}"
        )
        self._prog(10.0, "[SetFit] Подготовка датасета…")

        train_dataset = Dataset.from_dict({"text": list(X), "label": list(y)})

        _use_fp16 = False
        _use_bf16 = False
        if self.fp16 and device.startswith("cuda"):
            try:
                import torch as _torch_prec
                _dev_idx = int(device.split(":")[1]) if ":" in device else 0
                _props = _torch_prec.cuda.get_device_properties(_dev_idx)
                if _props.major >= 8:   # Ampere+ (A100) и Ada Lovelace (RTX 40xx)
                    _use_bf16 = True
                else:
                    _use_fp16 = True
            except Exception:
                _use_fp16 = True        # fallback на fp16 при ошибке

        _train_kwargs: Dict[str, Any] = dict(
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            num_iterations=self.num_iterations,
            show_progress_bar=False,  # пишем свой прогресс через log_cb
        )
        import inspect as _insp
        _ta_params = set(_insp.signature(TrainingArguments.__init__).parameters)
        if _use_bf16 and "bf16" in _ta_params:
            _train_kwargs["bf16"] = True
            self._log("[SetFit] BF16 включён (Ampere+/Ada Lovelace)")
        elif _use_fp16 and "use_amp" in _ta_params:
            _train_kwargs["use_amp"] = True
            self._log("[SetFit] FP16 AMP включён")
        args = TrainingArguments(**_train_kwargs)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )

        self._prog(20.0, "[SetFit] Контрастное обучение…")
        _cuda_cleanup(synchronize=False)
        trainer.train()
        self._prog(85.0, "[SetFit] Сохранение весов…")

        # Сохраняем веса на диск (НЕ в joblib)
        cache_base = self._default_cache_dir()
        cache_base.mkdir(parents=True, exist_ok=True)
        _slug = self.model_name.replace("/", "_").replace("\\", "_")
        _ts = int(time.time())
        local_path = cache_base / f"setfit_{_slug}_{_ts}"
        local_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(local_path))

        self._model = model
        self._local_model_path = str(local_path)

        self._log(f"[SetFit] ✅ Обучение завершено. Веса: {self._local_model_path}")
        self._prog(100.0, "[SetFit] Готово")
        return self

    # ------------------------------------------------------------------ predict

    def _ensure_model(self) -> None:
        """Ленивая загрузка модели из _local_model_path при первом predict."""
        if self._model is not None:
            return
        if not self._local_model_path:
            raise RuntimeError(
                "SetFitClassifier: модель не обучена. "
                "Вызовите fit() перед predict()."
            )
        if not self.is_available():
            raise ImportError("Пакет setfit не установлен. Выполните: pip install setfit>=0.9")

        from setfit import SetFitModel

        device = self._resolve_device()
        _cuda_cleanup(synchronize=False)
        self._log(f"[SetFit] Загрузка модели из {self._local_model_path} → {device}…")
        self._model = SetFitModel.from_pretrained(self._local_model_path)
        self._model.to(device)

    def predict(self, X: List[str]) -> List[str]:
        """Возвращает список предсказанных меток."""
        self._ensure_model()
        try:
            import torch as _torch_inf
            with _torch_inf.no_grad():
                preds = self._model.predict(list(X))
        except Exception:
            preds = self._model.predict(list(X))
        # predict() может вернуть torch.Tensor или numpy array
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        return [str(p) for p in preds]

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Возвращает матрицу вероятностей (n_samples, n_classes).
        Порядок классов соответствует self.classes_.
        """
        self._ensure_model()
        try:
            import torch as _torch_inf
            with _torch_inf.no_grad():
                proba = self._model.predict_proba(list(X))
        except Exception:
            proba = self._model.predict_proba(list(X))
        # Может быть torch.Tensor
        if hasattr(proba, "numpy"):
            proba = proba.numpy()
        arr = np.array(proba, dtype=np.float32)
        # Нормализуем строки на случай численных артефактов
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return arr / row_sums

    # ------------------------------------------------------------------ serialization

    def __getstate__(self) -> dict:
        """
        joblib/pickle сериализация: исключаем PyTorch-модель и callbacks.
        Сохраняем только гиперпараметры, classes_ и путь к весам.
        """
        state = self.__dict__.copy()
        state["_model"] = None        # torch.nn.Module — не сериализуется
        state["log_cb"] = None        # функции-callback
        state["progress_cb"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """При загрузке из joblib — сбрасываем модель (будет lazy-load)."""
        self.__dict__.update(state)
        self._model = None  # lazy reload при следующем predict()


# ---------------------------------------------------------------------------
# train_model_setfit
# ---------------------------------------------------------------------------

def train_model_setfit(
    X: List[str],
    y: List[str],
    model_name: str,
    num_iterations: int = 20,
    num_epochs: int = 3,
    batch_size: int = 16,
    fp16: bool = True,
    max_length: int = 128,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
    device: str = "auto",
    cache_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[SetFitClassifier, str, str, Optional[List[str]], Optional[Any], Dict]:
    """
    Обучает SetFitClassifier и возвращает тот же 6-tuple что и train_model().

    Returns:
        clf         — обученный SetFitClassifier
        clf_type    — строка "SetFit"
        report      — текстовый classification_report
        labels      — список меток (или None при пропуске валидации)
        cm          — confusion matrix (или None)
        extras      — {'thresh_90', 'thresh_75', 'thresh_50', 'report_dict',
                       'per_class_thresholds', 'roc_auc_macro'}
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import precision_recall_curve

    clf_type = "SetFit"

    if log_cb:
        log_cb(f"[SetFit] Режим: нейросетевой классификатор SetFit")
        log_cb(f"[SetFit] Модель: {model_name}")
        log_cb(f"[SetFit] Выборка: {len(X)} примеров | классов: {len(set(y))}")

    _cnt = Counter(y)
    _min_class = min(_cnt.values()) if _cnt else 0

    # Нет смысла в valset если данных очень мало
    skip_val = (
        test_size <= 0
        or len(set(y)) < 2
        or len(y) < 20
        or _min_class < 2
    )

    if skip_val:
        reason = (
            "test_size=0" if test_size <= 0
            else f"класс с {_min_class} примером(ами)"
            if _min_class < 2
            else "мало данных"
        )
        if progress_cb:
            progress_cb(50.0, f"[SetFit] Обучение (без валидации)…")
        clf = SetFitClassifier(
            model_name=model_name,
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            batch_size=batch_size,
            fp16=fp16,
            max_length=max_length,
            device=device,
            cache_dir=cache_dir,
            log_cb=log_cb,
            progress_cb=progress_cb,
            progress_range=(50.0, 92.0),
        )
        clf.fit(X, y)
        if log_cb:
            log_cb(f"[SetFit] Валидация пропущена: {reason}")
        return clf, clf_type, f"ВАЛИДАЦИЯ ПРОПУЩЕНА ({reason}).", None, None, {}

    Xtr, Xva, ytr, yva = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )
    if log_cb:
        log_cb(f"[SetFit] Разбивка: обучение={len(Xtr)}, валидация={len(Xva)}")

    # Текстовый оверсэмплинг (аналог SMOTE для строк).
    # Для SetFit используем множитель ×2 (не ×5): SetFit уже балансирует
    # классы через контрастные пары — агрессивный SMOTE лишь раздувает
    # датасет и ведёт к OOM при генерации пар.
    if use_smote:
        _rng = np.random.default_rng(int(random_state))
        _tr_cnt = Counter(ytr)
        _max_cnt = max(_tr_cnt.values())
        _min_cnt = min(_tr_cnt.values())
        if _max_cnt / max(_min_cnt, 1) >= 2.0:
            _Xaug: List[str] = list(Xtr)
            _yaug: List[str] = list(ytr)
            for _cls, _c in _tr_cnt.items():
                if _max_cnt / _c >= 2.0:
                    _target = min(_c * 2, _max_cnt)   # ×2 вместо ×5
                    _n_add = _target - _c
                    _src = [i for i, lbl in enumerate(ytr) if lbl == _cls]
                    for _ in range(_n_add):
                        _Xaug.append(Xtr[int(_rng.choice(_src))])
                        _yaug.append(_cls)
            if len(_Xaug) > len(Xtr):
                _orig = len(Xtr)
                _combined = list(zip(_Xaug, _yaug))
                _rng.shuffle(_combined)
                Xtr = [x for x, _ in _combined]
                ytr = [lbl for _, lbl in _combined]
                if log_cb:
                    log_cb(f"[Балансировка] оверсэмплинг: {_orig} → {len(Xtr)} примеров")

    # VRAM-профиль: лимит размера датасета и кол-ва пар. Настраивается через
    # VRAM_PROFILES / env BRT_SETFIT_MAX_TRAIN_OVERRIDE, BRT_SETFIT_MAX_PAIRS_OVERRIDE.
    _hw_vram = 0.0
    try:
        from hw_profile import detect as _hw_detect_sf
        _hw_vram = _hw_detect_sf().gpu_vram_gb or 0.0
    except (ImportError, RuntimeError):
        pass

    _SETFIT_MAX_TRAIN, _MAX_PAIRS = _pick_vram_profile(_hw_vram)

    if len(Xtr) > _SETFIT_MAX_TRAIN:
        _rng_cap = np.random.default_rng(int(random_state) + 99)
        _tr_cnt_cap = Counter(ytr)
        _cap_idx: List[int] = []
        for _cls, _c in _tr_cnt_cap.items():
            _keep = max(1, round(_c * _SETFIT_MAX_TRAIN / len(ytr)))
            _src_idx = [i for i, lab in enumerate(ytr) if lab == _cls]
            _chosen = _rng_cap.choice(_src_idx, min(len(_src_idx), _keep), replace=False)
            _cap_idx.extend(int(i) for i in _chosen)
        _before_cap = len(Xtr)
        Xtr = [Xtr[i] for i in _cap_idx]
        ytr = [ytr[i] for i in _cap_idx]
        if log_cb:
            log_cb(
                f"[Балансировка] cap SetFit: {_before_cap} → {len(Xtr)} примеров"
                f" (лимит {_SETFIT_MAX_TRAIN:,} для VRAM={_hw_vram:.0f}ГБ)"
            )

    _adj_iters = min(num_iterations, max(1, _MAX_PAIRS // (2 * max(len(Xtr), 1))))
    if _adj_iters < num_iterations:
        if log_cb:
            log_cb(
                f"[SetFit] num_iterations: {num_iterations} → {_adj_iters}"
                f" (лимит пар {_MAX_PAIRS:,} для VRAM={_hw_vram:.0f}ГБ)"
            )
        num_iterations = _adj_iters

    # Авто-коррекция batch_size для large-моделей (~870 MB vs ~430 MB у base).
    # Large-модели занимают вдвое больше VRAM при forward+backward → нужен меньший batch.
    # Применяем только если VRAM ограничена (< 24 ГБ) и явно выбрана large-модель.
    _model_lower = (model_name or "").lower()
    _is_large_model = any(tag in _model_lower for tag in ("-large", "_large", "large-"))
    if _is_large_model and _hw_vram < 24 and batch_size > 1:
        _batch_before = batch_size
        batch_size = max(1, batch_size // 2)
        if log_cb:
            log_cb(
                f"[SetFit] Large-модель на {_hw_vram:.0f}ГБ VRAM → "
                f"batch: {_batch_before} → {batch_size}"
            )

    if progress_cb:
        progress_cb(48.0, "[SetFit] Запуск обучения…")

    clf = SetFitClassifier(
        model_name=model_name,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
        batch_size=batch_size,
        fp16=fp16,
        max_length=max_length,
        device=device,
        cache_dir=cache_dir,
        log_cb=log_cb,
        progress_cb=progress_cb,
        progress_range=(50.0, 90.0),
    )
    clf.fit(Xtr, ytr)

    if progress_cb:
        progress_cb(91.0, "[SetFit] Валидация…")

    extras: Dict[str, Any] = {}

    proba = clf.predict_proba(Xva)
    classes = clf.classes_ or sorted(set(y))
    pred = [classes[int(row.argmax())] for row in proba]

    # Пороги уверенности
    val_confs = np.array([float(row.max()) for row in proba])
    if len(val_confs) > 0:
        extras["thresh_90"] = float(np.percentile(val_confs, 90))
        extras["thresh_75"] = float(np.percentile(val_confs, 75))
        extras["thresh_50"] = float(np.percentile(val_confs, 50))

    rep = classification_report(yva, pred, digits=3, zero_division=0)
    rep_dict = classification_report(yva, pred, digits=3, output_dict=True, zero_division=0)
    extras["report_dict"] = rep_dict

    if log_cb:
        _f1  = rep_dict.get("macro avg", {}).get("f1-score")
        _acc = rep_dict.get("accuracy")
        _f1_s  = f"{_f1:.3f}"  if _f1  is not None else "?"
        _acc_s = f"{_acc:.3f}" if _acc is not None else "?"
        log_cb(
            f"[Валидация SetFit] macro F1={_f1_s} | "
            f"accuracy={_acc_s} | val_size={len(yva)}"
        )

    labels = sorted(set(yva))
    cm = confusion_matrix(yva, pred, labels=labels)

    # Per-class пороги (precision >= 0.85)
    per_class_thresh: Dict[str, float] = {}
    for _i, _cls in enumerate(classes):
        if _i >= proba.shape[1]:
            continue
        _cls_proba = proba[:, _i]
        _yva_arr = np.array(yva)
        _cls_true = (_yva_arr == _cls).astype(int)
        if _cls_true.sum() < 2:
            per_class_thresh[str(_cls)] = 0.5
            continue
        _precisions, _recalls, _thresholds = precision_recall_curve(_cls_true, _cls_proba)
        _good = [
            (float(t), float(p), float(r))
            for p, r, t in zip(_precisions, _recalls, _thresholds)
            if p >= 0.85 and r > 0
        ]
        if _good:
            def _f05(tpr):
                _, p, r = tpr
                denom = 0.25 * p + r
                return 1.25 * p * r / denom if denom > 0 else 0.0
            per_class_thresh[str(_cls)] = max(_good, key=_f05)[0]
        elif len(_thresholds) > 0:
            per_class_thresh[str(_cls)] = float(np.median(_thresholds))
        else:
            per_class_thresh[str(_cls)] = 0.5
    extras["per_class_thresholds"] = per_class_thresh

    # ROC-AUC macro
    try:
        from sklearn.metrics import roc_auc_score as _roc_auc
        _roc = _roc_auc(
            yva, proba,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
        extras["roc_auc_macro"] = float(_roc)
        if log_cb:
            log_cb(f"[Валидация SetFit] ROC-AUC macro={float(_roc):.3f}")
    except Exception:
        pass

    return clf, clf_type, rep, labels, cm, extras
