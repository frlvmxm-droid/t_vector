# -*- coding: utf-8 -*-
"""
ml_sbert_bootstrap — bootstrap-патчи для безопасной загрузки sentence-transformers.

Проблемы, которые решает этот модуль, возникают из-за взаимодействия
Python 3.13 × torch × transformers:

  1. `torch.__version__ is None` на некоторых сборках → ломает
     `packaging.version.parse` при импорте transformers.
  2. Модули transformers используют `torch`, `nn`, `LRScheduler` в аннотациях
     типов без явного импорта — Python 3.13 падает с NameError.
  3. `sentence_transformers` подхватывает кэш с `is_torch_available() == False`.

Точки входа, в порядке вложенности:

  * :class:`SBERTBuiltinsPatch` — контекстный менеджер, инжектирует имена в
    ``builtins`` и гарантированно их убирает.
  * :func:`patch_torch_and_packaging` — идемпотентный fix для torch.__version__.
  * :func:`safe_import_sentence_transformers` — объединённый retry-импорт с
    автоинжекцией недостающих имён в builtins через ``SBERTBuiltinsPatch``.

Для обратной совместимости с внешним кодом, использующим классы с
подчёркиванием, модуль ``ml_vectorizers`` продолжает ре-экспортировать
``_BuiltinsPatch`` как алиас ``SBERTBuiltinsPatch``.
"""
from __future__ import annotations

import re as _re
from types import SimpleNamespace
from typing import Callable, Optional

from app_logger import get_logger
from config.ml_constants import SBERT_IMPORT_MAX_RETRIES

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Контекстный менеджер для инжекции имён в builtins
# ---------------------------------------------------------------------------


class SBERTBuiltinsPatch:
    """Контекстный менеджер, инжектирующий имена в builtins с гарантией очистки.

    Используется при загрузке SentenceTransformer, чтобы обойти NameError
    в Python 3.13, возникающий когда transformers-модули используют
    ``torch``/``nn``/``LRScheduler`` в аннотациях типов без явного импорта.

    Пример::

        with SBERTBuiltinsPatch() as bp:
            bp.inject('torch')
            bp.inject('nn')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(...)
        # builtins очищены — даже при исключении
    """

    def __init__(self) -> None:
        self._injected: list[str] = []

    def inject(self, name: str) -> bool:
        """Ищет *name* в torch-пакете и добавляет в ``builtins``.

        Возвращает ``True`` при успехе. Если имя уже есть в ``builtins`` —
        ничего не делает. В крайнем случае инжектирует ``SimpleNamespace``
        как заглушку (только для случая, когда имя используется как type-hint).
        """
        import builtins as _b
        if hasattr(_b, name):
            return True
        import sys as _s
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
            _sources: list = []
            try:
                import torch as _t
                _sources.append(_t)
            except ImportError:
                pass
            try:
                import torch.nn as _tnn
                _sources.append(_tnn)
            except ImportError:
                pass
            try:
                import torch.optim as _topt
                _sources.append(_topt)
            except ImportError:
                pass
            try:
                import torch.optim.lr_scheduler as _tlrs
                _sources.append(_tlrs)
            except ImportError:
                pass
            for _src in _sources:
                if hasattr(_src, name):
                    setattr(_b, name, getattr(_src, name))
                    self._injected.append(name)
                    return True
        except AttributeError:
            pass  # API torch изменился — пойдём по SimpleNamespace-фоллбэку

        _log.warning(
            "[ml_sbert_bootstrap] builtins.%s не найден — инжектирован "
            "SimpleNamespace-stub. Если transformers упадёт с AttributeError — "
            "обновите torch.",
            name,
        )
        setattr(_b, name, SimpleNamespace())
        self._injected.append(name)
        return True

    def __enter__(self) -> "SBERTBuiltinsPatch":
        return self

    def __exit__(self, *_exc_info) -> bool:
        try:
            import builtins as _b
            for _name in self._injected:
                if hasattr(_b, _name):
                    delattr(_b, _name)
        except (AttributeError, TypeError):
            pass
        return False  # не подавляем исключение


# ---------------------------------------------------------------------------
# Идемпотентный фикс torch.__version__ / packaging.version.parse
# ---------------------------------------------------------------------------


def patch_torch_and_packaging() -> None:
    """Идемпотентно поправляет ``torch.__version__`` и ``packaging.version.parse``.

    На некоторых сборках ``torch.__version__`` возвращает ``None``, что ломает
    ``packaging.version.parse()`` при импорте transformers. Здесь:

      1. Если ``torch.__version__ is None`` — берём версию через
         ``importlib.metadata.version("torch")``; при неудаче — ставим "2.4.0".
      2. Оборачиваем ``packaging.version.parse`` и ``Version`` в safe-версии,
         которые принимают ``None`` как "0.0.0". Патч ставится ровно один раз;
         повторный вызов — no-op.
    """
    try:
        import sys as _sys
        _torch_mod = _sys.modules.get("torch")
        if _torch_mod is not None and getattr(_torch_mod, "__version__", None) is None:
            # transformers требует torch >= 2.4 — "0.0.0" сломает is_torch_available().
            try:
                import importlib.metadata as _imeta
                _torch_mod.__version__ = _imeta.version("torch")
            except (ImportError, Exception):  # noqa: BLE001 — PackageNotFoundError varies
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
                "[ml_sbert_bootstrap] packaging.version.parse/Version "
                "пропатчены глобально (torch.__version__ == None). "
                "Патч идемпотентен и применяется один раз.",
            )
    except (ImportError, AttributeError) as _patch_e:
        _log.debug("patch_torch_and_packaging: non-critical patch failed: %s", _patch_e)


# ---------------------------------------------------------------------------
# Retry-импорт SentenceTransformer с автоинжекцией builtins
# ---------------------------------------------------------------------------


def _clear_broken_tf_cache() -> None:
    """Удаляет сломанные transformers/sentence_transformers из sys.modules.

    Нужно когда ``is_torch_available()`` был закэширован как ``False``
    (например, из-за ``torch.__version__ is None`` при первом импорте).

    Сначала сбрасывается module-level кэш ``_torch_available`` /
    ``_torch_version`` в ``transformers.utils.import_utils``: иначе при
    повторном импорте transformers увидит "torch недоступен" даже после
    ``patch_torch_and_packaging()``.
    """
    import sys as _s
    _iu = _s.modules.get("transformers.utils.import_utils")
    if _iu is not None:
        for _attr in ("_torch_available", "_torch_version"):
            try:
                delattr(_iu, _attr)
            except (AttributeError, TypeError):
                pass
    for _k in list(_s.modules.keys()):
        if _k.startswith("sentence_transformers.") or _k == "sentence_transformers":
            _s.modules.pop(_k, None)
        elif _k.startswith("transformers.") or _k == "transformers":
            _s.modules.pop(_k, None)


def safe_import_sentence_transformers(
    bp: SBERTBuiltinsPatch,
    *,
    max_retries: int = SBERT_IMPORT_MAX_RETRIES,
    log_cb: Optional[Callable[[str], None]] = None,
) -> type:
    """Импортирует ``SentenceTransformer`` с retry-автоинжекцией builtins.

    Некоторые файлы transformers используют ``torch``/``nn``/``LRScheduler`` в
    аннотациях без импорта → Python 3.13 падает с NameError. Стратегия:
    при каждом NameError имя инжектируется через *bp*, кэш transformers
    сбрасывается, делается повторная попытка (до *max_retries*).

    *bp* — уже активированный ``SBERTBuiltinsPatch``-контекст; __exit__ очистит
    builtins автоматически.
    """
    bp.inject("torch")
    bp.inject("nn")

    _SentenceTransformer = None
    for _attempt in range(max_retries):
        try:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer
            break
        except ModuleNotFoundError as _mnfe:
            if "sentence_transformers" in str(_mnfe) or "sentence-transformers" in str(_mnfe):
                raise ImportError(
                    "Пакет sentence-transformers не установлен.\n"
                    "Установите: pip install sentence-transformers\n"
                    "Или нажмите кнопку «Установить» в разделе SBERT."
                ) from _mnfe
            raise ImportError(
                f"Ошибка импорта зависимости sentence-transformers: {_mnfe}\n"
                "Попробуйте: pip install sentence-transformers transformers --upgrade"
            ) from _mnfe
        except NameError as _ne:
            _m = _re.search(r"name '(\w+)' is not defined", str(_ne))
            if not _m:
                raise ImportError(
                    f"NameError при импорте sentence-transformers: {_ne}"
                ) from _ne
            bp.inject(_m.group(1))
            _clear_broken_tf_cache()
        except ImportError as _ie:
            raise ImportError(
                f"Ошибка загрузки sentence-transformers: {_ie}\n"
                "Попробуйте: pip install sentence-transformers transformers --upgrade"
            ) from _ie
    else:
        raise ImportError(
            f"Не удалось загрузить sentence-transformers после {max_retries} попыток.\n"
            "Попробуйте: pip install transformers --upgrade"
        )
    if log_cb is not None:
        log_cb("[SBERT] sentence-transformers импортирован.")
    return _SentenceTransformer
