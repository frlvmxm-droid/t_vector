# -*- coding: utf-8 -*-
"""
Централизованное логирование приложения.

Предоставляет:
  get_logger(name)         — получить именованный logger в иерархии «app.*»
  setup_file_logging(...)  — подключить RotatingFileHandler к корневому логгеру «app»
  UILogHandler             — handler, пересылающий записи в UI-callback (tkinter-безопасен)
  redirect_warnings()      — перенаправить warnings.warn → logging.warning

Использование в модуле::

    from app_logger import get_logger
    log = get_logger(__name__)
    log.warning("Что-то пошло не так: %s", detail)

Инициализация при старте приложения::

    from app_logger import setup_file_logging
    setup_file_logging()          # пишет в logs/app.log
"""
from __future__ import annotations

import logging
import logging.handlers
import warnings
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

_ROOT_LOGGER_NAME = "app"
_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_LOG_FILE = "app.log"
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024   # 5 МБ
_DEFAULT_BACKUP_COUNT = 3

_FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Фабрика логгеров
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Возвращает logger в иерархии «app.*».

    Если *name* уже начинается с «app», используется как есть.
    Иначе возвращается ``logging.getLogger("app." + name)``.

    Пример::

        log = get_logger(__name__)   # app.ml_core, app.app_apply, …
    """
    if name == _ROOT_LOGGER_NAME or name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


# ---------------------------------------------------------------------------
# Настройка файлового логирования
# ---------------------------------------------------------------------------

def setup_file_logging(
    log_dir: Path = _DEFAULT_LOG_DIR,
    log_file: str = _DEFAULT_LOG_FILE,
    level: int = logging.DEBUG,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Подключает RotatingFileHandler к корневому логгеру «app».

    Безопасно вызывать многократно — повторный вызов не добавляет дублирующих handlers.

    Args:
        log_dir:      директория для файла логов (создаётся если не существует).
        log_file:     имя файла лога.
        level:        минимальный уровень для файлового handler.
        max_bytes:    максимальный размер файла до ротации.
        backup_count: количество архивных файлов.

    Returns:
        Корневой логгер «app».
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)  # handler-ы фильтруют по своему уровню

    # Проверяем, нет ли уже RotatingFileHandler с тем же путём
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # В крайнем случае — молчим, файловые логи не критичны

    log_path = log_dir / log_file
    for h in root.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            if getattr(h, "baseFilename", None) == str(log_path.resolve()):
                return root   # уже настроен

    try:
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)
    except OSError:
        pass  # Нет прав на запись — тихо пропускаем

    return root


# ---------------------------------------------------------------------------
# UI Handler
# ---------------------------------------------------------------------------

class UILogHandler(logging.Handler):
    """Logging handler, пересылающий записи в UI-callback.

    Предназначен для отображения WARNING+ сообщений в Tkinter-виджете.
    Callback вызывается напрямую (без `after(0, ...)`), поэтому вызывающий
    код должен убедиться, что callback потокобезопасен (использует `after`
    или `queue`).

    Пример::

        handler = UILogHandler(callback=self.log_apply)
        handler.setLevel(logging.WARNING)
        get_logger("app").addHandler(handler)
        ...
        get_logger("app").removeHandler(handler)   # убрать при закрытии окна
    """

    def __init__(
        self,
        callback: Callable[[str], None],
        level: int = logging.WARNING,
        fmt: str = "%(levelname)s: %(message)s",
    ) -> None:
        super().__init__(level)
        self.callback = callback
        self.setFormatter(logging.Formatter(fmt))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Перенаправление warnings → logging
# ---------------------------------------------------------------------------

def redirect_warnings() -> None:
    """Перенаправляет warnings.warn → logging.warning.

    После вызова все предупреждения Python попадают в логгер «app.warnings»
    (и далее — в файл / UI), а не в stderr.

    Вызывать один раз при старте приложения.
    """
    _warn_logger = logging.getLogger(f"{_ROOT_LOGGER_NAME}.warnings")

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        _warn_logger.warning(
            "%s:%d: %s: %s",
            filename, lineno, category.__name__, message,
        )

    warnings.showwarning = _showwarning


# ---------------------------------------------------------------------------
# Удобный синтаксический сахар: логгер уровня модуля для пакета
# ---------------------------------------------------------------------------

# Создаём корневой логгер сразу, чтобы иерархия была консистентной
# (без handlers по умолчанию — их добавляет setup_file_logging / UILogHandler)
_app_root = logging.getLogger(_ROOT_LOGGER_NAME)
if not _app_root.handlers:
    # Добавляем NullHandler, чтобы Python не выводил «No handlers could be found»
    _app_root.addHandler(logging.NullHandler())
