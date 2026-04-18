# -*- coding: utf-8 -*-
"""
bootstrap_run.py — единая точка входа для Windows и Linux.

Режим 1 — Python (run_app.bat / run.sh / python bootstrap_run.py):
  Определяет ОС, проверяет версию Python, наличие tkinter и pip-зависимостей,
  при необходимости устанавливает зависимости, затем запускает GUI.

Режим 2 — PyInstaller .exe (BankReasonTrainer.exe):
  Зависимости уже в бандле — все проверки пропускаются, GUI запускается сразу.
"""

import sys
import io
import inspect
import platform
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Python 3.11+ удалил inspect.getargspec; некоторые старые библиотеки
# (например, старые версии scipy/pymorphy2) вызывают его напрямую.
# Патчим до любых импортов зависимостей.
# ---------------------------------------------------------------------------
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

# ---------------------------------------------------------------------------
# Windows: переключаем stdout/stderr на UTF-8, чтобы кириллица в консоли
# не вызывала UnicodeEncodeError и не обрывала вывод ошибок.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name, None)
        if _stream is not None and hasattr(_stream, "buffer"):
            setattr(
                sys,
                _stream_name,
                io.TextIOWrapper(_stream.buffer, encoding="utf-8", errors="replace", line_buffering=True),
            )

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
MIN_PYTHON = (3, 9)
_BASE_DIR  = Path(__file__).resolve().parent
REQ_FILE   = _BASE_DIR / "requirements.txt"
LOG_FILE   = _BASE_DIR / "run_app.log"
_STAMP_DIR = Path.home() / ".classification_tool"
_STAMP_FILE = _STAMP_DIR / ".deps_stamp"  # хэш requirements.txt после последней установки

# Пакеты для быстрой проверки: (import_name, pip_package_name, min_version_tuple)
# Используется только как дополнительная быстрая проверка; основная проверка —
# через _parse_mandatory_reqs() из requirements.txt.
_CORE_DEPS = [
    ("pandas",         "pandas",          (2, 1)),
    ("openpyxl",       "openpyxl",        (3, 1)),
    ("sklearn",        "scikit-learn",    (1, 4)),
    ("joblib",         "joblib",          (1, 3)),
    ("PIL",            "Pillow",          (10, 0)),
    ("numpy",          "numpy",           (1, 24)),
    ("psutil",         "psutil",          (5, 9)),
    ("scipy",          "scipy",           (1, 11)),
    ("huggingface_hub","huggingface_hub", (0, 20)),
    ("cryptography",   "cryptography",    (42, 0)),
    ("safetensors",    "safetensors",     (0, 4)),
    ("customtkinter",  "customtkinter",   (5, 2)),
    ("matplotlib",     "matplotlib",      (3, 7)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _wait_and_exit(code: int = 1) -> None:
    """Пауза перед выходом — чтобы пользователь успел прочитать ошибку."""
    try:
        input("\nНажмите Enter для выхода…")
    except (EOFError, OSError):
        pass
    sys.exit(code)


def _log_error(text: str) -> None:
    """Дублирует сообщение об ошибке в run_app.log рядом со скриптом."""
    try:
        import datetime
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
            f.write(text + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Проверки окружения
# ---------------------------------------------------------------------------
def _check_python_version() -> None:
    """Проверяет Python >= MIN_PYTHON."""
    if sys.version_info < MIN_PYTHON:
        msg = (
            f"[ОШИБКА] Требуется Python {'.'.join(map(str, MIN_PYTHON))} или новее.\n"
            f"         Установлена версия: Python {sys.version.split()[0]}\n"
            f"         Скачайте: https://www.python.org/downloads/"
        )
        print(msg)
        _log_error(msg)
        _wait_and_exit()


def _check_tkinter() -> None:
    """
    Проверяет доступность tkinter.
    На большинстве Linux-дистрибутивов tkinter — отдельный пакет (python3-tk).
    """
    try:
        import tkinter  # noqa: F401
        return
    except ImportError:
        pass

    system = platform.system()
    print("[ОШИБКА] tkinter не найден — он нужен для GUI.")
    print("         tkinter входит в стандартную библиотеку Python,")
    print("         но на некоторых Linux-дистрибутивах устанавливается отдельно.\n")

    if system == "Linux":
        distro_id = _get_distro_id()
        if distro_id in ("ubuntu", "debian", "linuxmint", "pop", "elementary"):
            print("  Установите командой:")
            print("    sudo apt-get install python3-tk")
        elif distro_id in ("fedora",):
            print("  Установите командой:")
            print("    sudo dnf install python3-tkinter")
        elif distro_id in ("rhel", "centos", "almalinux", "rocky", "ol"):
            print("  Установите командой:")
            print("    sudo dnf install python3-tkinter")
        elif distro_id in ("arch", "manjaro", "endeavouros", "garuda"):
            print("  Установите командой:")
            print("    sudo pacman -S tk")
        elif distro_id in ("opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles"):
            print("  Установите командой:")
            print("    sudo zypper install python3-tk")
        else:
            print("  Для Debian/Ubuntu:    sudo apt-get install python3-tk")
            print("  Для Fedora/RHEL:      sudo dnf install python3-tkinter")
            print("  Для Arch/Manjaro:     sudo pacman -S tk")
            print("  Для openSUSE:         sudo zypper install python3-tk")
    elif system == "Darwin":
        print("  Рекомендуется установить Python через Homebrew:")
        print("    brew install python-tk")
        print("  Или скачайте Python с https://www.python.org/downloads/")
    elif system == "Windows":
        print("  Переустановите Python, убедившись что флажок «tcl/tk»")
        print("  отмечен в разделе Optional Features установщика.")
    _log_error("[ОШИБКА] tkinter не найден.")
    _wait_and_exit()


def _get_distro_id() -> str:
    """Возвращает идентификатор Linux-дистрибутива в нижнем регистре."""
    try:
        # Python 3.10+ — platform.freedesktop_os_release()
        info = platform.freedesktop_os_release()
        return info.get("ID", "").lower()
    except AttributeError:
        pass
    # Fallback: читаем /etc/os-release вручную
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("ID="):
                return line.split("=", 1)[1].strip().strip('"').lower()
    except Exception:
        pass
    return ""


def _check_version(pkg_name: str, min_ver: tuple) -> bool:
    """Возвращает True, если установленная версия пакета >= min_ver."""
    try:
        from importlib.metadata import version as pkg_version
        ver_str = pkg_version(pkg_name)
        ver_tuple = tuple(int(x) for x in ver_str.split(".")[:len(min_ver)] if x.isdigit())
        return (ver_tuple or (0,)) >= min_ver
    except Exception:
        return True  # не удалось проверить — не блокируем


def _req_file_hash() -> str:
    """SHA-256 (первые 16 байт, hex) от содержимого requirements.txt."""
    import hashlib
    try:
        return hashlib.sha256(REQ_FILE.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def _stamp_matches() -> bool:
    """True — requirements.txt не менялся с момента последней успешной установки."""
    try:
        stamp = _STAMP_FILE.read_text(encoding="utf-8").strip()
        # Формат: "<hash>|<python_exe_hash>"
        parts = stamp.split("|")
        if len(parts) != 2:
            return False
        req_hash, py_hash = parts
        import hashlib
        cur_py_hash = hashlib.sha256(sys.executable.encode()).hexdigest()[:16]
        return req_hash == _req_file_hash() and py_hash == cur_py_hash
    except Exception:
        return False


def _write_stamp() -> None:
    """Сохраняет хэш requirements.txt + текущего интерпретатора."""
    import hashlib
    try:
        _STAMP_DIR.mkdir(parents=True, exist_ok=True)
        py_hash = hashlib.sha256(sys.executable.encode()).hexdigest()[:16]
        _STAMP_FILE.write_text(f"{_req_file_hash()}|{py_hash}", encoding="utf-8")
    except Exception:
        pass


def _run_pip_install(extra_args: list = ()) -> bool:
    """Запускает pip install -r requirements.txt. Возвращает True при успехе."""
    cmd_base = [sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)]
    cmd_base += list(extra_args)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(cmd_base)
        return True
    except subprocess.CalledProcessError:
        return False


def _ensure_deps() -> None:
    """Проверяет наличие всех обязательных зависимостей, при необходимости устанавливает.

    Алгоритм:
    1. Если stamp-файл совпадает с requirements.txt — deps уже установлены, быстрый выход.
    2. Иначе — проверяем _CORE_DEPS на наличие/версию.
    3. Если что-то отсутствует — запускаем pip install -r requirements.txt.
    4. Записываем stamp при успехе.
    """
    # Быстрый путь: зависимости уже были установлены для этого requirements.txt
    if _stamp_matches():
        print("[OK]  Все зависимости найдены.")
        return

    # Полная проверка
    missing_pkgs: list = []
    outdated_pkgs: list = []
    for mod, pkg, min_ver in _CORE_DEPS:
        try:
            __import__(mod)
        except ImportError:
            missing_pkgs.append(pkg)
            continue
        if not _check_version(pkg, min_ver):
            ver_str = ".".join(map(str, min_ver))
            print(f"[INFO] {pkg} устарел — нужна версия >={ver_str}")
            outdated_pkgs.append(pkg)

    needs_install = missing_pkgs or outdated_pkgs
    if not needs_install:
        print("[OK]  Все зависимости найдены.")
        _write_stamp()
        return

    if missing_pkgs:
        print(f"[INFO] Отсутствуют пакеты    : {', '.join(missing_pkgs)}")
    if outdated_pkgs:
        print(f"[INFO] Устаревшие пакеты     : {', '.join(outdated_pkgs)}")
    print("[INFO] Устанавливаю зависимости из requirements.txt…\n")

    ok = _run_pip_install()
    if not ok:
        print("[INFO] Стандартная установка не удалась, пробую --user …")
        ok = _run_pip_install(["--user"])

    if not ok:
        msg = (
            "[ОШИБКА] Установка зависимостей завершилась с ошибкой.\n"
            "         Попробуйте вручную:\n"
            f"           pip install -r {REQ_FILE}\n"
            "         или с флагом --user:\n"
            f"           pip install --user -r {REQ_FILE}"
        )
        print(f"\n{msg}")
        _log_error(msg)
        _wait_and_exit()

    # Пост-установочная проверка
    still_missing = [
        pkg for mod, pkg, _ in _CORE_DEPS
        if not _try_import(mod)
    ]
    if still_missing:
        msg = (
            f"[ОШИБКА] После установки пакеты всё ещё недоступны: {', '.join(still_missing)}\n"
            f"         Возможно, установка прошла в другой Python или окружение.\n"
            f"         Текущий интерпретатор: {sys.executable}\n"
            f"         Попробуйте: {sys.executable} -m pip install -r {REQ_FILE}"
        )
        print(f"\n{msg}")
        _log_error(msg)
        _wait_and_exit()

    print("\n[OK]  Все зависимости установлены.")
    _write_stamp()


def _try_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Запуск приложения с умным обработчиком ошибок
# ---------------------------------------------------------------------------
def _launch_app(retry: bool = True) -> None:
    """Импортирует и запускает App. При ModuleNotFoundError — пробует auto-install."""
    try:
        from app import App
        App().mainloop()
    except SystemExit:
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        # Отсутствующий пакет — сбрасываем stamp и пробуем установить
        missing_mod = getattr(exc, "name", None) or str(exc)
        print(f"\n[ОШИБКА] Отсутствует модуль: {missing_mod}")
        if retry:
            _STAMP_FILE.unlink(missing_ok=True)
            print("[INFO] Запускаю автоматическую установку зависимостей…\n")
            ok = _run_pip_install()
            if not ok:
                ok = _run_pip_install(["--user"])
            if ok:
                print("\n[OK]  Зависимости установлены. Перезапускаю приложение…\n")
                _write_stamp()
                _launch_app(retry=False)
                return
        import traceback
        tb_text = traceback.format_exc()
        msg = (
            "\n" + "=" * 60 + "\n"
            "  [ОШИБКА] Отсутствует зависимость\n"
            "=" * 60 + "\n"
            + tb_text
            + "\nРешение:\n"
            f"  pip install -r {REQ_FILE}\n"
            f"\nПодробности сохранены в: {LOG_FILE}"
        )
        print(msg)
        _log_error(msg)
        _wait_and_exit(1)
    except Exception:
        import traceback
        tb_text = traceback.format_exc()
        header = (
            "\n" + "=" * 60 + "\n"
            "  [КРИТИЧЕСКАЯ ОШИБКА] Приложение не смогло запуститься\n"
            + "=" * 60
        )
        hints = (
            "\nВозможные причины:\n"
            "  • Несовместимая версия пакета (попробуйте: pip install -r requirements.txt --upgrade)\n"
            "  • Отсутствует зависимость (scikit-learn, pandas, openpyxl, Pillow)\n"
            "  • Повреждён файл модели или настроек\n"
            f"\nПодробности сохранены в: {LOG_FILE}"
        )
        print(header)
        print(tb_text)
        print(hints)
        _log_error(header + "\n" + tb_text + hints)
        _wait_and_exit(1)


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------
def main() -> None:
    system  = platform.system()
    os_ver  = platform.release()
    machine = platform.machine()
    py_ver  = sys.version.split()[0]

    _banner("BankReasonTrainer — запуск")
    print(f"ОС    : {system} {os_ver} ({machine})")
    print(f"Python: {py_ver}  |  {sys.executable}")

    # ── Детектор заглушки Microsoft Store ────────────────────────────────────
    # На Windows 10/11, если Python не установлен из официального дистрибутива,
    # «python» в PATH может ссылаться на заглушку из WindowsApps — она лишь
    # открывает Microsoft Store и завершается без ошибок (exit 0), поэтому
    # батник не видит проблемы. Детектируем это и сообщаем явно.
    if system == "Windows" and "WindowsApps" in str(Path(sys.executable)):
        msg = (
            "[ОШИБКА] Запущена заглушка Python из Microsoft Store.\n"
            "         Она не содержит реального интерпретатора.\n\n"
            "  Установите Python 3.9+ с официального сайта:\n"
            "    https://www.python.org/downloads/\n"
            "  При установке отметьте «Add Python to PATH» и «py launcher».\n"
            "  После установки запустите run_app.bat заново."
        )
        print(msg)
        _log_error(msg)
        _wait_and_exit(1)

    # ── DPI-осведомлённость для Windows 10/11 ────────────────────────────────
    # Без этого tkinter на HiDPI-экранах масштабирует окно операционной системой
    # (размытое масштабирование) или размещает его за пределами видимой области.
    # Устанавливаем PROCESS_PER_MONITOR_DPI_AWARE (значение 2) через shcore,
    # при неудаче — устаревший SetProcessDPIAware через user32.
    if system == "Windows":
        try:
            import ctypes
            # PROCESS_SYSTEM_DPI_AWARE (1): окно рендерится в системном DPI,
            # Windows не масштабирует его автоматически (убирает размытость и
            # предотвращает появление окна за пределами видимой области экрана).
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    # В PyInstaller-бандле всё уже внутри — пропускаем проверки окружения
    if getattr(sys, "frozen", False):
        print("Режим : PyInstaller (.exe)")
    else:
        print(f"Режим : Python-скрипт")
        _check_python_version()
        _check_tkinter()
        _ensure_deps()

    print("\n[ЗАПУСК] Загружаю интерфейс…")
    _launch_app()


if __name__ == "__main__":
    main()
