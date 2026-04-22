# -*- coding: utf-8 -*-
"""
setup_env.py — единый скрипт подготовки окружения для BankReasonTrainer.

Что делает:
  1. Проверяет Python >= 3.9
  2. Проверяет/предлагает установить tkinter
  3. Устанавливает обязательные пакеты из requirements.txt
  4. Предлагает установить опциональные пакеты (sentence-transformers, pymorphy2, torch)
  5. Скачивает SBERT-модель по умолчанию (rubert-tiny2, ~45 МБ)

Запуск:
    python setup_env.py            # интерактивный режим
    python setup_env.py --auto     # тихая установка всего необходимого без вопросов
    python setup_env.py --check    # только проверка, без установки
"""
from __future__ import annotations

import sys
import subprocess
import platform
import pathlib
import argparse
import shlex

BASE_DIR = pathlib.Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from config.ml_constants import hf_cache_key  # noqa: E402
REQ_FILE = BASE_DIR / "requirements.txt"
SBERT_DIR = BASE_DIR / "sbert_models"

# ── Минимальные версии обязательных пакетов ──────────────────────────────────
REQUIRED = [
    ("pandas",          "pandas",               (2, 1)),
    ("openpyxl",        "openpyxl",             (3, 1)),
    ("sklearn",         "scikit-learn",         (1, 4)),
    ("joblib",          "joblib",               (1, 3)),
    ("PIL",             "Pillow",               (10, 0)),
    ("numpy",           "numpy",                (1, 24)),
    ("psutil",          "psutil",               (5, 9)),
    ("scipy",           "scipy",                (1, 11)),
    ("huggingface_hub", "huggingface_hub",      (0, 20)),
]

# Модель по умолчанию — самая лёгкая (~45 МБ)
DEFAULT_SBERT_MODEL = "cointegrated/rubert-tiny2"


# =============================================================================
# Helpers
# =============================================================================

def _sep(char="=", width=62):
    print(char * width)


def _header(title: str):
    _sep()
    print(f"  {title}")
    _sep()


def _ok(msg):   print(f"  [OK]   {msg}")
def _info(msg): print(f"  [INFO] {msg}")
def _warn(msg): print(f"  [WARN] {msg}")
def _err(msg):  print(f"  [ERR]  {msg}")


def _ask(question: str, default_yes: bool = True) -> bool:
    hint = "[Y/n]" if default_yes else "[y/N]"
    try:
        ans = input(f"\n  {question} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default_yes
    if ans == "":
        return default_yes
    return ans in ("y", "yes", "да", "д")


def _pip(*args, quiet=False):
    """Запускает pip через текущий интерпретатор."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd += ["-q"]
    cmd += list(args)
    return subprocess.call(cmd)


def _pkg_version(pip_name: str) -> tuple:
    try:
        from importlib.metadata import version as _v
        parts = _v(pip_name).split(".")
        return tuple(int(x) for x in parts[:3] if x.isdigit()) or (0,)
    except Exception:
        return (0,)


def _distro_id() -> str:
    try:
        return platform.freedesktop_os_release().get("ID", "").lower()
    except AttributeError:
        pass
    try:
        for line in pathlib.Path("/etc/os-release").read_text().splitlines():
            if line.startswith("ID="):
                return line.split("=", 1)[1].strip().strip('"').lower()
    except Exception:
        pass
    return ""


# =============================================================================
# Шаг 1 — версия Python
# =============================================================================

def check_python() -> bool:
    _header("Шаг 1/5 — Версия Python")
    ver = sys.version_info
    print(f"  Найден: Python {ver.major}.{ver.minor}.{ver.micro}  ({sys.executable})")
    if ver < (3, 9):
        _err("Требуется Python 3.9 или новее.")
        print("  Скачайте: https://www.python.org/downloads/")
        return False
    _ok(f"Python {ver.major}.{ver.minor} — подходит.")
    return True


# =============================================================================
# Шаг 2 — tkinter
# =============================================================================

def check_tkinter(auto: bool, check_only: bool = False) -> bool:
    _header("Шаг 2/5 — tkinter (GUI)")
    try:
        import tkinter  # noqa: F401
        _ok("tkinter доступен.")
        return True
    except ImportError:
        pass

    _warn("tkinter не найден.")
    os_name = platform.system()
    distro  = _distro_id()

    install_cmds = {
        ("ubuntu", "debian", "linuxmint", "pop", "elementary"):
            "sudo apt-get install -y python3-tk",
        ("fedora",):
            "sudo dnf install -y python3-tkinter",
        ("rhel", "centos", "almalinux", "rocky", "ol"):
            "sudo dnf install -y python3-tkinter",
        ("arch", "manjaro", "endeavouros", "garuda"):
            "sudo pacman -S --noconfirm tk",
        ("opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles"):
            "sudo zypper install -y python3-tk",
    }

    cmd = None
    for distros, install_cmd in install_cmds.items():
        if distro in distros:
            cmd = install_cmd
            break

    if os_name == "Linux":
        if cmd is None:
            cmd = "sudo apt-get install python3-tk  # или аналог для вашего дистрибутива"
        _info(f"Команда установки: {cmd}")
        if check_only:
            _warn("Запустите без --check чтобы установить.")
            return False
        if not auto:
            do_it = _ask("Выполнить установку сейчас?", default_yes=True)
        else:
            do_it = True

        if do_it and "# или" not in cmd:
            ret = subprocess.call(shlex.split(cmd))
            if ret == 0:
                try:
                    import tkinter  # noqa: F401
                    _ok("tkinter успешно установлен.")
                    return True
                except ImportError:
                    pass
            _err("tkinter всё ещё недоступен. Установите вручную и перезапустите скрипт.")
            return False
    elif os_name == "Darwin":
        _info("Рекомендуется:  brew install python-tk")
        _info("Или скачайте Python с https://www.python.org/downloads/")
    elif os_name == "Windows":
        _info("Переустановите Python, отметив «tcl/tk» в Optional Features.")

    return False


# =============================================================================
# Шаг 3 — обязательные pip-пакеты
# =============================================================================

def ensure_required(auto: bool, check_only: bool) -> bool:
    _header("Шаг 3/5 — Обязательные пакеты")

    missing = []
    outdated = []

    for mod, pkg, min_ver in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            _warn(f"Не найден:  {pkg}")
            missing.append(pkg)
            continue
        installed = _pkg_version(pkg)
        if installed < min_ver:
            min_str = ".".join(map(str, min_ver))
            ins_str = ".".join(map(str, installed))
            _warn(f"Устарел:    {pkg} {ins_str} (нужен >= {min_str})")
            outdated.append(pkg)
        else:
            ins_str = ".".join(map(str, installed))
            _ok(f"{pkg} {ins_str}")

    need_install = missing + outdated

    if not need_install:
        _ok("Все обязательные пакеты в порядке.")
        return True

    if check_only:
        _err(f"Не установлены/устарели: {', '.join(need_install)}")
        return False

    _info(f"Будет установлено/обновлено: {', '.join(need_install)}")
    if not auto:
        if not _ask("Установить из requirements.txt?", default_yes=True):
            _warn("Пропущено. Пакеты нужны для запуска.")
            return False

    # Сначала обновим pip
    subprocess.call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "-q", "pip"]
    )

    ret = _pip("-r", str(REQ_FILE))
    if ret != 0:
        _info("Стандартная установка не удалась, пробую --user…")
        ret = _pip("--user", "-r", str(REQ_FILE))

    if ret != 0:
        _err(f"Установка завершилась с ошибкой (код {ret}).")
        _info(f"Попробуйте вручную: pip install -r {REQ_FILE}")
        return False

    # Проверяем после установки
    still = [pkg for mod, pkg, _ in REQUIRED if _silent_import(mod)]
    if still:
        _err(f"После установки всё ещё недоступны: {', '.join(still)}")
        _info(f"Интерпретатор: {sys.executable}")
        return False

    _ok("Обязательные пакеты установлены.")
    return True


def _silent_import(mod: str) -> bool:
    """Возвращает True если импорт НЕ удался."""
    try:
        __import__(mod)
        return False
    except ImportError:
        return True


# =============================================================================
# Шаг 4 — опциональные пакеты
# =============================================================================

def ensure_optional(auto: bool, check_only: bool):
    _header("Шаг 4/5 — Опциональные пакеты")

    # ── sentence-transformers ─────────────────────────────────────────────────
    try:
        import sentence_transformers  # noqa: F401
        ver = _pkg_version("sentence-transformers")
        _ok(f"sentence-transformers {'.'.join(map(str, ver))} — SBERT-векторизация доступна.")
        has_sbert = True
    except ImportError:
        _warn("sentence-transformers не установлен.")
        _info("Нужен для SBERT-режима обучения и кластеризации.")
        has_sbert = False
        if not check_only:
            if auto or _ask("Установить sentence-transformers?", default_yes=True):
                ret = _pip("sentence-transformers", quiet=True)
                if ret == 0:
                    _ok("sentence-transformers установлен.")
                    has_sbert = True
                else:
                    _warn("Установка не удалась. Продолжаем без SBERT.")

    # ── pymorphy2 ─────────────────────────────────────────────────────────────
    try:
        import pymorphy2  # noqa: F401
        _ok("pymorphy2 — лемматизация доступна.")
    except ImportError:
        _warn("pymorphy2 не установлен.")
        _info("Нужен для лемматизации русского текста (улучшает качество модели).")
        if not check_only:
            if auto or _ask("Установить pymorphy2?", default_yes=True):
                ret = _pip("pymorphy2", quiet=True)
                if ret == 0:
                    _ok("pymorphy2 установлен.")
                else:
                    _warn("Установка не удалась. Продолжаем без лемматизации.")

    # ── transformers + torch (T5-суммаризация) ────────────────────────────────
    try:
        import transformers  # noqa: F401
        ver = _pkg_version("transformers")
        _ok(f"transformers {'.'.join(map(str, ver))} — T5-суммаризация доступна.")
    except ImportError:
        _warn("transformers не установлен.")
        _info("Нужен только для T5-суммаризации диалогов (необязательно).")
        if not check_only:
            if not auto:
                if _ask("Установить transformers + torch (T5-суммаризация)?", default_yes=False):
                    ret = _pip("transformers>=4.40,<4.52", "torch", quiet=True)
                    if ret == 0:
                        _ok("transformers + torch установлены.")
                    else:
                        _warn("Установка не удалась.")
            # В --auto режиме transformers+torch НЕ ставим — он тяжёлый (~3 ГБ)

    return has_sbert


# =============================================================================
# Шаг 5 — скачивание SBERT-модели по умолчанию
# =============================================================================

def download_default_model(auto: bool, check_only: bool, has_sbert: bool):
    _header("Шаг 5/5 — SBERT-модель по умолчанию")

    if not has_sbert:
        _info("sentence-transformers не установлен — пропускаем.")
        return

    SBERT_DIR.mkdir(exist_ok=True)
    model_cache = SBERT_DIR / (hf_cache_key(DEFAULT_SBERT_MODEL))

    if model_cache.exists() and any(model_cache.iterdir()):
        _ok(f"Модель уже скачана: {DEFAULT_SBERT_MODEL}")
        _info(f"Путь: {model_cache}")
        return

    _info(f"Модель: {DEFAULT_SBERT_MODEL}  (~45 МБ)")
    _info("Будет скачана в папку sbert_models/ рядом со скриптом.")

    if check_only:
        _warn("Модель не скачана. Запустите без --check для загрузки.")
        return

    if not auto:
        if not _ask("Скачать модель сейчас?", default_yes=True):
            _warn("Пропущено. Модель будет скачана автоматически при первом обучении.")
            return

    try:
        from huggingface_hub import snapshot_download
        print(f"\n  Скачиваю {DEFAULT_SBERT_MODEL}…")
        snapshot_download(
            DEFAULT_SBERT_MODEL,
            cache_dir=str(SBERT_DIR),
            ignore_patterns=["*.h5", "*.ot", "flax_model*", "tf_model*",
                             "rust_model*", "onnx*"],
        )
        _ok(f"Модель сохранена в {SBERT_DIR}")
    except Exception as e:
        _warn(f"Не удалось скачать: {e}")
        _info("Модель будет скачана автоматически при первом использовании SBERT.")


# =============================================================================
# Итоговый отчёт
# =============================================================================

def print_summary():
    _sep()
    print()
    results = {}

    checks = [
        ("Python 3.9+",               "sys",                   None),
        ("tkinter",                    "tkinter",               None),
        ("pandas",                     "pandas",                None),
        ("scikit-learn",               "sklearn",               None),
        ("numpy",                      "numpy",                 None),
        ("scipy",                      "scipy",                 None),
        ("openpyxl",                   "openpyxl",              None),
        ("Pillow",                     "PIL",                   None),
        ("joblib",                     "joblib",                None),
        ("psutil",                     "psutil",                None),
        ("huggingface_hub",            "huggingface_hub",       None),
        ("sentence-transformers",      "sentence_transformers", "(опционально)"),
        ("pymorphy2",                  "pymorphy2",             "(опционально)"),
        ("transformers",               "transformers",          "(опционально)"),
    ]

    _header("Итоговый статус окружения")
    for label, mod, note in checks:
        try:
            __import__(mod)
            status = "OK "
        except ImportError:
            status = "---" if note else "ERR"
        suffix = f"  {note}" if note else ""
        print(f"  [{status}]  {label}{suffix}")

    # Модель
    model_cache = SBERT_DIR / (hf_cache_key(DEFAULT_SBERT_MODEL))
    has_model = model_cache.exists() and any(model_cache.iterdir())
    status = "OK " if has_model else "---"
    print(f"  [{status}]  SBERT-модель rubert-tiny2  (опционально)")

    print()
    _sep()
    print()
    print("  Для запуска web-UI:")
    print("    ./run_web.sh        (Linux / macOS)")
    print("    run_web.bat         (Windows)")
    print()
    _sep()


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Подготовка окружения для BankReasonTrainer"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Тихая установка без интерактивных вопросов"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Только проверка, без установки"
    )
    args = parser.parse_args()

    auto       = args.auto
    check_only = args.check

    print()
    _header("BankReasonTrainer — подготовка окружения")
    print(f"  ОС     : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python : {sys.version.split()[0]}  ({sys.executable})")
    if check_only:
        _info("Режим: только проверка (--check)")
    elif auto:
        _info("Режим: автоматическая установка (--auto)")
    print()

    ok = check_python()
    if not ok:
        sys.exit(1)

    tk_ok = check_tkinter(auto=auto, check_only=check_only)
    req_ok = ensure_required(auto=auto, check_only=check_only)
    has_sbert = ensure_optional(auto=auto, check_only=check_only)
    download_default_model(auto=auto, check_only=check_only, has_sbert=has_sbert)

    print_summary()

    if not tk_ok or not req_ok:
        _err("Есть нерешённые проблемы. Исправьте их перед запуском.")
        sys.exit(1)

    _ok("Окружение готово. Web-UI: ./run_web.sh  (или run_web.bat на Windows)")
    print()


if __name__ == "__main__":
    main()
