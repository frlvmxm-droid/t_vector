# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec-файл для сборки BankReasonTrainer.exe (Windows, onedir).

Запуск:
    pyinstaller bank_reason_trainer.spec --clean

Результат: папка dist\BankReasonTrainer\ со всеми файлами.
Основной исполняемый файл: dist\BankReasonTrainer\BankReasonTrainer.exe
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from pathlib import Path

# ─── Зависимости ──────────────────────────────────────────────────────────────
# sklearn содержит скомпилированные .pyd-расширения — собираем все подмодули.
hidden_sklearn  = collect_submodules("sklearn")
hidden_joblib   = collect_submodules("joblib")
hidden_openpyxl = collect_submodules("openpyxl")

hidden_imports = hidden_sklearn + hidden_joblib + hidden_openpyxl + [
    # pandas / numpy подтягиваются через sklearn, но на всякий случай
    "pandas",
    "numpy",
    # tkinter — обычно встроен в Python для Windows, но явно прописываем
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
]

# Данные scikit-learn (JSON-файлы, bundle-ресурсы)
datas_sklearn = collect_data_files("sklearn")

# Фоновое изображение интерфейса
datas_bg = [("background.png", ".")]
datas_icons = []
for _p in (Path("ui/app_icon.png"), Path("ui/icon.png"), Path("app_icon.png"), Path("icon.png")):
    if _p.exists():
        datas_icons.append((str(_p), str(_p.parent) if str(_p.parent) != "." else "."))

icon_path = None
for _ico in (Path("ui/app_icon.ico"), Path("ui/icon.ico"), Path("app_icon.ico"), Path("icon.ico")):
    if _ico.exists():
        icon_path = str(_ico)
        break

# ─── Анализ зависимостей ──────────────────────────────────────────────────────
a = Analysis(
    ["bootstrap_run.py"],
    pathex=[],
    binaries=[],
    datas=datas_sklearn + datas_bg + datas_icons,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Исключаем тяжёлые пакеты, которые приложению не нужны
    excludes=[
        "matplotlib", "PIL", "Pillow",
        "IPython", "notebook", "jupyterlab",
        "pytest", "setuptools", "distutils",
        "cryptography", "ssl",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,           # onedir: бинарники отдельно
    name="BankReasonTrainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                        # UPX-сжатие уменьшает размер папки
    upx_exclude=[],
    console=False,                   # без консоли — GUI-приложение
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BankReasonTrainer",
)
