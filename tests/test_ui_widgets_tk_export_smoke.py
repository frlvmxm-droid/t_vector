"""CI-guard: все top-level Tk-классы из ``ui_widgets_tk.py`` присутствуют
как атрибуты импортированного модуля.

Ловит два класса дефектов:
1. Класс определён в модуле, но забыт в callers (``app_*.py`` и т.п.).
2. Класс переименован/удалён, но внешний код ещё на него ссылается.

Используем AST-парсинг исходника (а не ``inspect.isclass``) чтобы тест
работал в CI под conftest'овским tkinter-mock'ом: под mock'ом
``class X(tk.Frame)`` превращается в MagicMock-инстанс, а не в
полноценный класс — ``inspect.isclass`` его не видит.
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ui_widgets_tk.py"


def _public_classes_from_source() -> list[str]:
    """Parse ui_widgets_tk.py and return names of top-level public classes."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    return sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    )


@pytest.fixture(scope="module")
def tk_module():
    try:
        return importlib.import_module("ui_widgets_tk")
    except ImportError as exc:
        pytest.skip(f"ui_widgets_tk недоступен (нет tkinter/PIL?): {exc}")


def test_ui_widgets_tk_file_exists():
    assert MODULE_PATH.is_file(), (
        f"ui_widgets_tk.py отсутствует по пути {MODULE_PATH}"
    )


def test_all_tk_classes_exported(tk_module):
    """Каждое ``class Foo(...):`` на top-level доступно как ``ui_widgets_tk.Foo``."""
    classes = _public_classes_from_source()
    assert classes, "ui_widgets_tk.py не содержит публичных классов"
    missing = [name for name in classes if not hasattr(tk_module, name)]
    assert not missing, (
        f"Классы определены, но не экспортируются: {missing}. "
        f"Проверь, что определение не обёрнуто в try/except ImportError."
    )


def test_known_legacy_classes_present():
    """Baseline — классы, от которых зависит desktop-код (``app_*.py``).

    Если один из них исчез — breaking change для ``app.py``,
    ``app_train.py`` и пр.; обновить callers и этот список вместе.
    """
    required = {
        "Tooltip",
        "ScrollableFrame",
        "GradientBackground",
        "ImageBackground",
        "RoundedButton",
        "RoundedCard",
        "CollapsibleSection",
        "ToggleSwitch",
        "PillTabBar",
        "StatusPill",
    }
    present = set(_public_classes_from_source())
    missing = required - present
    assert not missing, (
        f"В ui_widgets_tk.py отсутствуют обязательные классы: {sorted(missing)}"
    )
