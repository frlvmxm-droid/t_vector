"""UI E2E smoke test — bootstraps App() under Xvfb to prove that the
4 mixins (DepsTab/TrainTab/ApplyTab/ClusterTab) survive a real Tk
init, not just `import`. Static linting can't detect attribute lookups
that happen only after `__init__` walks the MRO.

Skipped when:
  * tkinter is not installed (no Tk binding в headless-image),
  * customtkinter is not installed (главная зависимость UI-стека),
  * DISPLAY is not set AND xvfb-run wasn't used to launch pytest.

Локально запуск:

    xvfb-run -a pytest tests/test_ui_smoke.py -v

В CI рецепт лежит в quality-gates.yml (ubuntu-latest подхватывает
xvfb-run из репозиториев).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ui_stack_available() -> tuple[bool, str]:
    """Возвращает (available, reason). reason пуст, если стек доступен."""
    if os.environ.get("DISPLAY", "") == "":
        return False, "no DISPLAY (run via xvfb-run -a)"
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        return False, f"tkinter не установлен: {exc}"
    try:
        import customtkinter  # noqa: F401
    except ImportError as exc:
        return False, f"customtkinter не установлен: {exc}"
    return True, ""


_OK, _REASON = _ui_stack_available()


@pytest.mark.slow
@pytest.mark.skipif(not _OK, reason=_REASON)
def test_app_boots_and_exposes_three_tabs() -> None:
    """App() поднимается, формирует 3 вкладки и корректно закрывается.

    Цель smoke-теста — не покрытие, а run-time проверка того, что
    атрибуты, добавляемые мixin-ами через `self._build_ui()`, не
    рассыпаются от drift-а классовых полей.
    """
    from app import App  # noqa: WPS433 — lazy import (Tk init happens in App)

    app = App()
    try:
        assert hasattr(app, "train_file"), "TrainTabMixin не инициализирован"
        assert hasattr(app, "apply_file"), "ApplyTabMixin не инициализирован"
        assert hasattr(app, "cluster_files") or hasattr(app, "cluster_file"), (
            "ClusterTabMixin не инициализирован"
        )
        assert hasattr(app, "_run_active_tab"), "F5-handler не подключён"
        # Окно должно быть в состоянии normal/withdrawn — главное, что
        # _build_ui() прошёл без исключений.
        state = app.state()
        assert state in {"normal", "withdrawn", "iconic"}, f"unexpected state: {state}"
    finally:
        try:
            app.destroy()
        except Exception:  # noqa: BLE001 — Tk-cleanup best-effort
            pass
