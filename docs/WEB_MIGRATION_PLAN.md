# План миграции BankReasonTrainer на Web-UI

> Источник: синтез трёх аудитов (launch-paths, monolith decomposition,
> ctk_migration_pack design gap), выполненных 2026-04-22 на ветке
> `claude/code-review-assistant-0Fmdx`.

## Контекст

Исторически проект — desktop-Tkinter-приложение (`app.py`,
`app_train.py`, `app_apply.py`, `app_cluster.py`). Принято решение
перевести UI на браузерный вариант (Voilà + ipywidgets), сохранив
визуальный дизайн из `ctk_migration_pack/BankReasonTrainer.html`.

Сервисный слой (`app_train_service`, `apply_prediction_service`,
`cluster_workflow_service`, `app_cluster_pipeline`) уже оторван от
Tk и покрыт тестами. Web-UI (`ui_widgets/*`) построен поверх него,
имеет session save/restore и работает в Voilà cold-start за ~1.4с.

## Статус «что уже сделано»

| Область | Статус |
|---|---|
| Service-layer (train/apply/cluster) | ✅ complete |
| Web-UI panels (ipywidgets) | ✅ complete |
| Voilà cold-start | ✅ 72s → 1.4s (fix settings/artifacts/torch) |
| Session save/restore в web-UI | ✅ complete |
| `ui_widgets_tk.py` rename + CI-guard | ✅ complete |
| `_service_snap` unification (Fix C) | ✅ complete |
| JupyterHub admin-docs | ✅ `docs/JUPYTERHUB_UI.md` |
| **Спринт 1 — launcher'ы + quickstart** | ✅ complete (`run_web.sh`, `run_web.bat`, `docs/QUICKSTART_WEB_UI.md`, README §8.3) |

## Известные блокеры / gap'ы

### Блокеры запуска
- ✅ Закрыты в Спринте 1 (`run_web.sh`/`run_web.bat` + quickstart).
  `bootstrap_run.py` как отдельный desktop-entry уйдёт в Спринте 3.

### Дизайн-дыры vs ctk_migration_pack
- Только одна тема (Dark Teal) вместо трёх (добавить Paper, Amber-CRT).
- Нет theme-switcher в sidebar.
- `section_card` не поддерживает right-slot для header-кнопок.
- Нет helper'ов `field_label()` и `separator()`.
- `chip()` / `badge()` — не покрывают все 6 kind'ов (нет `accent`;
  канонический список: `default | accent | ok | warn | err | info`).

### Архитектурные долги
- **`app_cluster.run_cluster()`** — сейчас ~100 строк
  (`app_cluster.py:3410–3510`), уже делегирует в
  `ClusteringWorkflow.run()`. Старая оценка «951 строка» устарела.
  Будет удалён целиком вместе с `app_cluster.py` в Спринте 3.
- Deprecated shims в `core/` и `utils/` — мёртвый re-export код
  (адресуется в Спринте 3).
- `customtkinter`, `pystray` в `pyproject.toml` — удаляются в
  Спринте 3 (вместе с `uv lock` регенерацией).

## Решение: полный отказ от desktop

**Принято 2026-04-22.** Ранее в этой секции фиксировалась развилка
«полный отказ vs desktop-fallback»; выбран **полный отказ**.

**Обоснование:**
- Сервис-слой (`app_train_service`, `apply_prediction_service`,
  `cluster_workflow_service`, `app_cluster_pipeline`) уже headless
  и покрыт тестами (`tests/test_e2e_*`, `tests/test_workflow_contracts.py`).
- `app_cluster.run_cluster()` сократился до ~100 строк и уже
  делегирует в `ClusteringWorkflow.run()` — держать Tk-обвязку
  ради одной функции нерационально.
- Plotly-визуализации кластеров воспроизводимы в ipywidgets
  (FigureWidget); tray-icon — заменяется нативным браузерным треем /
  системным пином вкладки.

**Ожидаемый эффект:** минус ~750 KB Tk/CTK-кода, ~15–20k LoC,
`pyproject.toml` без `customtkinter`/`pystray`, CI без Xvfb.

**Последствия по спринтам:** решение разворачивает Спринт 3 в
финальный детальный план удаления (см. ниже). Спринты 2 и 4 от
выбора не зависят.

---

# Спринт 1 — Запуск «из коробки» (1 день)

**Цель**: пользователь на Windows или Linux делает `git clone` +
двойной клик по launcher'у — и получает работающий web-UI на
`http://localhost:8866/`.

## Задачи

1. **`run_web.bat`** (новый файл, корень проекта)
   - Детект Python: `py -3` → `python` → фатальная ошибка с ссылкой на python.org
   - Проверка `voila` + `ipywidgets` через `python -c "import …"` — если нет, `pip install -e ".[ui]"`
   - Свободный порт: 8866 по умолчанию, fallback через env `BRT_PORT`
   - Автооткрытие браузера: `start "" http://localhost:%PORT%/`
   - Exec: `python -m voila notebooks\ui.ipynb --port=%PORT% --no-browser`

2. **`run_web.sh`** (новый файл, корень проекта, chmod +x)
   - Bash, `set -euo pipefail`
   - Детект: `$PYTHON` → `python3.11` → `python3` → фатальная ошибка
   - Проверка deps, `pip install -e ".[ui]"` при необходимости
   - Автооткрытие: `xdg-open` / `open` (macOS) / ничего (в SSH-сессии)
   - `exec python -m voila notebooks/ui.ipynb --port=$PORT --no-browser`
   - Поддержка env: `BRT_PORT`, `BRT_HOST` (default `127.0.0.1`)

3. **`docs/QUICKSTART_WEB_UI.md`** (новый)
   - §1 Требования (Python 3.11+, свободный порт 8866)
   - §2 Windows: `run_web.bat`
   - §3 Linux / macOS: `./run_web.sh`
   - §4 Альтернатива — ручной запуск через `uv`
   - §5 Troubleshooting (port в использовании, firewall, ImportError)
   - §6 Остановка: Ctrl+C в терминале
   - §7 Что дальше — ссылки на `JUPYTERHUB_UI.md` для продакшна

4. **Обновить `README.md`** §8.3
   - Заменить ручной «uv sync + voila» recipe на:
     - Windows: `run_web.bat`
     - Linux: `./run_web.sh`
   - Ссылка на `docs/QUICKSTART_WEB_UI.md`

## Критерии готовности

- [ ] `./run_web.sh` на чистом Linux (Ubuntu 22.04) поднимает UI < 2 минут
- [ ] `run_web.bat` на Windows 10/11 поднимает UI (Python из Microsoft Store **не** принимается)
- [ ] Документ `QUICKSTART_WEB_UI.md` покрывает оба ОС
- [ ] README §8.3 ссылается на новые launcher'ы
- [ ] Оба скрипта корректно обрабатывают: нет Python / нет pip / занят порт / Ctrl+C

---

# Спринт 2 — Дизайн-паритет с ctk_migration_pack (1 день)

**Цель**: три темы + theme-switcher в sidebar + недостающие
design-компоненты, чтобы web-UI визуально совпадал с эталоном из
`ctk_migration_pack/BankReasonTrainer.html`.

## Задачи

1. **Палитры в `ui_widgets/theme.py`**
   - Module-level `PALETTES: dict[str, dict[str, str]]`:
     - `"dark-teal"` (текущая, оставить как default)
     - `"paper"` — светлая, из `ctk_migration_pack/ui_theme_ctk.py`
     - `"amber-crt"` — ретро, из того же файла
   - `apply_theme(name: str) -> None` — переключает module-level
     `COLORS` + регенерирует CSS через `inject_css()`

2. **Theme-switcher в sidebar** (`ui_widgets/notebook_app.py`)
   - Под hardware-card — трёх-кнопочная группа `[Teal · Paper · CRT]`
   - `on_click` → `apply_theme(name)` + `inject_css()` re-run
   - Персистить выбор в `~/.classification_tool/last_session.json`
     (ключ `ui.theme`)

3. **Расширение `section_card`** (`ui_widgets/theme.py`)
   - Сигнатура: `section_card(title, children, subtitle="", right=None)`
   - Если `right` — widget/HBox, вставляется в header справа от title

4. **Helper'ы для полей**
   - `field_label(text)` → HTML-widget с `.brt-field-label` class
     (uppercase, muted, small font)
   - `separator()` → HTML-widget с `<div style='border-bottom: …'>` блоком

5. **Расширить badge / chip kinds**
   - Добавить `accent` kind (yellow/amber) в `chip()` и `badge()`
   - Убедиться, что все 5 kind'ов покрыты: `default | accent | success | warning | error`

## Критерии готовности

- [ ] Переключение темы в sidebar работает без перезагрузки страницы
- [ ] Выбор темы сохраняется между сессиями
- [ ] Скриншот текущего UI (Dark Teal) визуально совпадает с
      `ctk_migration_pack/BankReasonTrainer.html` на 95%
- [ ] `section_card(..., right=Button)` корректно рендерит кнопку в header
- [ ] Все 5 kind'ов `chip()` отображаются в test-notebook

---

# Спринт 3 — Очистка legacy-кода (1 день)

**Цель**: удалить desktop-UI и deprecated shims, если решено
окончательно отказаться от GUI.

## Задачи (при условии «полный отказ от desktop»)

1. **Удалить desktop entry-points**
   - `app.py`, `app_train.py`, `app_apply.py`, `app_cluster.py`
   - `app_train_view.py`, `app_apply_view.py`, `app_cluster_view.py`
   - `app_train_view_ctk.py`, `app_apply_view_ctk.py`, `app_cluster_view_ctk.py`
   - `app_deps.py`, `cluster_ui_builder.py`
   - `app_dialogs_ctk.py`, `experiment_history_dialog.py`
   - `bootstrap_run.py`, `bootstrap_ctk.py`, `bootstrap_ctk_demo.py`
   - `run.sh`, `run_app.bat`
   - `ui_widgets_tk.py`, `ui_theme.py`, `ui_theme_ctk.py`
   - `ctk_migration_pack/` (исходный материал уже использован)

2. **Удалить deprecated re-export shims**
   - `core/feature_builder.py`, `core/hw_profile.py`, `core/text_utils.py`
   - `utils/excel_utils.py`, `utils/t5_summarizer.py`, `utils/__init__.py`
   - Импорты из них в `.coveragerc` — убрать omit-записи

3. **Убрать лишние зависимости**
   - `pyproject.toml` → убрать `customtkinter`, `pystray` из основных
     dependencies (они только для desktop)
   - `requirements.txt` — переименовать в `requirements-desktop.txt`
     или удалить, использовать только `pyproject.toml`

4. **Обновить тесты**
   - Удалить `tests/test_app_*`, `tests/test_*_view*`, `tests/test_ui_smoke.py`
   - Удалить Tk-mock из `tests/conftest.py`
   - Убедиться что `pytest -q` проходит без desktop-модулей

5. **Обновить CI**
   - `.github/workflows/quality-gates.yml` — убрать Xvfb-шаг для UI smoke

6. **Обновить документацию**
   - `README.md` §6 (UI/Tray), §8.1 (Desktop), §9 (EXE build) — удалить
   - `CLAUDE.md` — убрать упоминания `app_*.py`, десктопного state
   - `docs/DEPLOY.md` — вычистить desktop-секции

## Задачи (при условии «desktop остаётся fallback»)

- `app_cluster.run_cluster()` → тонкая обёртка над `ClusteringWorkflow.run()`
  с адаптером Tk-progress → service-callback. Ожидаемый результат:
  951 строка → ~200 строк.
- Всё остальное из web-миграции остаётся.

## Критерии готовности

- [ ] `pytest -q` зелёный после удаления
- [ ] `python -m bank_reason_trainer --help` работает (CLI оставляем)
- [ ] `./run_web.sh` запускается на свежем клоне без desktop-deps
- [ ] Размер репо уменьшился минимум на 15k LoC (if full removal)

---

# Спринт 4 — Production-deploy templates для JupyterHub (0.5 дня)

**Цель**: готовые артефакты для админа, который хочет поднять
self-hosted JupyterHub с нашим web-UI.

## Задачи

1. **`jupyterhub_config.py.example`** (новый, корень репо)
   - Minimal single-admin вариант:
     ```python
     c.JupyterHub.admin_users = {"admin"}
     c.Spawner.cmd = ["jupyter-labhub"]
     c.Spawner.args = [
         "--ServerApp.default_url=/voila/render/notebooks/ui.ipynb",
     ]
     c.Spawner.environment = {
         "HF_HOME": "/shared/hf-cache",
         "BRT_LLM_PROVIDER": "offline",
     }
     ```
   - Комментарии: LDAP/OAuth варианты, ресурсные лимиты (RAM/GPU)

2. **`docker-compose.yml.example`** (новый)
   - JupyterHub + PostgreSQL + opt-in named-volume для user-homes
   - Volume mounts: `/shared/hf-cache`, `/data`
   - Env vars для LLM-ключей (с пометкой «use Docker secrets in prod»)

3. **`Dockerfile.jupyterhub`** (новый)
   - Base: `python:3.11-slim-bookworm`
   - `uv sync --frozen --extra ml --extra ui`
   - Pre-download SBERT-модели (opt-in через build-arg)
   - Entry-point: `jupyterhub -f /srv/jupyterhub_config.py`

4. **Обновить `docs/JUPYTERHUB_UI.md`**
   - Добавить §9 «Self-hosted via docker-compose» со ссылками на
     новые файлы
   - Security-нота: пермишены на shared HF-cache (0755), persistent
     per-user volumes для trust-store

## Критерии готовности

- [ ] `docker-compose -f docker-compose.yml.example up` поднимает hub
      с логином `admin/admin`
- [ ] После логина пользователь сразу попадает на web-UI
- [ ] `HF_HOME` работает — SBERT грузится из shared-cache
- [ ] Документ обновлён

---

# Сводная таблица спринтов

| # | Спринт | Срок | Зависимости | Приоритет |
|---|---|---|---|---|
| 1 | Launcher'ы + quickstart | 1 день | — | **P0** |
| 2 | Дизайн-паритет | 1 день | — | **P0** |
| 3 | Очистка legacy | 1 день | Решение по desktop | P1 |
| 4 | JupyterHub deploy templates | 0.5 дня | — | P1 |

## Порядок выполнения

```
Спринт 1  ─┐
           ├── могут идти параллельно
Спринт 2  ─┘
           │
Спринт 3  ─┤  (после решения по desktop)
           │
Спринт 4  ─┘
```

## Критерии успеха всего плана

- [ ] Коммиты в `claude/code-review-assistant-0Fmdx`, атомарные per-спринт
- [ ] Полная regression: `pytest --ignore=tests/test_heavy_modules.py` зелёный
- [ ] Voilà cold-start ≤ 2 секунды
- [ ] Пользователь на чистой машине запускает UI ≤ 5 минут от git clone
      до рабочего dashboard в браузере
