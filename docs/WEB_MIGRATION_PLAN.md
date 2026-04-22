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

# Спринт 1 — ✅ DONE

Задачи выполнены, раздел свернут. Артефакты:
- `run_web.bat`, `run_web.sh` (корень репо) — детект Python, dep-check,
  автооткрытие браузера, env-override `BRT_PORT` / `BRT_HOST`.
- `docs/QUICKSTART_WEB_UI.md` — локальный запуск Windows/Linux/macOS
  + troubleshooting.
- `README.md` §8.3 переведён на launcher'ы.

Подробный чек-лист был в версии документа до 2026-04-22 (см.
`git log -- docs/WEB_MIGRATION_PLAN.md`). В сводной таблице внизу —
одна строка со статусом.

---

# Спринт 2 — Дизайн-паритет с ctk_migration_pack (1 день)

**Цель**: три темы + theme-switcher в sidebar + недостающие
design-компоненты, чтобы web-UI визуально совпадал с эталоном из
`ctk_migration_pack/BankReasonTrainer.html`.

**Важно**: `section_card`, `section_header`, `chip`, `chips_row`,
`badge`, `metric_card`, `metric_row`, `inject_css`, `header_chip_row`,
`overlay_card`, `card_layout`, `status_badge` уже существуют в
`ui_widgets/theme.py` и используются во всех панелях. Задачи ниже —
это **расширение существующих функций**, а не создание с нуля.

## Задачи

1. **Палитры в `ui_widgets/theme.py`**
   - Сейчас на ст. 13–28 лежат плоские module-level константы
     (`BG`, `PANEL`, `FG`, `ACCENT`, и т.д.) — они зашивают
     единственную тему Dark Teal в CSS на ст. 31–354.
   - Ввести над ними `PALETTES: dict[str, dict[str, str]]`:
     - `"dark-teal"` — текущие значения, default.
     - `"paper"` — светлая, взять из
       `ctk_migration_pack/ui_theme_ctk.py` (пакет удаляется в
       Спринте 3, палитры копируются сюда).
     - `"amber-crt"` — ретро, из того же файла.
   - Плоские константы заменить на динамическое чтение из
     `PALETTES[_ACTIVE]`; `_CSS` сделать функцией (не f-string на
     module-level), чтобы он регенерировался при смене темы.

2. **`apply_theme(name: str) -> None`** (новая функция в `theme.py`)
   - Перезаписывает module-level активную палитру.
   - Возвращает свежий `ipywidgets.HTML` через `inject_css()` для
     ре-инъекции (вызывающий код заменяет старый CSS-widget).

3. **Theme-switcher в sidebar** (`ui_widgets/notebook_app.py`)
   - Под hw-card — трёх-кнопочная группа `[Teal · Paper · CRT]`
     (Toggle/RadioButtons).
   - `on_click` → `apply_theme(name)` + замена CSS-widget'а в
     sidebar DOM.
   - Persist: через существующий `ui_widgets/session.py:save_session()`
     (atomic-write + `DebouncedSaver` уже готовы), новый ключ
     `ui.theme`. **Не создавать** свой persistence-слой.
   - На старте `build_app()` читать `load_last_session().get("ui", {}).get("theme")`
     и вызывать `apply_theme()` до первого `inject_css()`.

4. **Расширение `section_card`** (`ui_widgets/theme.py:422`)
   - Текущая сигнатура: `section_card(title, children, subtitle="")`.
   - Новая: `section_card(title, children, subtitle="", right=None)`.
   - Если `right` — widget или `HBox`, вставляется в header справа
     от title (через внутренний `HBox([section_header(...), right])`).

5. **Helper'ы для полей** (новые в `theme.py`)
   - `field_label(text: str) -> ipywidgets.HTML` — CSS-класс
     `.brt-field-label` (uppercase, muted, 11px, letter-spacing 1px).
   - `separator() -> ipywidgets.HTML` — тонкий `<div>` с
     `.brt-sep` (1px border-bottom, color `BORDER2`, margin 8px 0).
   - Оба CSS-класса добавить в `_CSS` в `theme.py`.

6. **Расширить `chip()` / `badge()` — kind `accent`**
   - Сейчас `chip()` (`theme.py:376`) принимает
     `{default, ok, warn, err, info}`, `badge()` (`theme.py:433`) —
     `{ok, warn, err, info, default}`.
   - Добавить kind `accent` (amber/yellow) в обе функции.
   - В `_CSS` добавить правила `.brt-chip.accent` (рядом с
     ст. 222–225) и `.brt-badge-accent` (рядом с ст. 328–331).
   - Канонический список после правок:
     `{default, accent, ok, warn, err, info}` — задокументировать в
     docstring обеих функций.

## Критерии готовности

- [ ] `apply_theme("paper")` в ноутбуке мгновенно меняет тему без
      перезагрузки Voilà-страницы.
- [ ] Выбор темы сохраняется в `~/.classification_tool/last_session.json`
      (ключ `ui.theme`) и применяется при следующем `build_app()`.
- [ ] Скриншот текущего UI в Dark Teal визуально совпадает с
      `ctk_migration_pack/BankReasonTrainer.html` на 95%.
- [ ] `section_card(..., right=Button("…"))` корректно рендерит
      кнопку в header справа от title.
- [ ] Все 6 kind'ов `chip()` и `badge()` отображаются в
      test-notebook (`notebooks/ui_smoketest.ipynb` или аналоге).

---

# Спринт 3 — Очистка legacy-кода (1 день)

**Цель**: удалить весь desktop-UI (Tk + CTK), deprecated shims, сборку
PyInstaller, Xvfb-джоб в CI и связанные упоминания в документации.
Развилка из §Решение закрыта: **полный отказ от desktop**.

**Ожидаемый эффект:** минус ~750 KB Tk/CTK-кода, ~15–20k LoC; чистый
`pyproject.toml` без `customtkinter`/`pystray`; `pytest` без Xvfb.

## 3.1 Удалить UI-слой (GUI-логика + виджеты + темы)

```
app.py                              (~122 KB, ~2 399 строк)
app_train.py                        (~100 KB, ~4 877 строк)
app_apply.py                        (~100 KB, ~1 753 строки)
app_cluster.py                      (~250 KB, ~4 362 строки;
                                     run_cluster() на 3410–3510 уходит с файлом)
app_deps.py                         (~31 KB, 614 строк)
app_dialogs_ctk.py                  (~36 KB, 776 строк)
experiment_history_dialog.py        (~8 KB, 188 строк)
cluster_ui_builder.py               (~600 B, 19 строк)
app_train_view.py, app_train_view_ctk.py
app_apply_view.py, app_apply_view_ctk.py
app_cluster_view.py, app_cluster_view_ctk.py
ui_widgets_tk.py                    (~40 KB)
ui_theme.py                         (~16 KB, 457 строк)
ui_theme_ctk.py                     (~10 KB, 292 строки)
bank_reason_trainer_gui.py          (~1 KB, 29 строк — deprecated shim)
background.png                      (1.9 MB)
ctk_migration_pack/                 (после копирования paper/amber-crt
                                     палитр в ui_widgets/theme.py)
```

## 3.2 Удалить desktop-лаунчеры и сборку

```
bootstrap_run.py                    (~20 KB, 456 строк)
bootstrap_ctk.py                    (~8 KB, 191 строка)
bootstrap_ctk_demo.py               (если есть)
run.sh, run_app.bat
bank_reason_trainer.spec            (PyInstaller — CI его не билдит,
                                     сборка только локально)
build_exe.bat
```

Оставить: `download_sbert.bat` (не Tk-зависим, полезен для web).

## 3.3 Deprecated shims в `core/`, `utils/`

- `core/feature_builder.py`, `core/hw_profile.py`, `core/text_utils.py` —
  удалить если это re-exports (проверить перед удалением grep'ом).
- `utils/__init__.py` — **не удалять**. Содержит
  `from utils import excel_utils, t5_summarizer` — это рабочие
  headless-утилиты. Почистить только re-exports, которые тянут
  desktop-модули (если такие есть).
- `.coveragerc` — убрать `omit`-записи для удалённых файлов.

## 3.4 `pyproject.toml` + `uv.lock`

- Убрать из runtime-deps: `customtkinter>=5.2.0`, `pystray`.
- **Запустить `uv lock`** и закоммитить `pyproject.toml` + `uv.lock`
  одним коммитом — CI требует `uv lock --check` (ADR-0008).
- `requirements.txt` — удалить (или оставить одну строку
  `# см. pyproject.toml / uv.lock`, если на него ещё ссылается
  документация до Спринта 3.7).

## 3.5 Тесты + `conftest.py`

- **Удалить**:
  - `tests/test_ui_smoke.py` (CTK под Xvfb).
  - `tests/test_ui_cluster_e2e.py` (CTK e2e).
  - `tests/test_app_mixins_import_smoke.py` (import-smoke для
    `TrainTabMixin`/`ApplyTabMixin`/`ClusterTabMixin`).
  - `tests/test_ui_widgets_tk_export_smoke.py` (CI-guard для
    `ui_widgets_tk.py`).
  - Любые оставшиеся `tests/test_app_*`, `tests/test_*_view*`.
- **Оставить**:
  - `tests/test_ui_widgets_import_smoke.py` (ipywidgets).
  - `tests/test_voila_smoke.py` (Voilà end-to-end).
- **`tests/conftest.py`** (ст. 23–41 по текущему файлу) — снять Tk/CTK
  моки: удалить записи `tkinter`, `tkinter.ttk`,
  `tkinter.filedialog`, `tkinter.messagebox`, `ui_theme`,
  `ui_widgets`, `ui_widgets_tk`, `customtkinter` из `_make_ui_mock`.
  После удаления Tk-модулей они больше не импортируются — моки
  не нужны.
- Убедиться: `PYTHONPATH=. pytest -q` зелёный без `xvfb-run`.

## 3.6 CI

- `.github/workflows/quality-gates.yml`:
  - Удалить шаг «UI smoke under Xvfb» и `sudo apt-get install -y
    xvfb python3-tk`.
  - Pytest-матрица (3.11–3.13) остаётся как есть.
- Если `tests/test_voila_smoke.py` сейчас opt-in через
  `RUN_VOILA_SMOKE=1` — включить по-дефолту в CI (проставить env в
  job), чтобы web-UI smoke шёл вместо Tk-UI smoke.

## 3.7 Документация

- **`README.md`** — удалить:
  - §6 «UI / Tray»,
  - §8.1 «Desktop quick start» (`run_app.bat` / `run.sh`),
  - §9 «Building a distributable EXE»,
  - §11 «Keyboard shortcuts» (все шорткаты были Tk).
  - §8.3 («Web-UI quick start») оставить как единственный путь
    запуска. Поднять его на верхний уровень `§Quick Start`.
- **`CLAUDE.md`** — переписать:
  - «Module Map» → таблицу «Entry Points» убрать целиком
    (`app.py`, `app_train.py`, `app_apply.py`, `app_cluster.py`,
    `bootstrap_run.py`).
  - Раздел «Adding a New ML Config Flag» — убрать пункт 3 про
    `app_train.py` (ст. ≈3754 и ≈3422); оставить pyproject/service-layer.
  - Раздел «Building a Distributable» — удалить.
  - Раздел «Keyboard Shortcuts» — удалить.
  - «Quick Start» — заменить `python bootstrap_run.py` на
    `./run_web.sh`.
- **`docs/DEPLOY.md`** — вычистить десктопные секции (PyInstaller,
  tray-icon, Tk-headless recipes).

## Критерии готовности Спринта 3

- [ ] `grep -rn "^import tkinter\|^import customtkinter\|^from tkinter\|^from customtkinter" --include="*.py" .`
      → пусто (исключая тесты, которые тоже удалены).
- [ ] `pytest -q` зелёный без `xvfb-run` и `python3-tk`.
- [ ] `./run_web.sh` на свежем клоне поднимает UI (без
      `customtkinter`/`pystray` в окружении).
- [ ] `uv lock --check` проходит в CI.
- [ ] `git diff --stat main` показывает минус ≥ 15k LoC.
- [ ] README/CLAUDE.md/docs/DEPLOY.md не содержат слов
      «tkinter», «customtkinter», «PyInstaller», «tray» вне
      исторического контекста.

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
   - Base: `python:3.11-slim-bookworm` — **без X11 / Tk / GUI-пакетов**
     (Спринт 3 убирает `customtkinter`, `pystray`, `python3-tk` как
     class, так что `apt install xvfb python3-tk` здесь не нужен).
   - `uv sync --frozen --extra ml --extra ui` (без `--extra desktop`
     или эквивалента — desktop-extras нет).
   - Pre-download SBERT-моделей (opt-in через build-arg).
   - Entry-point: `jupyterhub -f /srv/jupyterhub_config.py`.

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
| 1 | Launcher'ы + quickstart | — | — | ✅ DONE 2026-04-22 |
| 2 | Дизайн-паритет | 1 день | — | **P0** |
| 3 | Очистка legacy (полный отказ от desktop) | 1 день | §Решение (закрыто) | **P0** |
| 4 | JupyterHub deploy templates | 0.5 дня | Спринт 3 (Dockerfile использует очищенный `pyproject.toml`) | P1 |

## Порядок выполнения

```
Спринт 1 ✅ DONE
          │
          ├── Спринт 2 ─┐
          │              ├── могут идти параллельно
          └── Спринт 3 ─┘
                         │
                         └── Спринт 4 (Dockerfile наследует чистые deps)
```

## Критерии успеха всего плана

- [ ] Коммиты в `claude/add-pyinstaller-workflow-TKjgJ`, атомарные per-спринт
- [ ] Полная regression: `pytest --ignore=tests/test_heavy_modules.py` зелёный
- [ ] `grep -rn "import tkinter\|customtkinter" --include="*.py" .` → пусто
- [ ] Voilà cold-start ≤ 2 секунды
- [ ] Пользователь на чистой машине запускает UI ≤ 5 минут от git clone
      до рабочего dashboard в браузере
