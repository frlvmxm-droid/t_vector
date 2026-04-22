# План миграции BankReasonTrainer на Web-UI

> Источник: синтез трёх аудитов (launch-paths, monolith decomposition,
> ctk_migration_pack design gap), выполненных 2026-04-22 на ветке
> `claude/code-review-assistant-0Fmdx`.

## Контекст

Исторически проект — desktop-Tkinter-приложение (`app.py`,
`app_train.py`, `app_apply.py`, `app_cluster.py`). В ходе миграции
UI переведён на браузерный вариант (Voilà + ipywidgets), исходный
визуальный дизайн из `ctk_migration_pack/BankReasonTrainer.html` и
палитры `ui_theme_ctk.py:PALETTES` перенесены в
`ui_widgets/theme.py`. После этого сам пакет `ctk_migration_pack/`
удалён (см. git history до коммита Sprint 2.4).

Сервисный слой (`app_train_service`, `apply_prediction_service`,
`cluster_workflow_service`, `app_cluster_pipeline`) всегда был
Tk-free и покрыт тестами. Web-UI (`ui_widgets/*`) построен поверх
него, имеет session save/restore и работает в Voilà cold-start за
~1.4с.

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
| **Спринт 2 — темы + theme-switcher** | ✅ complete (PALETTES dark-teal/paper/amber-crt, `apply_theme`/`rebuild_css`, accent kind, `field_label`/`separator`, `section_card(right=)`, sidebar switcher + persist `ui.theme`) |
| **Спринт 3 — удаление desktop-UI** | ✅ complete (минус 19 файлов Tk/CTK, 6 лаунчеров/сборка, 9 UI-тестов, `customtkinter`/`pystray`/`Pillow` из deps, Xvfb-job из CI, переписана документация; кумулятивно −20 986 / +1 453 LoC vs pre-Sprint-3) |

## Известные блокеры / gap'ы

*Все перечисленные ниже блокеры закрыты в ходе Спринтов 1–3. Оставлено
как исторический контекст для git blame.*

### Блокеры запуска — ✅ закрыты (Спринт 1)

### Дизайн-дыры vs ctk_migration_pack — ✅ закрыты (Спринт 2)

### Архитектурные долги — ✅ закрыты (Спринт 3)
- `app_cluster.run_cluster()` и весь `app_cluster.py` удалены; бизнес-
  логика живёт в `cluster_workflow_service.ClusteringWorkflow.run()`.
- Deprecated shims в `core/` удалены целиком.
- `customtkinter`, `pystray`, `Pillow` удалены из `pyproject.toml`;
  `uv.lock` регенерирован.

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

## Критерии готовности Спринта 3 — ✅ выполнены

- [x] `grep -rn "^import tkinter\|^import customtkinter\|^from tkinter\|^from customtkinter" --include="*.py" .`
      → пусто.
- [x] `pytest -q` зелёный без `xvfb-run` и `python3-tk`
      (job `web-smoke` в `quality-gates.yml` поднимает Voilà).
- [x] `./run_web.sh` на свежем клоне поднимает UI (без
      `customtkinter`/`pystray` в окружении).
- [x] `uv lock --check` проходит в CI (shag `supply-chain`).
- [x] `git diff --stat 99f7985 HEAD` показывает −20 986 / +1 453 LoC
      после всех коммитов Спринта 3 (3.1–3.6).
- [x] README/CLAUDE.md/docs/DEPLOY.md не содержат слов
      «tkinter», «customtkinter», «PyInstaller», «tray» вне
      исторического контекста.

---

# Спринт 4 — ✅ DONE

**Результат**: добавлены три reference-артефакта для self-hosted
JupyterHub, секция в `docs/JUPYTERHUB_UI.md` со ссылками и
инструкциями по hardening для production.

## Что сделано

1. **`Dockerfile.jupyterhub`** — база `jupyterhub/jupyterhub:4.1.5`,
   `uv sync --frozen --extra ml --extra ui`, hub-deploy extras
   (`dummyauthenticator`, `psycopg2-binary`, `jupyterlab`, `notebook`),
   optional `PREDOWNLOAD_SBERT=1` build-arg, demo users
   `admin`/`demo`, auto-gen cookie_secret при первом старте.
2. **`jupyterhub_config.py.example`** — `DummyAuthenticator`,
   `LocalProcessSpawner` с `jupyter-labhub --default_url=/voila/render/
   notebooks/ui.ipynb`, spawner env (`PYTHONPATH`, `HF_HOME`,
   `BRT_LLM_PROVIDER`), hub_connect_ip из env, db_url с fallback
   на sqlite, fail-closed на пустом `CONFIGPROXY_AUTH_TOKEN`.
3. **`docker-compose.yml.example`** — сервисы `jupyterhub` +
   `postgres:16-alpine` (healthcheck), volumes `pg-data` / `hub-state` /
   `hf-cache` / `user-home`; env-vars `:?` fail-closed на пустые
   `POSTGRES_PASSWORD` и `CONFIGPROXY_AUTH_TOKEN`.
4. **`docs/JUPYTERHUB_UI.md` — новая секция** «Self-hosted via
   docker-compose»: ссылки на артефакты, quick-start, production-
   hardening табличка (Authenticator/Spawner/TLS/Secrets/Postgres
   digest/HF cache/LLM), таблица volumes с рекомендациями по
   бэкапам.

## Критерии готовности

- [x] `docker-compose -f docker-compose.yml.example up` поднимает hub
      с admin login (пароль задаётся в jupyterhub_config.py —
      `HUB_DUMMY_PASSWORD`, default `change-me`).
- [x] После логина пользователь сразу попадает на web-UI
      (`/voila/render/notebooks/ui.ipynb`).
- [x] `HF_HOME=/shared/hf-cache` пробрасывается в spawner env —
      SBERT грузится из shared-cache.
- [x] Документ обновлён (новая секция «Self-hosted via docker-compose»
      со ссылками на все три файла).

---

# Сводная таблица спринтов

| # | Спринт | Срок | Зависимости | Статус |
|---|---|---|---|---|
| 1 | Launcher'ы + quickstart | — | — | ✅ DONE 2026-04-22 |
| 2 | Дизайн-паритет (PALETTES + theme-switcher) | — | — | ✅ DONE 2026-04-22 |
| 3 | Очистка legacy (полный отказ от desktop) | — | §Решение (закрыто) | ✅ DONE 2026-04-22 |
| 4 | JupyterHub deploy templates | — | Спринт 3 (Dockerfile наследует очищенный `pyproject.toml`) | ✅ DONE 2026-04-22 |

## Порядок выполнения

```
Спринт 1 ✅ DONE   (launcher'ы + quickstart)
Спринт 2 ✅ DONE   (ui_widgets/theme.py + notebook_app.py theme-switcher)
Спринт 3 ✅ DONE   (−20 986 LoC; customtkinter/pystray/Pillow/Xvfb out)
Спринт 4 ✅ DONE   (Dockerfile.jupyterhub + jupyterhub_config.py.example +
                    docker-compose.yml.example)
```

## Критерии успеха всего плана

- [x] Коммиты в `claude/add-pyinstaller-workflow-TKjgJ`, атомарные per-спринт
- [x] `grep -rn "import tkinter\|customtkinter" --include="*.py" .` → пусто
- [x] Voilà cold-start ≤ 2 секунды
- [x] Пользователь на чистой машине запускает UI ≤ 5 минут от `git clone`
      до рабочего dashboard в браузере (`./run_web.sh`)
- [x] JupyterHub deploy-templates готовы (Спринт 4).
- [ ] Полная regression: `pytest --ignore=tests/test_heavy_modules.py` зелёный
      *(проверяется в CI `quality-gates.yml` → job `tests` по 3.11/3.12/3.13 и
      `web-smoke` c Voilà)*
