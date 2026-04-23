# Quickstart: Web-UI BankReasonTrainer на ноутбуке

Гайд под одиночный локальный запуск (Windows или Linux/macOS). Для
multi-user JupyterHub — см. [`JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md).

## 1. Требования

- **Python 3.11+** (Python 3.12, 3.13 тоже подходят)
  - Windows: поставь с [python.org/downloads](https://python.org/downloads/),
    при установке отметь **«Add python.exe to PATH»**.
  - Linux: обычно уже есть. Проверить: `python3 --version`.
  - macOS: `brew install python@3.11`.
  - ⚠️ На Windows **не используй** Python-stub из Microsoft Store — он
    не работает с pip.
- **Свободный порт 8888** (или задай другой через `BRT_PORT`, см. ниже).
- **~500 МБ свободного места** на первую установку зависимостей.

Tkinter **не нужен** — web-UI работает в браузере.

## 2. Windows: двойной клик

```bat
git clone <repo-url>
cd t_vector
run_web.bat
```

Что происходит внутри:
1. Детектится Python (`py -3.11`, `py -3`, `python`)
2. При первом запуске — `pip install -e ".[ui]"` (скачает ipywidgets, voila, sklearn, ~300 МБ)
3. Запускается Voilà на `http://127.0.0.1:8888/`
4. Открывается браузер

Терминал оставляй открытым пока работаешь — Ctrl+C останавливает сервер.

## 3. Linux / macOS: скрипт

```bash
git clone <repo-url>
cd t_vector
chmod +x run_web.sh  # один раз
./run_web.sh
```

То же что и на Windows: проверка Python, установка [ui]-зависимостей
при первом запуске, старт Voilà, автооткрытие браузера.

## 4. Переменные окружения (опционально)

| Переменная | По-умолчанию | Что делает |
|---|---|---|
| `BRT_PORT` | `8888` | Порт для Voilà |
| `BRT_HOST` | `127.0.0.1` | Хост привязки. Ставь `0.0.0.0` только если нужно доступ из локальной сети |
| `BRT_NO_OPEN` | `0` | `1` — не открывать браузер автоматически (удобно в SSH) |
| `PYTHON` | авто-детект | Явно указать интерпретатор (только в `run_web.sh`) |
| `HF_HOME` | `~/.cache/huggingface` | Куда кэшировать SBERT / transformers модели |
| `BRT_LLM_PROVIDER` | `offline` | LLM для нейминга кластеров (`anthropic` / `openai` / `ollama` / `offline`) |
| `BRT_LLM_API_KEY` | — | Ключ к LLM-провайдеру |

Пример:
```bash
BRT_PORT=9000 BRT_HOST=0.0.0.0 ./run_web.sh
```

```bat
set BRT_PORT=9000 && set BRT_HOST=0.0.0.0 && run_web.bat
```

## 5. Альтернатива — ручной запуск через uv

Для тех, кто предпочитает `uv` вместо `pip`:

```bash
# Linux / macOS
uv sync --frozen --extra ui
# opt-in heavy ML stack (torch, transformers):
uv sync --frozen --extra ml --extra ui

PYTHONPATH=. uv run voila notebooks/ui.ipynb --port=8888 --no-browser
```

## 6. Troubleshooting

| Проблема | Решение |
|---|---|
| `Port 8888 already in use` | Либо занять другой: `BRT_PORT=8889 ./run_web.sh`. Либо найти процесс: `lsof -i :8888` (Linux) / `netstat -ano \| findstr 8888` (Windows) и остановить. |
| `Python 3.11+ required` | Обновить Python до 3.11+. На Windows — `py -3.11` должен работать после установки с python.org. |
| `pip install failed` | Нет интернета или нужен proxy. Установи вручную: `pip install "ipywidgets>=8.0" "voila>=0.5" "ipykernel>=6.0"`. |
| 500 / `No Jupyter kernel for language 'python' found` | Нет зарегистрированного kernelspec (частая проблема на свежем python.org 3.13 без Anaconda). Launcher регистрирует его автоматически начиная с этого релиза; для ранее установленных окружений — `python -m ipykernel install --user --name python3`. |
| `Notebook ui.ipynb is not trusted` (warning, не блокер) | Безопасно игнорировать: Voilà рендерит виджеты, а не выполняет произвольные cell outputs. При желании: `jupyter trust notebooks/ui.ipynb`. |
| Страница открылась, но висит «Executing 2 of 2» | Это первый запуск — подожди 5-10 секунд. Если больше минуты — перезапусти сервер (Ctrl+C + заново). Cold-start ≤ 2с после всех фиксов. |
| `ModuleNotFoundError: ui_widgets` | Запусти скрипт из корня репо (там где лежит `pyproject.toml`), не из подпапки. |
| Браузер не открылся автоматически | Открой вручную: `http://localhost:8888/`. На Linux без DE используй `BRT_NO_OPEN=1`. |
| Microsoft Store Python stub | `run_web.bat` его отсекает. Поставь реальный Python с python.org. |
| Занят HTTPS-порт 443, хочу скрыть 8888 | Это локальный порт. Для продакшн-доступа через HTTPS — используй reverse proxy (nginx) или JupyterHub (см. [`JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md)). |

## 7. Остановка

Нажми **Ctrl+C** в терминале где запущен скрипт. Voilà корректно
закроет воркеров и kernel.

Вкладка браузера после этого покажет «Connection lost» — просто закрой.

## 8. Сохранение состояния

Web-UI сохраняет значения виджетов (выбранные файлы не сохраняются,
все параметры — да) в
`~/.classification_tool/last_session.json` и восстанавливает их при
следующем запуске. Если файл повреждён, он будет проигнорирован без
ошибки.

## 9. Что дальше

- **Multi-user deployment** на JupyterHub — [`JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md)
- **CLI без UI** (train/apply/cluster) — [`DEPLOY.md`](DEPLOY.md)
- **Описание вкладок** (что где жать) — [`JUPYTERHUB_UI.md`](JUPYTERHUB_UI.md#user-guide)
- **Формат `.joblib`-модели** — [`../CLAUDE.md`](../CLAUDE.md) → «Model Bundle Format»
