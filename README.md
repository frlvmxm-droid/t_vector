# BankReasonTrainer

Десктопное приложение для:
- обучения модели классификации причин обращений,
- пакетной классификации Excel/CSV,
- кластеризации неразмеченных текстов,
- генерации названий/описаний кластеров (LLM или без LLM).

---

## 1) Возможности

### Обучение (вкладка «Обучение»)
- Обучение `CalibratedClassifierCV` + TF-IDF/SBERT/гибридных признаков.
- Режим «с нуля» и «дообучение» от базовой `.joblib` модели.
- Безопасная загрузка базовой модели через trust-check/loader hooks.
- Пресеты весов секций текста, авто-профили, guardrails.

### Классификация (вкладка «Классификация»)
- Применение модели к Excel/CSV.
- Выгрузка предсказаний, confidence, review-флагов и сводок.
- Поддержка ансамбля моделей.

### Кластеризация (вкладка «Кластеризация»)
- KMeans / HDBSCAN / LDA / BERTopic-like / FASTopic / потоковый режим.
- Качество кластеров (`high/medium/low/single`).
- Иерархическая разбивка и пост-обработка шума.
- Сохранение/загрузка модели кластеризации в **валидационном `.joblib` bundle**.
- Поддержка LLM-нейминга и LLM-feedback рекомендаций.
- Генерация «причин кластера»:
  - через LLM,
  - через rule-based fallback (без LLM).

---

## 2) Безопасность загрузки моделей

### Кластерные модели
- Используется `.joblib` bundle (вместо legacy `.pkl`).
- Проверяется `artifact_type`, `schema_version` и обязательные ключи.
- Перед загрузкой вызывается trust-confirm (или хостовый `_ensure_trusted_path`, если доступен).

### Базовая модель обучения
- При выборе base model используется trust-check.
- Если доступен `_load_model_pkg`, применяется централизованный loader; иначе fallback на `joblib.load`.

> Важно: `.joblib`/pickle — это сериализация Python-объектов. Загружайте только доверенные файлы.

---

## 3) LLM-провайдеры

Для нейминга/описаний/feedback в кластеризации поддержаны:
- `anthropic` (Claude)
- `openai` (ChatGPT)
- `gigachat`
- `qwen`
- `ollama` (локальный, без API-ключа)

Реализован единый клиент вызова (`_llm_complete_text`), чтобы все LLM-функции работали одинаково.

### Как использовать локальную Ollama в кластеризации
1. Убедитесь, что сервер Ollama запущен локально (`http://127.0.0.1:11434`).
2. Проверьте, что модель установлена: `ollama list`.
3. Во вкладке **«Кластеризация»** включите **«Использовать LLM для названий кластеров»**.
4. Выберите провайдера **`ollama`**.
5. В поле **«Модель»** укажите тег из `ollama list` (например, `qwen3:30b`).
6. Поле API-ключа оставьте пустым (для `ollama` ключ не нужен).

Если локальный сервер не отвечает, в логе кластеризации появится сообщение об ошибке сети/таймаута LLM.

---

## 4) Описания причин кластеров

### LLM-режим
Опция: **«Генерировать обобщённое описание причин кластера»**.
Для каждого кластера формируются 1–2 предложения из ключевых слов + примеров строк.

### Без LLM (эвристика)
Опция: **«Формировать описание причин без LLM (эвристика)»**.
Используется rule-based генерация по ключевым словам/паттернам.

Обе опции пишут в столбец `cluster_reason` (или пользовательское имя).

---

## 5) Как уменьшены «пустые/неопределённые» значения

В выгрузке кластеризации:
- Для `cid < 0` подставляются явные значения вместо пустот (`cluster_name`, `cluster_reason`, `cluster_quality`, `llm_feedback`).
- `cluster_keywords` всегда заполняется (fallback на доминирующий кластер файла).
- Доля `cid < 0` ограничивается до 10% на файл (избыточные строки переназначаются в доминирующий кластер, с логом предупреждения).

---

## 6) UI/Tray и поведение окна

- На Windows окно работает в стандартном оконном режиме (видно в taskbar/Alt-Tab).
- Трей-иконка динамическая:
  - `idle` (зелёный) — ожидание,
  - `busy` (жёлтый) — идёт обработка.
- Tooltip трея показывает краткий runtime-статус и ETA.

Иконки окна/сборки подхватываются автоматически (если файлы есть):
- `.ico`: `ui/app_icon.ico`, `ui/icon.ico`, `app_icon.ico`, `icon.ico`
- `.png`: `ui/app_icon.png`, `ui/icon.png`, `app_icon.png`, `icon.png`

---

## 7) Память и устойчивость

После завершения кластеризации (успех/ошибка/отмена):
- вызывается `gc.collect()`,
- при CUDA — `torch.cuda.empty_cache()`.

Это снижает накопление памяти между прогонами (с оговоркой: аллокатор не всегда мгновенно возвращает RAM ОС).

---

## 8) Установка и запуск

### 8.1 Десктоп (Windows / macOS / Linux с X11)

- **Windows:** `run_app.bat`
- **Linux/macOS:** `./run.sh`
- Универсально: `python bootstrap_run.py`

Ручной вариант:
```bash
pip install -r requirements.txt
python bootstrap_run.py
```

Требования: Python 3.9+ и рабочий Tkinter.

---

### 8.2 Linux: headless CLI (без GUI)

Для серверов без X11 — обучение, применение и кластеризация через
командную строку. Tkinter не нужен.

```bash
# 1) Клонирование
git clone <repo-url> t_vector && cd t_vector

# 2) Установка — pinned через uv (рекомендуется, matches CI + Docker)
pip install uv
uv sync --frozen --extra ml          # добавьте --extra ui для web-UI

# 2-alt) Или обычный pip
pip install -r requirements.txt

# 3) Обучение
PYTHONPATH=. uv run python -m bank_reason_trainer train \
    --data train.xlsx --out model.joblib \
    --text-col text --label-col label

# 4) Применение
PYTHONPATH=. uv run python -m bank_reason_trainer apply \
    --model model.joblib --data in.xlsx --out out.xlsx \
    --text-col text

# 5) Кластеризация
PYTHONPATH=. uv run python -m bank_reason_trainer cluster \
    --files a.xlsx b.xlsx --out clusters.csv \
    --text-col text --k-clusters 8
```

Подробнее по CLI, Docker, переменным окружения — см.
[`docs/DEPLOY.md`](docs/DEPLOY.md) и `CLAUDE.md → Headless CLI`.

---

### 8.3 Web-UI на Linux (Voilà, локально)

Браузерный вариант всех трёх вкладок (Обучение / Применение /
Кластеризация) поверх service-слоя. Tkinter не требуется.

```bash
# 1) Установить ML- и UI-extras
uv sync --frozen --extra ml --extra ui
#   или: pip install "ipywidgets>=8.0" "voila>=0.5" scikit-learn joblib pandas openpyxl

# 2) Проверка: импорт приложения
PYTHONPATH=. python -c "from ui_widgets import build_app; print(type(build_app()))"
#   → <class 'ipywidgets.widgets.widget_box.VBox'>

# 3) Запуск Voilà
PYTHONPATH=. voila notebooks/ui.ipynb --port 8866 --no-browser
#   откройте http://localhost:8866/  (на удалённом хосте — проброс порта
#   через `ssh -L 8866:localhost:8866 user@host`)
```

Страница сохраняет значения виджетов (`k_clusters`, `vec_mode`, пороги
и т. д.) в `~/.classification_tool/last_session.json` и восстанавливает
их при следующем открытии вкладки.

Опционально — переменные окружения:
```bash
export HF_HOME=/shared/hf-cache            # общий кэш HuggingFace
export BRT_LLM_PROVIDER=ollama             # LLM для нейминга кластеров
export BRT_LLM_MODEL=qwen3:30b             # тег из `ollama list`
export BRT_LLM_API_KEY=""                  # пусто для ollama
export LLM_SNAPSHOT_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
```

---

### 8.4 JupyterHub (multi-user deployment)

Подробный гайд — [`docs/JUPYTERHUB_UI.md`](docs/JUPYTERHUB_UI.md).
Кратко, шаги администратора:

1. **Установить extras в user-image / per-user env:**
   ```bash
   uv sync --frozen --extra ml --extra ui
   # или: pip install "ipywidgets>=8.0" "voila>=0.5"
   ```

2. **Перевести spawner на дашборд** в `jupyterhub_config.py`:
   ```python
   c.Spawner.cmd = ["jupyter-labhub"]
   c.Spawner.args = [
       "--ServerApp.default_url=/voila/render/notebooks/ui.ipynb",
   ]
   ```
   Пользователи будут логиниться сразу в Voilà-дашборд; полный Lab
   остаётся доступен по `/user/<name>/lab`.

3. **(Опц.) общий HuggingFace-кэш** — чтобы SBERT / T5 не качались
   у каждого пользователя отдельно:
   ```python
   c.Spawner.environment = {"HF_HOME": "/shared/hf-cache"}
   ```

4. **(Опц.) общий том с данными** — смонтируйте volume и скажите
   пользователям абсолютный путь: web-UI принимает его в поле
   *«Путь:»* (обходит 50 MB-лимит browser-upload).

5. **Trust-store для моделей.** `.joblib` проверяются по SHA-256
   через `model_loader.TrustStore` (файл
   `~/.classification_tool/trusted_models.json`). На мульти-юзер
   хабе этот путь должен лежать на персистентном per-user volume.

6. **Smoke-тест на одном юзере:**
   ```bash
   RUN_VOILA_SMOKE=1 PYTHONPATH=. pytest tests/test_voila_smoke.py -v
   ```

Известные ограничения web-UI (полный список —
[`docs/JUPYTERHUB_UI.md#known-limitations`](docs/JUPYTERHUB_UI.md)):
- Plotly/Cleanlab-визуализации доступны только в desktop-версии.
- BERTopic / SetFit / FASTopic — только через CLI (`--allow-skeleton`)
  или desktop.
- GPU-contention на мульти-юзер-хабах регулируется лимитами
  JupyterHub, а не приложением.

---

## 9) Сборка EXE (Windows)

```bat
build_exe.bat
```
или
```bat
pyinstaller bank_reason_trainer.spec --clean
```

`bank_reason_trainer.spec` автоматически:
- подключает `background.png`,
- добавляет иконки в `datas` при наличии,
- выставляет `icon` для EXE, если найден `.ico`.

---

## 10) Тесты

Базовый запуск тестов:

```bash
pytest -q
```

Запуск отдельного теста:

```bash
pytest tests/test_train_service.py -q
```

---

## 11) Ключевые директории/файлы

- `app.py` — основное окно, переменные состояния, tray/window поведение.
- `app_train.py` — обучение модели.
- `app_apply.py` — классификация.
- `app_cluster.py` — кластеризация, LLM-интеграция, экспорт.
- `ml_vectorizers.py` — TF-IDF/SBERT/гибридные векторайзеры.
- `bank_reason_trainer.spec` — конфиг сборки PyInstaller.

Рабочие папки (создаются автоматически):
- `model/`
- `classification/`
- `clustering/`
- `sbert_models/`

---

## 12) Замечания по обратной совместимости

- Legacy `.pkl` для кластерной модели не используется как основной формат.
- Рекомендуемый формат — `.joblib` bundle со схемой.
- При миграции старых пайплайнов проверьте настройки колонок/порогов и LLM-параметры.

---

## 13) Диагностика проблем

1. **LLM не отвечает**: проверьте провайдера, модель, API-ключ (для `ollama` ключ не нужен), сеть/endpoint.
2. **Много шума (`cid=-1`)**: увеличьте качество текстовых полей, настройте алгоритм/параметры, проверяйте dedup/noise фильтры.
3. **Высокая память после прогона**: это частично поведение аллокаторов; перезапуск процесса обычно полностью освобождает RAM.
4. **Проблема иконки**: положите `.ico/.png` в один из поддержанных путей, затем перезапустите приложение/сборку.

---

## 14) Лицензия/дисклеймер

Этот проект работает с пользовательскими данными и сериализованными артефактами моделей.
Используйте доверенные источники файлов и соблюдайте внутренние политики ИБ вашей организации.
