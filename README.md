# BankReasonTrainer

Web-приложение (Voilà + ipywidgets) для:
- обучения модели классификации причин обращений,
- пакетной классификации Excel/CSV,
- кластеризации неразмеченных текстов,
- генерации названий/описаний кластеров (LLM или без LLM).

Запуск — `./run_web.sh` (Linux/macOS) или `run_web.bat` (Windows).
Headless CLI — `python -m bank_reason_trainer {train,apply,cluster}`.

---

## 1) Возможности

### Обучение (панель «Обучение»)
- Обучение `CalibratedClassifierCV` + TF-IDF/SBERT/гибридных признаков.
- Режим «с нуля» и «дообучение» от базовой `.joblib` модели.
- Безопасная загрузка базовой модели через trust-check/loader hooks.
- Пресеты весов секций текста, авто-профили, guardrails.

### Классификация (панель «Классификация»)
- Применение модели к Excel/CSV.
- Выгрузка предсказаний, confidence, review-флагов и сводок.
- Поддержка ансамбля моделей.

### Кластеризация (панель «Кластеризация»)
- KMeans / HDBSCAN / LDA / Agglomerative / SBERT / Combo / Ensemble.
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
- Перед загрузкой вызывается trust-confirm (`model_loader.TrustStore`).

### Базовая модель обучения
- При выборе base model используется trust-check.
- Применяется централизованный loader `model_loader.load_model_artifact`.

> Важно: `.joblib`/pickle — это сериализация Python-объектов. Загружайте только доверенные файлы.

---

## 3) LLM-провайдеры

Для нейминга/описаний/feedback в кластеризации поддержаны:
- `anthropic` (Claude)
- `openai` (ChatGPT)
- `gigachat`
- `qwen`
- `ollama` (локальный, без API-ключа)

Реализован единый клиент вызова (`llm_client.py`), чтобы все LLM-функции работали одинаково.

### Как использовать локальную Ollama в кластеризации
1. Убедитесь, что сервер Ollama запущен локально (`http://127.0.0.1:11434`).
2. Проверьте, что модель установлена: `ollama list`.
3. В панели **«Кластеризация»** включите **«Использовать LLM для названий кластеров»**.
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

## 6) Память и устойчивость

После завершения кластеризации (успех/ошибка/отмена):
- вызывается `gc.collect()`,
- при CUDA — `torch.cuda.empty_cache()`.

Это снижает накопление памяти между прогонами (с оговоркой: аллокатор не всегда мгновенно возвращает RAM ОС).

---

## 7) Установка и запуск

### 7.1 Web-UI на localhost (Windows / Linux / macOS) — основной путь

Браузерный вариант всех трёх панелей (Обучение / Применение /
Кластеризация) поверх service-слоя.

**Самый быстрый способ** — launcher-скрипты из корня репо:

```bash
# Linux / macOS
./run_web.sh

# Windows (двойной клик или из cmd):
run_web.bat
```

Что делает скрипт:
1. Находит Python 3.11+ (детектит `py -3.11` на Windows, `python3.11` / `python3` на Linux).
2. При первом запуске ставит web-UI deps: `pip install -e ".[ui]"` (ipywidgets + voila + базовые ML-либы).
3. Проверяет что порт свободен.
4. Запускает Voilà на `http://127.0.0.1:8866/` и открывает браузер.

Ctrl+C в терминале корректно останавливает сервер.

Подробный гайд (env-переменные, troubleshooting, ручной запуск через
`uv`) — [`docs/QUICKSTART_WEB_UI.md`](docs/QUICKSTART_WEB_UI.md).

**Ручной запуск** (для разработки, CI, пользовательских pipeline):

```bash
# Установить UI-extra (ipywidgets + voila) + базовые ML
uv sync --frozen --extra ui
#   или: pip install -e ".[ui]"

# Запуск Voilà
PYTHONPATH=. voila notebooks/ui.ipynb --port 8866 --no-browser
```

Страница сохраняет значения виджетов (`k_clusters`, `vec_mode`, пороги
и т. д.) в `~/.classification_tool/last_session.json` и восстанавливает
их при следующем открытии.

Опционально — переменные окружения:
```bash
export HF_HOME=/shared/hf-cache            # общий кэш HuggingFace
export BRT_LLM_PROVIDER=ollama             # LLM для нейминга кластеров
export BRT_LLM_MODEL=qwen3:30b             # тег из `ollama list`
export BRT_LLM_API_KEY=""                  # пусто для ollama
export LLM_SNAPSHOT_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
export BRT_PORT=9000                       # свой порт
```

### 7.2 Linux: headless CLI (без GUI)

Для серверов и batch-пайплайнов — обучение, применение и кластеризация
через командную строку.

```bash
# 1) Клонирование
git clone <repo-url> t_vector && cd t_vector

# 2) Установка — pinned через uv (matches CI + Docker)
pip install uv
uv sync --frozen --extra ml          # добавьте --extra ui для Voilà

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

### 7.3 JupyterHub (multi-user deployment)

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
- BERTopic / SetFit / FASTopic — только через CLI (`--allow-skeleton`).
- GPU-contention на мульти-юзер-хабах регулируется лимитами
  JupyterHub, а не приложением.

---

## 8) Тесты

Базовый запуск тестов:

```bash
PYTHONPATH=. pytest -q
```

Запуск отдельного теста:

```bash
PYTHONPATH=. pytest tests/test_e2e_train_predict.py -v
```

Voilà smoke-тест (поднимает реальный сервер, медленнее):

```bash
RUN_VOILA_SMOKE=1 PYTHONPATH=. pytest tests/test_voila_smoke.py -v
```

---

## 9) Ключевые директории/файлы

- `ui_widgets/` — web-UI (ipywidgets): `notebook_app.py`, `train_panel.py`,
  `apply_panel.py`, `cluster_panel.py`, `session.py`, `theme.py`, dialogs/.
- `notebooks/ui.ipynb` — entry-point, который рендерит Voilà.
- `bank_reason_trainer/cli.py` — headless CLI (`train` / `apply` / `cluster`).
- Service-слой (Tk-free, покрыт тестами):
  - `app_train_service.py` — `TrainingWorkflow`.
  - `apply_prediction_service.py` — `predict_with_thresholds`.
  - `cluster_workflow_service.py` — `ClusteringWorkflow.run`.
  - `app_cluster_pipeline.py` — pure-pipeline функции.
- `ml_vectorizers.py` — TF-IDF/SBERT/гибридные векторайзеры.
- `run_web.sh` / `run_web.bat` — launcher-скрипты.

Рабочие папки (создаются автоматически):
- `model/`
- `classification/`
- `clustering/`
- `sbert_models/`

---

## 10) Замечания по обратной совместимости

- Legacy `.pkl` для кластерной модели не используется как основной формат.
- Рекомендуемый формат — `.joblib` bundle со схемой.
- При миграции старых пайплайнов проверьте настройки колонок/порогов и LLM-параметры.

---

## 11) Диагностика проблем

1. **LLM не отвечает**: проверьте провайдера, модель, API-ключ (для `ollama` ключ не нужен), сеть/endpoint.
2. **Много шума (`cid=-1`)**: увеличьте качество текстовых полей, настройте алгоритм/параметры, проверяйте dedup/noise фильтры.
3. **Высокая память после прогона**: это частично поведение аллокаторов; перезапуск процесса обычно полностью освобождает RAM.
4. **Voilà не стартует**: проверьте, что установлены `voila` и `ipywidgets` (`pip list | grep -E "voila|ipywidgets"`), порт 8866 свободен, Python ≥3.11.

---

## 12) Лицензия/дисклеймер

Этот проект работает с пользовательскими данными и сериализованными артефактами моделей.
Используйте доверенные источники файлов и соблюдайте внутренние политики ИБ вашей организации.
