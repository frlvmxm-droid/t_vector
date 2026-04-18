# UI Implementation Guide — Hearsy / BankReasonTrainer

> **Источник истины:** HTML-прототип в этом проекте (`BankReasonTrainer.html` + `styles.css` + `components.jsx` + `tab_*.jsx` + `overlays.jsx`).
> **Цель этого файла:** дать пошаговую инструкцию Claude Code, как привести CTK-пакет в репе `frlvmxm-droid/classification-tool@claude/russian-text-analysis-ZKVRy/ctk_migration_pack` к текущему рендеру прототипа.
>
> Документ читать линейно — разделы идут от «срочно поправить» к «добавить новое».

---

## 0. Состояние на текущий момент

В репе уже есть:

| Файл | Статус |
|---|---|
| `ui_theme_ctk.py` | ✅ палитры dark-teal / paper / amber-crt совпадают с `styles.css`. Можно не трогать. |
| `app_train_view_ctk.py` | ⚠️ покрывает ~40% текущего Train-таба. Нет per-column mapping, Auto-profile, device-selector, SetFit, calib_method, section-weights, Advanced-секций. |
| `app_apply_view_ctk.py` | ⚠️ покрывает ~50% Apply-таба. Нет per-class thresholds, ensemble, LLM-rerank, Other-label, ambiguity. |
| `app_cluster_view_ctk.py` | ⚠️ покрывает ~55% Cluster-таба. Нет детальных UMAP / HDBSCAN / PCA knobs, T5, merge, streaming, BERTopic/LDA настроек. |
| `bootstrap_ctk_demo.py` | ⚠️ bran = «BR / BankReason / TRAINER · RU». Нет оверлеев (История / Артефакты / Настройки). |

Расхождения ниже сгруппированы по файлам.

---

## 1. Брендинг и базовое окно

### 1.1. Переименование BankReason → Hearsy

В `bootstrap_ctk_demo.py`:

```python
# было
self.title("BankReasonTrainer · CustomTkinter demo")
ctk.CTkLabel(brand, text="BR", ...)
ctk.CTkLabel(text, text="BankReason", ...)
ctk.CTkLabel(text, text="TRAINER · RU", ...)

# стало
self.title("Hearsy · Classifier")
ctk.CTkLabel(brand, text="Hs", ...)
ctk.CTkLabel(text, text="Hearsy", ...)
ctk.CTkLabel(text, text="CLASSIFIER · RU", ...)
```

**Если переименовываем продукт целиком** (не только в UI): также меняем `BankReasonTrainerApp` → `HearsyApp`, путь `~/.bankreason/…` → `~/.hearsy/…`, и пр. Решение за владельцем репы — **по умолчанию в этом гайде трогаем только UI-строки**.

### 1.2. TitleBar с macOS-точками (опционально)

В прототипе есть строка с тремя точками (red / yellow / green) над sidebar.
На Tk/CTk это бесполезно (окном управляет ОС), **не переносим**. В инструкции — пропустить.

### 1.3. Sidebar — расположение элементов

Актуальный порядок элементов sidebar (сверху вниз):

```
┌─ Бренд (Hs + Hearsy + Classifier · ru)
│
├─ ── разделитель ──
│
├─ «WORKFLOW»
│   ├─ Обучение
│   ├─ Классификация (+ badge «8.4k»)
│   └─ Кластеризация (+ badge «8»)
│
├─ «КОНТЕКСТ»
│   ├─ История экспериментов    → open_history_dialog()
│   ├─ Артефакты моделей        → open_artifacts_dialog()
│   └─ Настройки                → open_settings_dialog()
│
├─ ── spacer ──
│
├─ Hardware-карточка (CPU / RAM / GPU / torch)
│
└─ Футер: «v3.4.1 · build 1248» + иконка help
```

**Тема** (Teal / Paper / CRT) перенесена в отдельный **Tweaks-панель**, который открывается кнопкой в main-header. В CTk-версии можно оставить `CTkSegmentedButton` в sidebar (как сейчас) ИЛИ сделать отдельное окно `open_tweaks_dialog()`. **Рекомендация:** оставить в sidebar, но дать ему label «ТЕМА» и placement — снизу (как сейчас в `bootstrap_ctk_demo.py`). Это удобнее для Tk.

### 1.4. Badges у пунктов меню

В CTk badge = маленький `CTkLabel` справа от текста. Шаблон:

```python
def _nav_item(parent, label, badge=None, active=False, on_click=None):
    row = ctk.CTkFrame(parent, fg_color=(COLORS["select"] if active else "transparent"),
                       corner_radius=6, height=32)
    row.pack(fill="x", padx=10, pady=1)
    row.pack_propagate(False)
    ctk.CTkLabel(row, text=label, font=font_base(), anchor="w",
                 text_color=(COLORS["accent2"] if active else COLORS["muted"])).pack(
        side="left", padx=10, fill="x", expand=True)
    if badge:
        ctk.CTkLabel(row, text=badge, font=font_xs(),
                     fg_color=COLORS["panel2"], text_color=COLORS["muted"],
                     corner_radius=999, padx=6, pady=1).pack(side="right", padx=8)
    row.bind("<Button-1>", lambda e: on_click and on_click())
    return row
```

### 1.5. Hardware-карточка

Статический блок внизу sidebar:

```python
def _build_hardware_card(parent):
    card = ctk.CTkFrame(parent, fg_color=COLORS["panel2"],
                        corner_radius=8, border_width=1, border_color=COLORS["border2"])
    card.pack(fill="x", padx=10, pady=(10, 6))
    rows = [
        ("CPU",   "12 cores · 38%"),
        ("RAM",   "14.2 / 32 ГБ"),
        ("GPU",   "RTX 4070 · 6.1 ГБ"),
        ("torch", "2.4.1+cu121"),
    ]
    for k, v in rows:
        r = ctk.CTkFrame(card, fg_color="transparent")
        r.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(r, text=k, font=font_xs(), text_color=COLORS["muted2"]).pack(side="left")
        ctk.CTkLabel(r, text=v, font=font_mono(), text_color=COLORS["fg"]).pack(side="right")
```

Опционально можно обновлять значения раз в 2 секунды через `psutil` — см. §9.3.

---

## 2. Диалоги-оверлеи (модальные окна)

В HTML-прототипе есть **три полноэкранных оверлея**, которые открываются из sidebar:
- **История экспериментов** (`HistoryOverlay`)
- **Артефакты моделей** (`ArtifactsOverlay`)
- **Настройки** (`SettingsOverlay` с двумя табами: Зависимости / LLM keys)

В CTk это реализуется через `CTkToplevel` с `grab_set()` и затемнённой подложкой.

### 2.1. Базовый класс Dialog

Создать **`app_dialogs_ctk.py`**:

```python
import customtkinter as ctk
from ui_theme_ctk import COLORS, font_md_bold, font_sm, font_label

class ModalDialog(ctk.CTkToplevel):
    def __init__(self, parent, title: str, subtitle: str = "",
                 width: int = 900, height: int = 620):
        super().__init__(parent)
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.configure(fg_color=COLORS["panel"])
        self.transient(parent)
        self.grab_set()

        # header
        header = ctk.CTkFrame(self, fg_color="transparent", height=60)
        header.pack(fill="x", padx=22, pady=(18, 0))
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left")
        ctk.CTkLabel(left, text=title, font=font_md_bold(),
                     text_color=COLORS["fg"]).pack(anchor="w")
        if subtitle:
            ctk.CTkLabel(left, text=subtitle, font=font_sm(),
                         text_color=COLORS["muted"]).pack(anchor="w", pady=(2, 0))
        ctk.CTkButton(header, text="✕", width=32, height=32,
                      fg_color="transparent", text_color=COLORS["muted"],
                      hover_color=COLORS["hover"],
                      command=self.destroy).pack(side="right")

        ctk.CTkFrame(self, height=1, fg_color=COLORS["border2"]).pack(
            fill="x", padx=22, pady=(10, 0))

        # body (контейнер для контента)
        self.body = ctk.CTkScrollableFrame(self, fg_color=COLORS["panel"])
        self.body.pack(fill="both", expand=True, padx=10, pady=10)
```

### 2.2. История экспериментов

```python
class HistoryDialog(ModalDialog):
    def __init__(self, parent):
        super().__init__(parent, title="История экспериментов",
                         subtitle="Все запуски обучения, классификации и кластеризации · "
                                  "хранится локально в experiment_log.jsonl",
                         width=900, height=600)
        headers = [("ID", 80), ("Дата", 140), ("Тип", 100),
                   ("Запуск", 0), ("Метрика", 0), ("Длит.", 80), ("Статус", 90)]
        self._render_table(headers, self._load_runs())

    def _load_runs(self):
        # читать из experiment_log.jsonl в реальном приложении
        # структура: {id, date, kind:'train|apply|cluster', name, metric, dur, status}
        return [...]
```

Типы в `kind`:
- `train` → badge «Обучение» (accent)
- `apply` → badge «Применение» (default)
- `cluster` → badge «Кластер» (default)

Статус-бейдж: `success=✓` (green), `warning=!` (yellow), `error=✕` (red).

### 2.3. Артефакты моделей

Группы (поле `kind`) и их локализация:

| kind         | label        |
|---           |---           |
| `model`      | Модель       |
| `vectorizer` | Vectorizer   |
| `embeddings` | Embeddings   |
| `cluster`    | Кластеры     |
| `calib`      | Калибровка   |
| `llm`        | LLM-кэш      |
| `thresholds` | Thresholds   |

Фильтр — как `CTkSegmentedButton` или набор toggle-кнопок: `Все / Модель / Vectorizer / ...`.
Пометка `current` — маленький бейдж после имени файла (background `COLORS["select"]`, text `COLORS["accent2"]`).

Внизу — info-блок:
> «Артефакты отмеченные `current` используются текущей активной моделью. Удаление других не повлияет на классификацию.»

Колонки таблицы: `Тип | Файл | Размер | Изменён | Метаданные | Действия (Download / Delete)`.

Источник данных — сканирование папки `./artifacts/` в реальном приложении. Mock-данные для dev-режима:

```python
SAMPLE_ARTIFACTS = [
    {"kind": "model",     "name": "baseline_v3_854.joblib",  "size": "184 МБ",
     "date": "2025-04-17 14:22", "tag": "current", "meta": "F1=0.854"},
    {"kind": "model",     "name": "sbert-only_812.joblib",   "size": "172 МБ",
     "date": "2025-04-16 14:30", "tag": "",        "meta": "F1=0.812"},
    # ... см. overlays.jsx → SAMPLE_ARTIFACTS для полного списка
]
```

### 2.4. Настройки (Зависимости + LLM keys)

**Настройки** — это CTkToplevel с `CTkSegmentedButton` вверху для переключения табов.

#### Таб «Зависимости»

**Цель:** показать состояние python-пакетов, необходимых для `Hearsy`, и дать кнопку «Установить».

Структура:

1. **Summary strip** (4 плитки):
   - Установлено (зелёная цифра)
   - Устарело (жёлтая)
   - Отсутствует (красная)
   - Кнопка «Установить всё ({N})» (primary)

2. **Фильтр:** chips `Все / Не установлены / Устаревшие / Готовые`.

3. **Таблицы по группам:**
   - `Ядро`: numpy, pandas, scikit-learn, scipy, joblib
   - `Embeddings`: torch, transformers, sentence-transformers, setfit, accelerate
   - `Кластеризация`: umap-learn, hdbscan, bertopic, gensim
   - `Русский NLP`: pymorphy3, razdel, natasha
   - `Качество`: optuna, cleanlab, imbalanced-learn
   - `LLM`: openai, anthropic, tiktoken
   - `UI / IO`: customtkinter, matplotlib, openpyxl, pyarrow

   Колонки: `Пакет | Требуется | Установлено | Статус | Действие`.

4. **Блок pip-команды** внизу:

   ```
   $ pip install -U \
       accelerate  \
       bertopic    \
       natasha     \
       cleanlab
   ```

**Как проверять установку пакетов в runtime:**

```python
import importlib.metadata as md
from packaging.version import Version
from packaging.specifiers import SpecifierSet

REQUIREMENTS = [
    # (package, required_spec, group)
    ("numpy",                   ">=1.24",  "Ядро"),
    ("pandas",                  ">=2.0",   "Ядро"),
    ("scikit-learn",            ">=1.3",   "Ядро"),
    ("scipy",                   ">=1.10",  "Ядро"),
    ("joblib",                  ">=1.3",   "Ядро"),
    ("torch",                   ">=2.1",   "Embeddings"),
    ("transformers",            ">=4.40",  "Embeddings"),
    ("sentence-transformers",   ">=3.0",   "Embeddings"),
    ("setfit",                  ">=1.0",   "Embeddings"),
    ("accelerate",              ">=0.26",  "Embeddings"),
    ("umap-learn",              ">=0.5.5", "Кластеризация"),
    ("hdbscan",                 ">=0.8.33","Кластеризация"),
    ("bertopic",                ">=0.16",  "Кластеризация"),
    ("gensim",                  ">=4.3",   "Кластеризация"),
    ("pymorphy3",               ">=2.0",   "Русский NLP"),
    ("razdel",                  ">=0.5",   "Русский NLP"),
    ("natasha",                 ">=1.6",   "Русский NLP"),
    ("optuna",                  ">=3.4",   "Качество"),
    ("cleanlab",                ">=2.6",   "Качество"),
    ("imbalanced-learn",        ">=0.12",  "Качество"),
    ("openai",                  ">=1.30",  "LLM"),
    ("anthropic",               ">=0.28",  "LLM"),
    ("tiktoken",                ">=0.7",   "LLM"),
    ("customtkinter",           ">=5.2",   "UI / IO"),
    ("matplotlib",              ">=3.7",   "UI / IO"),
    ("openpyxl",                ">=3.1",   "UI / IO"),
    ("pyarrow",                 ">=15",    "UI / IO"),
]

def check_package(name: str, spec: str) -> tuple[str, str | None]:
    """Возвращает (status, installed_version). status ∈ {'ok','outdated','missing'}."""
    try:
        v = md.version(name)
    except md.PackageNotFoundError:
        return ("missing", None)
    if Version(v) in SpecifierSet(spec):
        return ("ok", v)
    return ("outdated", v)
```

**Установка пакета (кнопка «Установить» / «Обновить»):**

```python
import subprocess, sys, threading

def install_package(name: str, on_line, on_done):
    """Запускает pip install в фоне, стримит stdout построчно."""
    def run():
        proc = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "-U", name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", bufsize=1,
        )
        for line in proc.stdout:
            on_line(line.rstrip())
        proc.wait()
        on_done(proc.returncode == 0)
    threading.Thread(target=run, daemon=True).start()
```

Во время установки — ячейка «Действие» показывает `CTkProgressBar` в неопределённом режиме (`.start()`) и подпись `installing…`. По завершении — обновить статус строки.

**Важно:** `subprocess` должен писать только через thread + `self.after(0, ...)` в UI-тред, иначе Tk упадёт.

#### Таб «LLM keys»

Провайдеры: `OpenAI / Anthropic / YandexGPT / GigaChat`.

Каждая строка:
```
[Название]       [env var]        [  input type=password  ]   [status pill]
OpenAI           OPENAI_API_KEY   sk-proj-••••••3aF2          ✓ активен
Anthropic        ANTHROPIC_API_KEY ..                          ✓ активен
YandexGPT        YANDEX_GPT_KEY   (пусто)                      пусто
GigaChat         GIGACHAT_KEY     (пусто)                      пусто
```

Внизу — кнопки `[Отмена] [Сохранить]`.

Хранение — в `~/.hearsy/credentials.enc` (AES-256 через `cryptography.fernet`).
Приоритет загрузки: env var > credentials.enc.

---

## 3. `app_train_view_ctk.py` — расширение

### 3.1. Секция «Колонки Excel/CSV» (новая, критично)

Сейчас этого блока нет. Добавить сразу после «Файлы обучающей выборки», перед «Векторизация».

```python
def _build_columns_section(parent, app):
    card = Card(parent, title="Колонки файлов",
                subtitle="Укажи имена колонок в Excel/CSV. Значения берутся из config.yaml.")
    card.pack(fill="x", padx=20, pady=(0, 12))

    grid = ctk.CTkFrame(card.body, fg_color="transparent")
    grid.pack(fill="x")
    for i in range(4):
        grid.grid_columnconfigure(i, weight=1, uniform="cols")

    fields = [
        ("col_description",   "Описание",                  "desc"),
        ("col_client",        "Реплики клиента",           "client"),
        ("col_operator",      "Реплики оператора",         "operator"),
        ("col_summary",       "Саммари звонка",            "summary"),
        ("col_answer_short",  "Краткий ответ (короткий)",  "ans_short"),
        ("col_answer_full",   "Ответ (полный)",            "ans_full"),
        ("col_label",         "Целевая метка (label)",     "label"),
    ]
    for i, (key, label, default) in enumerate(fields):
        cell = ctk.CTkFrame(grid, fg_color="transparent")
        cell.grid(row=i // 4, column=i % 4, sticky="ew", padx=4, pady=4)
        _field_label(cell, label)
        e = ctk.CTkEntry(cell, font=font_mono(),
                         placeholder_text=f"'{default}'")
        e.pack(fill="x")
        setattr(app, key, e)   # app.col_description.get() и т.д.
```

### 3.2. Auto-profile + SBERT device + SetFit

После «Векторизация», в том же Card или отдельным Card'ом «Режим запуска»:

```python
_field_label(card.body, "Авто-профиль")
ctk.CTkSegmentedButton(card.body,
    values=["Выкл.", "Smart", "Strict"],
    command=lambda v: setattr(app, "_auto_profile", v),
).set("Smart")

_field_label(card.body, "SBERT device")
ctk.CTkSegmentedButton(card.body,
    values=["Auto", "CPU", "CUDA", "MPS"],
    command=lambda v: setattr(app, "_sbert_device", v.lower()),
).set("Auto")

_field_label(card.body, "SetFit model")
ctk.CTkEntry(card.body, font=font_mono(),
    placeholder_text="sberbank-ai/ruBert-base  или  cointegrated/rubert-tiny2"
).pack(fill="x")
```

### 3.3. Калибровка

Вместо текущего checkbox «CalibratedClassifierCV» — сделать radio или dropdown:

```python
_field_label(card.body, "Калибровка вероятностей")
ctk.CTkSegmentedButton(card.body,
    values=["Off", "Sigmoid", "Isotonic", "Temperature"],
).set("Sigmoid")
```

### 3.4. Веса секций

Если разные колонки текста (`desc`, `client`, `operator`, `summary`, …) — нужна группа слайдеров с весами.

```python
def _build_section_weights(parent, app):
    card = Card(parent, title="Веса секций",
                subtitle="Суммировать не обязательно в 1.0. Применяется при fusion эмбеддингов.")
    card.pack(fill="x", padx=20, pady=(0, 12))
    weights = [
        ("desc",      "Описание",          1.0),
        ("client",    "Клиент",            0.8),
        ("operator",  "Оператор",          0.5),
        ("summary",   "Саммари",           0.6),
        ("ans_short", "Ответ (краткий)",   0.3),
        ("ans_full",  "Ответ (полный)",    0.3),
    ]
    for key, label, default in weights:
        row = ctk.CTkFrame(card.body, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, width=140, anchor="w",
                     font=font_base(), text_color=COLORS["muted"]).pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=f"{default:.2f}", width=50,
                               font=font_mono(), text_color=COLORS["accent2"])
        val_lbl.pack(side="right")
        def on_change(v, lbl=val_lbl, k=key):
            lbl.configure(text=f"{float(v):.2f}")
            setattr(app, f"_weight_{k}", float(v))
        sl = ctk.CTkSlider(row, from_=0, to=2, number_of_steps=40, command=on_change)
        sl.set(default)
        sl.pack(side="left", fill="x", expand=True, padx=10)
```

### 3.5. Advanced (аккордеон)

Последняя секция Train — «Продвинутые настройки», развёрнутая через toggle:

```python
class Accordion(ctk.CTkFrame):
    def __init__(self, parent, title: str, **kw):
        super().__init__(parent, fg_color=COLORS["panel"],
                         corner_radius=10, border_width=1,
                         border_color=COLORS["border2"], **kw)
        self._open = False
        self._header = ctk.CTkButton(
            self, text=f"▸  {title}", anchor="w",
            fg_color="transparent", hover_color=COLORS["hover"],
            text_color=COLORS["fg"], font=font_md_bold(),
            command=self._toggle,
        )
        self._header.pack(fill="x", padx=6, pady=6)
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        # body не pack'ается до открытия
        self._title = title

    def _toggle(self):
        self._open = not self._open
        if self._open:
            self.body.pack(fill="x", padx=18, pady=(0, 14))
            self._header.configure(text=f"▾  {self._title}")
        else:
            self.body.pack_forget()
            self._header.configure(text=f"▸  {self._title}")
```

Внутри Accordion:
- **Optuna** — N trials (int), Direction (`maximize/minimize`), Sampler (TPE/CMA-ES/Random)
- **K-fold CV** — n_splits (3/5/10), Stratified checkbox
- **Confident Learning** — cleanlab on/off, Prune method
- **Label smoothing** — alpha slider 0..0.3
- **LLM augmentation** — enabled, target_per_class, provider
- **Near-duplicate detection** — cosine threshold, keep strategy
- **Hierarchical classification** — enabled, levels (2..4)
- **Anchor texts** — embeddings fixed anchors per class

Каждый подраздел — отдельный Card или нестед Frame.

---

## 4. `app_apply_view_ctk.py` — расширение

### 4.1. Per-class thresholds (критично)

Эта фича в репе не имплементирована. В UI выглядит как редактируемая таблица:

| Класс                   | Порог (0..1)   | F1 на validation | Поведение              |
|---                      |---             |---               |---                     |
| Карты — блокировка      | `[slider 0.50]`| 0.92             | above → assign        |
| Карты — выпуск          | `[slider 0.50]`| 0.88             | above → assign        |
| …                       | …              | …                | …                     |

Глобальный порог (`thr`) остаётся — но применяется только к классам, не переопределённым per-class.

```python
def _build_per_class_thresholds(parent, app, classes):
    card = Card(parent, title="Пороги по классам",
                subtitle="Переопределяют глобальный порог. Оптимально — автоподбор по F1.")
    card.pack(fill="x", padx=20, pady=(0, 12))

    # Кнопки сверху
    actions = ctk.CTkFrame(card.body, fg_color="transparent")
    actions.pack(fill="x", pady=(0, 10))
    ctk.CTkButton(actions, text="✨ Автоподбор порогов",
                  command=app.optimize_thresholds).pack(side="left", padx=(0, 6))
    ctk.CTkButton(actions, text="↺ Сбросить к глобальному",
                  fg_color="transparent", border_width=1,
                  border_color=COLORS["border2"],
                  command=app.reset_thresholds).pack(side="left")

    # Таблица
    for cls in classes:
        row = ctk.CTkFrame(card.body, fg_color=COLORS["panel2"],
                           corner_radius=6, border_width=1, border_color=COLORS["border2"])
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=cls["label"], width=220, anchor="w",
                     font=font_base(), text_color=COLORS["fg"]).pack(side="left", padx=10, pady=8)
        val_lbl = ctk.CTkLabel(row, text=f"{cls['threshold']:.2f}",
                               font=font_mono(), text_color=COLORS["accent2"], width=50)
        val_lbl.pack(side="right", padx=10)
        sl = ctk.CTkSlider(row, from_=0, to=1, number_of_steps=100,
                           command=lambda v, lbl=val_lbl: lbl.configure(text=f"{float(v):.2f}"))
        sl.set(cls["threshold"])
        sl.pack(side="right", padx=10, fill="x", expand=True)
```

### 4.2. Ensemble (2-я модель)

```python
_field_label(card.body, "Ансамбль моделей")
ensemble_on = ctk.CTkSwitch(card.body, text="Включить ансамбль 2 моделей")
ensemble_on.pack(anchor="w", pady=4)

m2_row = ctk.CTkFrame(card.body, fg_color="transparent")
m2_row.pack(fill="x", pady=4)
ctk.CTkEntry(m2_row, placeholder_text="model/baseline_v2_812.joblib",
             font=font_mono()).pack(side="left", fill="x", expand=True, padx=(0, 6))
ctk.CTkButton(m2_row, text="…", width=40, command=app.pick_model_2).pack(side="left")

# Weight slider 0..1
_field_label(card.body, "Вес модели #1 в ансамбле")
# 0.0 = только модель 2, 1.0 = только модель 1
```

### 4.3. LLM-rerank

```python
_field_label(card.body, "LLM-rerank для сомнительных предсказаний")
ctk.CTkSegmentedButton(card.body,
    values=["Off", "Low (< 0.50)", "Medium (< 0.70)", "Top-K only"]
).set("Low (< 0.50)")

# Top-K input
topk = ctk.CTkEntry(card.body, placeholder_text="K = 3")
topk.pack(anchor="w", pady=4)
```

### 4.4. Other-label

```python
other_on = ctk.CTkSwitch(card.body, text="Класс «Другое» — если все пороги не пройдены")
other_on.pack(anchor="w", pady=4)
```

### 4.5. Ambiguity detector

```python
amb_on = ctk.CTkSwitch(card.body, text="Детектор неоднозначности (top1 − top2 < 0.10)")
amb_on.pack(anchor="w", pady=4)
```

---

## 5. `app_cluster_view_ctk.py` — расширение

### 5.1. UMAP / PCA параметры

Новый Card «Проекция и редукция размерности» между «Алгоритм» и «Визуализация»:

```python
def _build_projection_card(parent, app):
    card = Card(parent, title="Проекция (UMAP)",
                subtitle="2D-проекция для визуализации; не влияет на сам алгоритм кластеризации.")
    card.pack(fill="x", padx=20, pady=(0, 12))

    grid = ctk.CTkFrame(card.body, fg_color="transparent")
    grid.pack(fill="x")
    for i in range(4):
        grid.grid_columnconfigure(i, weight=1, uniform="umap")

    # n_components
    c0 = ctk.CTkFrame(grid, fg_color="transparent")
    c0.grid(row=0, column=0, sticky="ew", padx=4)
    _field_label(c0, "n_components"); e1 = ctk.CTkEntry(c0, font=font_mono()); e1.insert(0, "2"); e1.pack(fill="x")

    # n_neighbors
    c1 = ctk.CTkFrame(grid, fg_color="transparent")
    c1.grid(row=0, column=1, sticky="ew", padx=4)
    _field_label(c1, "n_neighbors"); e2 = ctk.CTkEntry(c1, font=font_mono()); e2.insert(0, "15"); e2.pack(fill="x")

    # min_dist
    c2 = ctk.CTkFrame(grid, fg_color="transparent")
    c2.grid(row=0, column=2, sticky="ew", padx=4)
    _field_label(c2, "min_dist"); e3 = ctk.CTkEntry(c2, font=font_mono()); e3.insert(0, "0.1"); e3.pack(fill="x")

    # metric
    c3 = ctk.CTkFrame(grid, fg_color="transparent")
    c3.grid(row=0, column=3, sticky="ew", padx=4)
    _field_label(c3, "metric")
    ctk.CTkOptionMenu(c3, values=["cosine", "euclidean", "manhattan", "hellinger"],
                      font=font_base()).pack(fill="x")

    # PCA перед UMAP
    pca_on = ctk.CTkSwitch(card.body, text="PCA → UMAP (для ускорения при n > 10к)")
    pca_on.pack(anchor="w", pady=(10, 0))

    pca_row = ctk.CTkFrame(card.body, fg_color="transparent")
    pca_row.pack(fill="x", pady=4)
    _field_label(pca_row, "PCA n_components")
    e_pca = ctk.CTkEntry(pca_row, width=80, font=font_mono()); e_pca.insert(0, "50"); e_pca.pack(side="left")
```

### 5.2. HDBSCAN детальные параметры (показывать только если выбран HDBSCAN)

```python
def _build_hdbscan_params(parent, app):
    card = Card(parent, title="HDBSCAN параметры")
    # visible only when algo == 'hdbscan'
    card.pack(fill="x", padx=20, pady=(0, 12))

    _field_label(card.body, "min_cluster_size")
    ctk.CTkEntry(card.body, font=font_mono(), placeholder_text="15").pack(fill="x")

    _field_label(card.body, "min_samples")
    ctk.CTkEntry(card.body, font=font_mono(), placeholder_text="10").pack(fill="x")

    _field_label(card.body, "cluster_selection_epsilon")
    ctk.CTkEntry(card.body, font=font_mono(), placeholder_text="0.0").pack(fill="x")

    ctk.CTkSwitch(card.body,
        text="Reclustering шумовых точек (через KMeans)"
    ).pack(anchor="w", pady=4)
```

### 5.3. T5 summarization + Merge similar

```python
def _build_postproc_card(parent, app):
    card = Card(parent, title="Пост-обработка кластеров")
    card.pack(fill="x", padx=20, pady=(0, 12))

    t5_on = ctk.CTkSwitch(card.body,
        text="T5 суммаризация обращений внутри кластера")
    t5_on.pack(anchor="w", pady=4)

    _field_label(card.body, "T5 модель")
    ctk.CTkEntry(card.body, font=font_mono(),
                 placeholder_text="cointegrated/rut5-base-absum").pack(fill="x")

    merge_on = ctk.CTkSwitch(card.body,
        text="Сливать похожие кластеры (cosine sim > threshold)")
    merge_on.pack(anchor="w", pady=4)

    _field_label(card.body, "Merge threshold")
    # slider 0.7..0.99
    _slider_with_label(card.body, from_=0.7, to=0.99, steps=29, default=0.85)
```

### 5.4. Streaming / incremental (только для KMeans)

```python
streaming_on = ctk.CTkSwitch(card.body, text="Streaming (MiniBatch) режим")
streaming_on.pack(anchor="w", pady=4)
_field_label(card.body, "Batch size")
ctk.CTkEntry(card.body, font=font_mono(), placeholder_text="1000").pack(fill="x")
```

### 5.5. BERTopic / LDA параметры

Показывать conditional-блок в зависимости от выбранного алгоритма в `algo_var`:

```python
def _on_algo_change(value, app):
    # скрыть все блоки
    app._hdbscan_card.pack_forget()
    app._bertopic_card.pack_forget()
    app._lda_card.pack_forget()
    # показать нужный
    if value == "hdbscan":   app._hdbscan_card.pack(fill="x", padx=20, pady=(0, 12))
    elif value == "bertopic":app._bertopic_card.pack(fill="x", padx=20, pady=(0, 12))
    elif value == "lda":     app._lda_card.pack(fill="x", padx=20, pady=(0, 12))
```

**BERTopic** knobs: `min_topic_size (int)`, `nr_topics (int | 'auto')`, `diversity (0..1)`.
**LDA** knobs: `n_topics (int)`, `alpha (str: 'auto'|'asymmetric'|float)`, `iterations (int)`.

---

## 6. Стилизация range-слайдеров

В HTML — кастомный `input[type=range]` (cм. `styles.css` строки 400–458). В CTk эквивалент — `CTkSlider`, и он **уже настроен** через `_build_ctk_theme_json` в `ui_theme_ctk.py`:

```python
"CTkSlider": {
    "corner_radius": 1000,
    "button_corner_radius": 1000,
    "border_width": 0,
    "button_length": 0,
    "fg_color": pair(p["entry"]),
    "progress_color": pair(p["accent"]),
    "button_color": pair(p["accent2"]),
    "button_hover_color": pair(p["accent"]),
}
```

Ничего править не нужно. **Главное** — всегда использовать `CTkSlider`, а не `tk.Scale`.

---

## 7. Hotkeys

В HTML-прототипе не реализованы явно, но структура совместима. Добавить в `bootstrap_ctk_demo.py::__init__`:

```python
self.bind("<F5>",             lambda e: self.start_current_task())
self.bind("<Control-Return>", lambda e: self.start_current_task())
self.bind("<Escape>",         lambda e: self.cancel_current_task())
self.bind("<Control-Key-1>",  lambda e: self._switch_tab("train"))
self.bind("<Control-Key-2>",  lambda e: self._switch_tab("apply"))
self.bind("<Control-Key-3>",  lambda e: self._switch_tab("cluster"))
self.bind("<Control-h>",      lambda e: HistoryDialog(self))
self.bind("<Control-comma>",  lambda e: SettingsDialog(self))
```

Показать пользователю — добавить иконку `?` в main-header, клик открывает оверлей со списком хоткеев.

---

## 8. Переключение темы без destroy()

Сейчас `_switch_theme` делает `self.destroy() + new_app.mainloop()` — это работает, но теряется состояние (прогресс, выбранный кластер и т.п.).

Решение: сохранять состояние перед destroy, передавать его в new app:

```python
def _switch_theme(self, label):
    mapping = {"Teal": "dark-teal", "Paper": "paper", "CRT": "amber-crt"}
    state = self._dump_state()     # вытащить всё важное
    apply_theme(mapping[label])
    self.destroy()
    new_app = FakeApp()
    new_app._restore_state(state)
    new_app.mainloop()

def _dump_state(self):
    return {
        "tab": self._current_tab,
        "train_files": getattr(self, "_train_files", []),
        "apply_file":  getattr(self, "_apply_file", None),
        "cluster_selected": getattr(self, "_cluster_selected", 0),
        # ...
    }
```

---

## 9. Прочее

### 9.1. `CTkScrollableFrame` — известная проблема

Внутри `CTkScrollableFrame` нельзя размещать другой `CTkScrollableFrame` без явного `height=`. Всегда ставить высоту на внутренних.

### 9.2. Matplotlib scatter — перерисовка при смене темы

`FigureCanvasTkAgg` не перекрашивается автоматически. При `apply_theme` нужно:

```python
fig.patch.set_facecolor(COLORS["entry"])
ax.set_facecolor(COLORS["entry"])
for spine in ax.spines.values():
    spine.set_color(COLORS["border2"])
ax.tick_params(colors=COLORS["muted"])
canvas.draw_idle()
```

### 9.3. Hardware info (опционально)

```python
import psutil, torch
def hw_snapshot():
    return {
        "cpu":   f"{psutil.cpu_count(logical=True)} cores · {int(psutil.cpu_percent())}%",
        "ram":   f"{psutil.virtual_memory().used / 1e9:.1f} / {psutil.virtual_memory().total / 1e9:.0f} ГБ",
        "gpu":   torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—",
        "torch": torch.__version__,
    }
```

Обновлять раз в 2 секунды через `self.after(2000, refresh_hw)`.

---

## 10. Чек-лист перед мёрджем

- [ ] Все три таба отображаются без ошибок консоли
- [ ] `python bootstrap_ctk_demo.py` запускается на все 3 темы без разрушения layout
- [ ] Диалоги `История / Артефакты / Настройки` открываются из sidebar
- [ ] В Настройках → Зависимости: статус пакетов детектируется корректно (проверить на `pip uninstall cleanlab && python bootstrap_ctk_demo.py`)
- [ ] Установка через `pip install -U cleanlab` в UI — работает, статус обновляется
- [ ] Все пороги / слайдеры / optionmenu привязаны к `app.*` переменным и читаются в бизнес-логике
- [ ] Hotkeys работают
- [ ] Brand: `Hs / Hearsy / Classifier · ru`
- [ ] `pytest -q` — зелёный (smoke-тест создания окна во все темы)

---

## 11. PR description template

```md
## UI Migration — Hearsy / CustomTkinter

### Что сделано
- Полностью переписан UI-слой с ttk на CustomTkinter
- Брендинг: BankReasonTrainer → Hearsy (UI strings)
- Добавлены диалоги: История экспериментов, Артефакты моделей, Настройки (Зависимости + LLM keys)
- Train-таб: +Колонки, +Auto-profile, +SBERT device, +SetFit, +Калибровка, +Веса секций, +Advanced аккордеон (Optuna/K-fold/CL/Smoothing/Aug)
- Apply-таб: +Per-class thresholds, +Ensemble, +LLM-rerank, +Other-label, +Ambiguity
- Cluster-таб: +UMAP params, +PCA, +HDBSCAN detailed, +T5 summarization, +Merge similar, +Streaming, +BERTopic/LDA knobs
- Hotkeys: F5 / Ctrl+Enter / Esc / Ctrl+1..3 / Ctrl+H / Ctrl+,

### Что НЕ затронуто
- Бизнес-логика (ML-pipeline, embeddings, clustering, LLM)
- `experiment_log.jsonl` формат
- API методов `app.*` — только дополнены новыми

### Как проверить
```bash
pip install -r requirements.txt
python bootstrap_ctk_demo.py    # smoke-demo
python app.py                    # полный запуск
pytest -q                        # тесты
```

### Скриншоты
(приложить 3 скриншота — по одному на каждый таб в dark-teal)
```

---

## 12. Если застрял

1. **Сначала** открой `BankReasonTrainer.html` в браузере — это эталон визуала.
2. **Потом** — смотри соответствующий `.jsx`-файл для структуры (`tab_train.jsx`, `tab_apply.jsx`, `tab_cluster.jsx`, `overlays.jsx`).
3. **Цвета / радиусы / отступы** — всегда из `ui_theme_ctk.py::COLORS`, не хардкодить hex.
4. **Шрифты** — через `font_*()` хелперы, не `ctk.CTkFont(...)` inline.
5. Если виджет не отображается — проверь, не забыл ли `.pack()` / `.grid()`.
