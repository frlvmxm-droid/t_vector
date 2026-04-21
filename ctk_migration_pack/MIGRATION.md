# Миграция UI на CustomTkinter — инструкция

Этот пакет содержит готовые view'ы на CustomTkinter, которые повторяют дизайн из HTML-прототипа `BankReasonTrainer.html`. Логика приложения не трогается — переписывается только UI-слой.

## Файлы пакета

```
ui_theme_ctk.py           # Палитра + JSON-тема (Dark Teal / Paper / Amber-CRT)
app_train_view_ctk.py     # Вкладка «Обучение»
app_apply_view_ctk.py     # Вкладка «Классификация»
app_cluster_view_ctk.py   # Вкладка «Кластеризация» (с matplotlib scatter)
bootstrap_ctk_demo.py     # Запускаемое демо
BankReasonTrainer.html    # Исходный HTML-прототип (для справки)
MIGRATION.md              # Этот файл
```

---

## Быстрый старт

```bash
pip install customtkinter matplotlib
python bootstrap_ctk_demo.py
```

Должно открыться окно с боковой навигацией, тремя вкладками и переключателем тем внизу sidebar.

---

## План миграции (для Claude Code)

> Передай этот документ Claude Code — он выполнит все шаги по порядку.

### Шаг 1. Установить зависимости

```bash
pip install customtkinter>=5.2 matplotlib
```

Добавь в `requirements.txt`:
```
customtkinter>=5.2.0
matplotlib>=3.7
```

### Шаг 2. Скопировать новые файлы в корень проекта

Файлы из этого архива (`ui_theme_ctk.py`, `app_train_view_ctk.py`, `app_apply_view_ctk.py`, `app_cluster_view_ctk.py`) положить рядом со старыми `ui_theme.py`, `app_train_view.py` и т.д. Старые **не удалять** на этом этапе — они нужны как fallback и референс.

### Шаг 3. Адаптировать `app.py`

В начале файла:

```python
import customtkinter as ctk
from ui_theme_ctk import apply_theme, COLORS

# Применить тему ДО создания корневого окна
apply_theme("dark-teal")     # или "paper" / "amber-crt"
```

Заменить наследование основного класса:

```python
# Было:
class BankReasonTrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # ...

# Стало:
class BankReasonTrainerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.configure(fg_color=COLORS["bg"])
        # ...
```

### Шаг 4. Заменить Notebook на боковую навигацию

Старый `ttk.Notebook` заменить на структуру `sidebar + content` из `bootstrap_ctk_demo.py` (`_build_sidebar()`, `_build_main()`, `_switch_tab()`). Реализация ~70 строк.

### Шаг 5. Подключить новые view-функции

В местах, где раньше вызывались `build_train_files_card`, `build_apply_files_card`, `build_cluster_files_card` (из старых `app_*_view.py`), вызывать новые сборщики:

```python
from app_train_view_ctk   import build_train_tab
from app_apply_view_ctk   import build_apply_tab
from app_cluster_view_ctk import build_cluster_tab

# В _show_train():
build_train_tab(self, self.content_frame)

# В _show_apply():
build_apply_tab(self, self.content_frame)

# В _show_cluster():
build_cluster_tab(self, self.content_frame)
```

### Шаг 6. Контракты «app»-объекта

Новые view'ы ожидают у `app` те же методы, что и старые — никаких новых не вводится. Список см. в docstring каждого `app_*_view_ctk.py`. В частности:

**Train:** `add_train_files`, `add_train_folder`, `show_dataset_stats`, `start_training`
**Apply:** `pick_model`, `pick_apply_file`, `start_apply`, `export_predictions`
**Cluster:** `add_cluster_files`, `add_cluster_folder`, `start_cluster`, `export_cluster_results`, `_auto_detect_cluster_params`

Если какой-то метод не реализован — view покажет кнопку, но клик ничего не сделает (использует `getattr(app, name, lambda: None)`).

### Шаг 7. Тестирование

```bash
# Запусти полный набор тестов
pytest -q

# Запусти приложение
python bootstrap_run.py
```

Существующие тесты не должны сломаться — мы не трогаем бизнес-логику.

### Шаг 8. Обработка специфичных виджетов

Эти виджеты остаются от Tkinter/ttk — CustomTkinter их не покрывает:

| Виджет | Решение |
|---|---|
| `tk.Listbox` (lb_train, lb_cluster) | Оставить, стилизовать через `bg=COLORS["entry"], fg=COLORS["fg"]` |
| `ttk.Treeview` | Оставить, стилизовать через `ttk.Style().configure()` с цветами из `COLORS` |
| Tooltips (`Tooltip` class) | Работает поверх любых виджетов, не трогать |
| Scatter-плот | Уже сделан через matplotlib в `app_cluster_view_ctk.py` |
| Tray icon, menubar | Не затрагиваются — это OS-level |

### Шаг 9. Удалить старые view'ы (опционально)

Когда новый UI работает стабильно — старые `app_train_view.py`, `app_apply_view.py`, `app_cluster_view.py`, `ui_theme.py` можно удалить. Но **проверь, что они нигде больше не импортируются** (`grep -r "from app_train_view" .`).

---

## Структура CustomTkinter view'ов

Все три view'а используют один паттерн:

```python
def build_X_tab(app, parent: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
    scroll = ctk.CTkScrollableFrame(parent, fg_color=COLORS["bg"])
    scroll.pack(fill="both", expand=True)

    # 1. Card с файлами
    files_card = Card(scroll, title="...", subtitle="...")
    files_card.pack(...)

    # 2. Card с конфигурацией (часто 2 колонки через grid)
    # 3. Card с прогрессом / результатами
    # 4. Card с детализацией

    return scroll
```

Переиспользуемые компоненты (определены в `app_train_view_ctk.py`, импортируются в остальных):

- `Card(parent, title, subtitle, right)` — панель с заголовком
- `Pill(parent, text, kind)` — бейдж (default/accent/success/warning/error)
- `Metric(parent, label, value, delta)` — KPI-тайл
- `_field_label(parent, text)` — uppercase-подпись над полем
- `_separator(parent)` — горизонтальная линия

---

## Переключение темы

```python
from ui_theme_ctk import apply_theme

apply_theme("paper")          # светлая, бумажная
apply_theme("dark-teal")      # тёмная teal (по умолчанию)
apply_theme("amber-crt")      # ретро CRT
```

⚠️ CustomTkinter применяет тему один раз при старте. Для смены на лету нужно пересоздать root-окно (см. `_switch_theme()` в `bootstrap_ctk_demo.py`).

---

## Ограничения, о которых стоит знать

1. **Скруглённые карточки внутри `CTkScrollableFrame`** — иногда обрезаются по краям. Если критично — оборачивай в `CTkFrame(corner_radius=0)`.
2. **Tooltip от старого `ui_widgets_tk.py`** работает с любыми Tk-виджетами, включая CTk, потому что CTk наследуется от `tk.Frame`/`tk.Canvas`.
3. **Treeview-стилизация под dark-теmu** требует ручной настройки `ttk.Style()` — пример ниже.

```python
import tkinter.ttk as ttk
from ui_theme_ctk import COLORS

style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview",
    background=COLORS["entry"], foreground=COLORS["fg"],
    fieldbackground=COLORS["entry"], borderwidth=0)
style.configure("Treeview.Heading",
    background=COLORS["panel2"], foreground=COLORS["accent2"],
    relief="flat")
style.map("Treeview", background=[("selected", COLORS["select"])])
```

---

## Чек-лист готовности

- [ ] `pip install customtkinter matplotlib` выполнено
- [ ] Файлы `*_view_ctk.py` лежат в корне проекта
- [ ] `apply_theme()` вызвана до создания root
- [ ] `app.py` наследуется от `ctk.CTk`
- [ ] Notebook заменён на sidebar + content
- [ ] Все три `build_*_tab()` подключены
- [ ] `pytest` проходит зелёным
- [ ] Приложение запускается через `python bootstrap_run.py`
- [ ] Tooltip'ы работают
- [ ] Treeview / Listbox стилизованы
- [ ] Старые `app_*_view.py` удалены (или помечены как deprecated)

---

## Если что-то пошло не так

- **Фон везде серый, цвета не применились** → `apply_theme()` вызвана после создания `ctk.CTk()`. Перенеси выше.
- **Шрифты слишком крупные / мелкие** → подкрути в `ui_theme_ctk.py` функции `font_*()`.
- **Scatter не отображается** → `pip install matplotlib` пропущен.
- **`AttributeError: 'BankReasonTrainerApp' object has no attribute 'add_train_files'`** → метод не реализован у `app`. Добавь его (даже пустую заглушку с `pass`).

Удачи с миграцией.
