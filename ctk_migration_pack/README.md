# BankReasonTrainer · CustomTkinter UI Pack

Готовый набор файлов для миграции UI-слоя `BankReasonTrainer` с `ttk` на CustomTkinter.

## Содержимое

| Файл | Назначение |
|---|---|
| `ui_theme_ctk.py` | Палитры и JSON-темы (Dark Teal / Paper / Amber-CRT) |
| `app_train_view_ctk.py` | Вкладка «Обучение» + переиспользуемые `Card`/`Pill`/`Metric` |
| `app_apply_view_ctk.py` | Вкладка «Классификация» с таблицей предсказаний |
| `app_cluster_view_ctk.py` | Вкладка «Кластеризация» с matplotlib-scatter |
| `bootstrap_ctk_demo.py` | Запускаемое демо со всеми вкладками и переключателем тем |
| `requirements.txt` | `customtkinter>=5.2`, `matplotlib>=3.7` |
| `MIGRATION.md` | Пошаговая инструкция миграции (передать Claude Code) |

## Запуск демо

```bash
pip install -r requirements.txt
python bootstrap_ctk_demo.py
```

## Миграция в прод

Читай `MIGRATION.md` — там пошаговый план для Claude Code.

## Контракты

Новые view'ы ожидают у `app` те же методы, что и старые `app_*_view.py`. Никаких новых контрактов не вводится. Бизнес-логику (`*_service.py`, `*_workflow.py`) трогать не нужно.
