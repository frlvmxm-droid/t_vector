# UI refactor plan (incremental)

Цель: уменьшить связанность крупных UI-модулей (`app_apply.py`, `app_cluster.py`) и постепенно перейти к presenter/view-model паттерну.

## Этапы

1. **Presenter helpers**
   - вынос форматирования сообщений и view-state маппинга в `ui/tabs/*_presenter.py`;
   - UI-файл оставляет только wiring и callbacks.

2. **View-model snapshots**
   - отдельные DTO/снапшоты для входных параметров запуска;
   - минимум бизнес-логики в Tk callbacks.

3. **Service orchestration**
   - orchestration + lifecycle в сервисы (`*_runtime_service.py`);
   - UI только делегирует вызовы.

Начальный шаг уже применён для Apply auto-profile лога через `ui/tabs/apply_presenter.py`.
