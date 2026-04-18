# Security checklist: model loading

Применять для всех путей загрузки `.joblib`:

1. Подтверждение доверия к файлу (consent + trusted cache).
2. Разрешены только whitelist-расширения (`.joblib`).
3. Валидация структуры артефакта (`dict` bundle).
4. Проверка `schema_version` (тип + верхняя граница).
5. Проверка `artifact_type` и обязательных ключей.
6. Ошибки загрузки поднимаются как единый тип (`ModelLoadError`/доменные обёртки).
7. Логи не должны содержать полный payload модели или секреты.
8. Для новых callsite запрещён прямой `joblib.load` вне `model_loader.py`.

