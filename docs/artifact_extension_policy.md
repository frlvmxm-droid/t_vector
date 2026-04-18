# Artifact extension policy

Матрица разрешённых расширений по сценариям:

| Сценарий | Разрешённые расширения | Комментарий |
|---|---|---|
| Apply model load (`app_apply._load_model_pkg`) | `.joblib` | Только train-bundle приложения для инференса/совместимости |
| Train base model load (`app_train.pick_base_model`) | `.joblib` | Базовая модель дообучения должна быть bundle приложения |
| Generic loader default (`model_loader.load_model_artifact`) | `.joblib`, `.safetensors` | Общий безопасный loader для разных сценариев |

## Принцип

Per-callsite policy приоритетнее default: конкретный callsite может сужать список форматов.
