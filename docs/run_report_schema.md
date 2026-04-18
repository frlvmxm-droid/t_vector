# Unified run-report schema

`run_report.build_unified_run_report(...)` формирует единый JSON-пейлоад для `train` / `apply` / `cluster`.
`run_report.write_unified_run_report(...)` сразу записывает его в файл отчёта.

## Поля

- `schema_version`: версия схемы (сейчас `1`)
- `pipeline`: `train|apply|cluster`
- `status`: `ok|warning|error` (или иное значение уровня выполнения)
- `params`: параметры запуска (конфиг, batch/chunk, feature flags)
- `metrics`: метрики качества/скорости (macro_f1, throughput и т.п.)
- `timings`: тайминги этапов (`fit_sec`, `predict_sec`, ...)
- `errors`: список ошибок в структурированном формате (`error_code`, `message`, `stage`)
- `metadata`: произвольные доп. данные (host, git_sha, dataset info)

## Совместимость

`run_observability` поддерживает:
1) legacy-плоский отчёт с метриками в top-level;
2) unified-формат с метриками в `metrics`.
