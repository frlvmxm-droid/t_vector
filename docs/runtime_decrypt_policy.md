# Runtime decrypt policy (LLM snapshot keys)

Матрица поведения при ошибке расшифровки значения snapshot:

| Mode | Поведение при ошибке | Fail-closed | Error code |
|---|---|---|---|
| `strict` | выбрасывается `LLMSnapshotDecryptError` | Да | настраиваемый (`strict_error_code`) |
| `fail_closed` | возвращается пустая строка | Да | нет |
| `legacy` | возвращается исходное значение | Нет | нет |

## Как выбирается режим

1. Если явно передан `decrypt_mode`, он имеет приоритет.
2. Иначе используется env-конфиг:
   - `STRICT_SECRET_DECRYPT=1` / `strict_secret_decrypt=1` / `LLM_SNAPSHOT_DECRYPT_STRICT=1` → `strict`
   - `LLM_SNAPSHOT_DECRYPT_FAIL_CLOSED=1` → `fail_closed`
   - иначе → `legacy`

## Рекомендуемая эксплуатация

- production: `strict`
- controlled rollout: `fail_closed`
- migration/compatibility only: `legacy`
