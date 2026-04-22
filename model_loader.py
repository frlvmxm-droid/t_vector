"""Единая безопасная загрузка модельных .joblib-артефактов."""
from __future__ import annotations

import getpass
import hashlib
import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from exceptions import ModelLoadError, SchemaError

_HASH_CACHE_LOCK = threading.Lock()
_HASH_CACHE: dict[str, tuple[tuple[int, int], str]] = {}

# Тип callback-функции подтверждения: принимает label → True (доверяем) / False (отказ/timeout)
ConfirmFn = Callable[[str], bool]


def _canonical_path_key(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return str(Path(str(resolved)).resolve()).casefold()


def _file_sig(path: Path) -> tuple[int, int]:
    st = path.stat()
    return int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size)


def file_sha256(path: str | Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Потоковый SHA-256 без чтения файла целиком в RAM."""
    path_obj = Path(path)
    cache_key = str(path_obj.resolve())
    sig = _file_sig(path_obj)
    with _HASH_CACHE_LOCK:
        cached = _HASH_CACHE.get(cache_key)
        if cached and cached[0] == sig:
            return cached[1]

    h = hashlib.sha256()
    with path_obj.open("rb") as f:
        while True:
            chunk = f.read(max(1024, int(chunk_size)))
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    with _HASH_CACHE_LOCK:
        _HASH_CACHE[cache_key] = (sig, digest)
        if len(_HASH_CACHE) > 1024:
            _HASH_CACHE.pop(next(iter(_HASH_CACHE)))
    return digest


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    base_dir = base_dir.resolve()
    target_path = target_path.resolve()
    try:
        target_path.relative_to(base_dir)
        return True
    except ValueError:
        return False


def is_safe_path_strict(base_dir: Path, target_path: Path) -> bool:
    """Stricter variant: rejects any path containing a symlink component.

    Mitigates TOCTOU attacks where a symlink can be swapped between the
    resolve() call and the subsequent file open. Walks the target's
    ancestor chain; if ANY component is a symlink, the path is refused.
    The final resolved path must still lie within ``base_dir``.

    Use this for high-trust operations (e.g. joblib trust-store loads);
    `is_safe_path` remains for plain boundary checks.
    """
    try:
        resolved_base = base_dir.resolve()
        # Refuse if any component on the way to the target is a symlink.
        current = target_path
        # Walk the user-supplied path (not the resolved one) to detect
        # a symlink swap that happens BEFORE resolution.
        while True:
            if current.is_symlink():
                return False
            parent = current.parent
            if parent == current:
                break
            current = parent
        resolved_target = target_path.resolve()
        resolved_target.relative_to(resolved_base)
        return True
    except (OSError, ValueError):
        return False


class TrustStore:
    """Изолированное хранилище доверенных путей моделей.

    Заменяет три атрибута, ранее хранившихся напрямую в App-объекте:
      _trusted_model_paths, _trusted_model_hashes, _revoked_model_paths.

    Все операции работают с каноническими путями (case-folded resolved).
    """

    def __init__(self) -> None:
        self._trusted_paths: set[str] = set()
        self._trusted_hashes: dict[str, str] = {}
        self._revoked_paths: set[str] = set()

    def is_trusted(self, path: str) -> bool:
        """Проверяет, находится ли путь в доверенном списке."""
        return _canonical_path_key(path) in self._trusted_paths

    def is_revoked(self, path: str) -> bool:
        """Проверяет, был ли путь явно отозван."""
        return _canonical_path_key(path) in self._revoked_paths

    def add_trusted(self, path: str, sha256: str = "") -> None:
        """Добавляет путь в доверенный список (с опциональным хешем)."""
        canonical = _canonical_path_key(path)
        self._trusted_paths.add(canonical)
        if sha256:
            self._trusted_hashes[canonical] = sha256

    def revoke(self, path: str, *, logger: Any = None) -> None:
        """Отзывает доверие к пути."""
        canonical = _canonical_path_key(path)
        self._trusted_paths.discard(canonical)
        self._trusted_hashes.pop(canonical, None)
        self._revoked_paths.add(canonical)
        if logger:
            logger.info("model trust revoked path=%s", path)

    def get_hash(self, path: str) -> str | None:
        """Возвращает сохранённый SHA-256 для пути (если есть)."""
        canonical = _canonical_path_key(path)
        val = self._trusted_hashes.get(canonical)
        return val if isinstance(val, str) and val else None

    def trusted_canonical_paths(self) -> set[str]:
        """Возвращает копию множества доверенных канонических путей."""
        return set(self._trusted_paths)

    def clear_trust(self, path: str) -> None:
        """Снимает доверие к пути без занесения в чёрный список (soft revoke).

        Используется при обнаружении изменения файла на диске: позволяет
        пользователю заново подтвердить обновлённую модель.
        """
        canonical = _canonical_path_key(path)
        self._trusted_paths.discard(canonical)
        self._trusted_hashes.pop(canonical, None)

    def check_hash_changed(self, path: str, current_hash: str) -> bool:
        """Возвращает True, если файл был изменён с момента последнего подтверждения."""
        prev = self.get_hash(path)
        return bool(prev and current_hash and prev != current_hash)


def make_tkinter_confirm_fn(app: Any, timeout_sec: float = 30.0) -> ConfirmFn:
    """Создаёт callback подтверждения через Tkinter messagebox (thread-safe).

    Предназначен для вызова из фонового потока: планирует диалог через
    app.after() на главный Tkinter-поток и ожидает ответа.

    Импорт tkinter отложен до момента вызова фабрики, чтобы модуль
    model_loader можно было импортировать без GUI (тесты, CLI).
    """
    def _confirm(label: str) -> bool:
        from tkinter import messagebox  # отложенный импорт — только при наличии GUI

        done = threading.Event()
        decision = [False]

        def _ask() -> None:
            decision[0] = messagebox.askyesno(
                "Безопасность",
                f"Файл «{label}» (.joblib) содержит сериализованный Python-код.\n\n"
                "⚠ Загружай ТОЛЬКО файлы из доверенных источников!\n"
                "Вредоносный файл может выполнить произвольный код.\n\n"
                "Продолжить загрузку?",
            )
            done.set()

        app.after(0, _ask)
        if not done.wait(timeout=timeout_sec):
            return False
        return bool(decision[0])

    return _confirm


def ensure_trusted_model_path(
    store: TrustStore,
    path: str,
    label: str = "Модель",
    *,
    confirm_fn: ConfirmFn,
    logger: Any = None,
) -> bool:
    """Проверяет, что путь подтверждён пользователем в текущей сессии.

    Args:
        store:      TrustStore, хранящий trust-state для текущей сессии.
        path:       Путь к файлу модели.
        label:      Отображаемое имя для диалога подтверждения.
        confirm_fn: Callback UI-подтверждения (например, make_tkinter_confirm_fn(app)).
        logger:     Опциональный логгер.

    Returns:
        True  — путь уже доверенный или пользователь подтвердил.
        False — отказ пользователя, timeout или отозванный путь.
    """
    # Вычисляем текущий хеш для детектирования изменений файла
    current_hash = ""
    try:
        current_hash = file_sha256(path)
    except OSError as ex:
        if logger:
            logger.warning("unable to hash model path=%s reason=%s", path, ex)

    # Если путь уже доверенный — проверяем не изменился ли файл на диске
    if store.is_trusted(path):
        if store.check_hash_changed(path, current_hash):
            if logger:
                logger.warning("trusted model changed on disk; clearing trust path=%s", path)
            store.clear_trust(path)
        else:
            return True

    # Явно отозванные пути — немедленный отказ
    if store.is_revoked(path):
        if logger:
            logger.warning("model path explicitly revoked path=%s", path)
        return False

    # Запрашиваем подтверждение через переданный callback
    approved = confirm_fn(label)

    if approved:
        store.add_trusted(path, current_hash)
    return approved


def revoke_trusted_model_path(store: TrustStore, path: str, *, logger: Any = None) -> None:
    """Отзывает доверие к ранее подтверждённому пути модели."""
    store.revoke(path, logger=logger)


def get_trusted_model_hash(store: TrustStore, path: str) -> str | None:
    """Возвращает ранее сохранённый SHA-256 для доверенного пути (если есть)."""
    return store.get_hash(path)


def load_model_artifact(
    path: str,
    *,
    supported_schema_version: int = 1,
    expected_artifact_types: Iterable[str] | None = None,
    required_keys: Iterable[str] | None = None,
    required_key_types: Mapping[str, type | tuple[type, ...]] | None = None,
    expected_sha256: str | None = None,
    precomputed_sha256: str | None = None,
    allow_missing_schema: bool = True,
    allowed_extensions: Iterable[str] = (".joblib", ".safetensors"),
    allowed_base_dir: str | None = None,
    require_trusted: bool = False,
    trusted_paths: Iterable[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Загружает и валидирует model-bundle в единой точке."""
    if allowed_base_dir:
        if not is_safe_path(Path(allowed_base_dir), Path(path)):
            raise ModelLoadError(
                f"[UNSAFE_PATH] Запрещён путь вне разрешённой директории: {path}"
            )
    if require_trusted:
        trusted_norm = {_canonical_path_key(p) for p in (trusted_paths or ())}
        if _canonical_path_key(path) not in trusted_norm:
            raise ModelLoadError(
                f"[UNTRUSTED_MODEL_PATH] Путь не подтверждён как доверенный: {path}"
            )

    started_at = datetime.now(UTC).isoformat()
    actor = getpass.getuser()
    sha256 = (precomputed_sha256 or "").strip().lower() or file_sha256(path)
    if logger:
        logger.info(
            "model artifact load requested user=%s ts_utc=%s file=%s sha256=%s",
            actor,
            started_at,
            path,
            sha256,
        )
    if log_fn:
        log_fn(f"user={actor} ts_utc={started_at} sha256={sha256}")
    if expected_sha256 and sha256.lower() != expected_sha256.lower():
        raise ModelLoadError(
            f"[SHA256_MISMATCH] sha256={sha256} не совпадает с ожидаемым {expected_sha256}."
        )

    ext = Path(path).suffix.lower()
    allowed = {e.lower() for e in allowed_extensions}
    if ext not in allowed:
        raise ModelLoadError(
            f"[UNSUPPORTED_EXTENSION] Неподдерживаемый формат файла: {ext or '<без расширения>'}. "
            f"Разрешены: {', '.join(sorted(allowed))}."
        )

    try:
        if ext == ".safetensors":
            try:
                from safetensors import safe_open
            except Exception as ex:
                raise ModelLoadError(
                    "Для загрузки .safetensors установите пакет safetensors."
                ) from ex
            tensors: dict[str, Any] = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            pkg = {
                "artifact_type": "safetensors",
                "schema_version": supported_schema_version,
                "tensors": tensors,
            }
        else:
            pkg = joblib.load(path)
    except Exception as ex:
        raise ModelLoadError(f"Не удалось прочитать модель: {ex}") from ex

    if not isinstance(pkg, dict):
        raise ModelLoadError("Некорректный формат модели: ожидается dict bundle.")

    if expected_artifact_types:
        allowed_types = set(expected_artifact_types)
        artifact_type = pkg.get("artifact_type")
        if artifact_type not in allowed_types:
            if artifact_type is None and allow_missing_schema:
                # Старый бандл без artifact_type — считаем legacy (аналогично отсутствию schema_version)
                _warn = (
                    "⚠ artifact_type отсутствует — модель сохранена старой версией программы. "
                    "Переобучите модель для полной совместимости."
                )
                if log_fn:
                    log_fn(_warn)
                if logger:
                    logger.warning("legacy model: artifact_type missing path=%s", path)
            else:
                raise SchemaError(
                    f"[ARTIFACT_TYPE_MISMATCH] Некорректный тип артефакта: {artifact_type!r}. "
                    f"Ожидалось одно из: {sorted(allowed_types)}."
                )

    sv = pkg.get("schema_version")
    if sv is None:
        if not allow_missing_schema:
            raise SchemaError("[SCHEMA_MISSING] В модели отсутствует schema_version.")
        msg = (
            "⚠ Модель сохранена в старом формате (schema_version отсутствует). "
            "Часть функций может быть недоступна."
        )
        if log_fn:
            log_fn(msg)
        if logger:
            logger.warning("legacy schema model loaded: %s", path)
    elif not isinstance(sv, int):
        raise SchemaError(
            f"[SCHEMA_TYPE_INVALID] Неверный тип schema_version={sv!r} (ожидается int или None)."
        )
    elif sv > supported_schema_version:
        raise SchemaError(
            f"[SCHEMA_UNSUPPORTED] Модель создана более новой версией приложения "
            f"(schema_version={sv}, поддерживается ≤{supported_schema_version})."
        )

    for key in required_keys or ():
        if key not in pkg:
            raise ModelLoadError(f"В модели отсутствует обязательный ключ: {key}")
    for key, expected_type in (required_key_types or {}).items():
        if key in pkg and not isinstance(pkg[key], expected_type):
            expected_name = (
                ", ".join(t.__name__ for t in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise ModelLoadError(
                f"Неверный тип ключа '{key}': "
                f"{type(pkg[key]).__name__} (ожидалось: {expected_name})."
            )

    return pkg
