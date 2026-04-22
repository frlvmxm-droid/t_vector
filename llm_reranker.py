"""
llm_reranker — LLM-повторное ранжирование top-K кандидатов для неуверенных строк.

Идея: для строк с уверенностью в диапазоне [low_thr, high_thr] (обычно 0.50–0.70)
классификатор «колеблется» между близкими классами. LLM с few-shot контекстом из
этих кандидатов (2–3 класса + их типичные примеры из обучения) часто выбирает
правильный класс точнее, чем калиброванная LinearSVC-вероятность.

Публичный API:
    rerank_top_k(texts, top_k_candidates, provider, model, api_key, ...) -> List[str]

Требования:
    • `llm_client.LLMClient.complete_text` доступен.
    • `class_examples`: Dict[label -> List[str]] — несколько (2–3) типичных текстов
      на класс, взятых из обучающей выборки (найдены ближайшими к центроиду).

Провайдер sandboxed; при ошибке/timeouts возвращает argmax исходного предсказания.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app_logger import get_logger
from exceptions import FeatureBuildError
from llm_client import LLMClient

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates (RU, банковский домен)
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = (
    "Ты — аналитик клиентских обращений в банке. Твоя задача — выбрать "
    "из предложенных классов ОДИН, наиболее точно описывающий обращение. "
    "Отвечай ТОЛЬКО названием класса — без пояснений, кавычек и префиксов. "
    "Если ни один класс не подходит — верни первый предложенный класс."
)


def _build_rerank_user(
    text: str,
    candidates: Sequence[str],
    class_examples: dict[str, Sequence[str]] | None,
    max_examples_per_class: int = 2,
    max_text_chars: int = 800,
) -> str:
    """Формирует user-часть промпта с few-shot примерами по кандидатам."""
    _txt = str(text or "")[:max_text_chars]
    lines: list[str] = [f"ОБРАЩЕНИЕ:\n{_txt}", "", "КЛАССЫ-КАНДИДАТЫ:"]
    for c in candidates:
        _exs = []
        if class_examples and c in class_examples:
            _exs = [str(e)[:220] for e in list(class_examples[c])[:max_examples_per_class]]
        lines.append(f"• {c}")
        for ex in _exs:
            lines.append(f"    — {ex}")
    lines.append("")
    lines.append("Выбери ровно один класс из списка выше. Напиши только название класса.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_RESPONSE_PREFIX_RE = re.compile(
    r"^\s*(?:класс|category|class|ответ|answer|result|итог)\s*[:\-–—]\s*",
    re.IGNORECASE,
)
_RESPONSE_TRIM_RE = re.compile(r'^[\s"«\'`*\-•]+|[\s"»\'`*\-•.]+$')


def _parse_rerank_response(
    response: str,
    candidates: Sequence[str],
    fallback: str,
) -> str:
    """Мапит свободный ответ LLM к одному из допустимых кандидатов."""
    r = (response or "").strip()
    if not r:
        return fallback
    # Берём только первую непустую строку (LLM иногда возвращает "класс\n\nобоснование…")
    for line in r.splitlines():
        line = line.strip()
        if line:
            r = line
            break
    # Снимаем типичные префиксы "Класс:", "Ответ -" и т.п.
    r = _RESPONSE_PREFIX_RE.sub("", r)
    # Снимаем кавычки, точки, маркеры
    r = _RESPONSE_TRIM_RE.sub("", r).strip()
    if not r:
        return fallback
    # 1. Точное совпадение (регистронезависимо)
    _r_lower = r.lower()
    for c in candidates:
        if c.lower() == _r_lower:
            return c
    # 2. Подстрока (класс внутри ответа)
    for c in sorted(candidates, key=len, reverse=True):
        if c.lower() in _r_lower:
            return c
    # 3. Fallback
    return fallback


# ---------------------------------------------------------------------------
# Persistent disk cache (sha256(text,candidates,model,provider) → label)
# ---------------------------------------------------------------------------

_CACHE_DIR = pathlib.Path.home() / ".classification_tool" / "llm_rerank_cache"
_CACHE_LOCK = threading.Lock()


def _cache_key(
    text: str,
    candidates: Sequence[str],
    provider: str,
    model: str,
) -> str:
    h = hashlib.sha256()
    h.update(provider.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(text or "").encode("utf-8"))
    h.update(b"\x00")
    for c in candidates:
        h.update(str(c).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def _cache_path(key: str) -> pathlib.Path:
    return _CACHE_DIR / key[:2] / f"{key}.json"


def _cache_read(key: str) -> str | None:
    p = _cache_path(key)
    try:
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                label = json.load(f).get("label")
                return str(label) if isinstance(label, str) else None
    except (OSError, ValueError):
        return None
    return None


def _cache_write(key: str, label: str) -> None:
    p = _cache_path(key)
    try:
        with _CACHE_LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump({"label": label}, f, ensure_ascii=False)
            os.replace(tmp, p)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_MAX_WORKERS = 8


def rerank_top_k(
    texts: Sequence[str],
    top_candidates: Sequence[Sequence[str]],
    argmax_labels: Sequence[str],
    *,
    provider: str,
    model: str,
    api_key: str,
    class_examples: dict[str, Sequence[str]] | None = None,
    timeout_sec: float = 20.0,
    max_retries: int = 3,
    temperature: float | None = 0.2,
    max_workers: int | None = None,
    use_disk_cache: bool = False,
    log_fn: Any | None = None,
) -> list[str]:
    """Для каждой строки отдельно просит LLM выбрать класс из top-K.

    Возвращает список меток длины len(texts). Если LLM недоступен для конкретной
    строки — ставит argmax_labels[i].

    Параметры:
      texts           — входные тексты
      top_candidates  — для каждой строки список кандидатов (2–5 классов, уже
                        отсортированный по убыванию вероятности)
      argmax_labels   — fallback-метки (argmax-вывод модели)
      class_examples  — опционально: типичные примеры для каждого класса
      temperature     — температура семплирования. 0.2 по умолчанию даёт
                        детерминизм при сохранении гибкости; None — использовать
                        дефолт провайдера.
      max_workers     — число параллельных LLM-вызовов. None → 8.
      use_disk_cache  — использовать персистентный кэш ~/.classification_tool/
                        llm_rerank_cache/ (по SHA-256 от текста+кандидатов+модели).
    """
    if not (len(texts) == len(top_candidates) == len(argmax_labels)):
        raise ValueError("texts, top_candidates, argmax_labels must have equal length")

    n = len(texts)
    out: list[str] = [""] * n
    # Per-row diagnostic slots; sum after the parallel pass.
    _n_ok = 0
    _n_fail = 0
    _n_skip = 0
    _n_mem_cache = 0
    _n_disk_cache = 0
    _counters_lock = threading.Lock()

    # In-batch кэш: дубликаты (текст + набор кандидатов) не уходят повторно в LLM.
    mem_cache: dict[tuple[str, tuple[str, ...]], str] = {}
    mem_cache_lock = threading.Lock()

    # Pre-processing: compute per-row cache lookups and partition into «needs LLM» vs «done».
    # Dedupe within the batch: rows sharing (text, candidates) generate ONE work item.
    mem_groups: dict[tuple[str, tuple[str, ...]], list[tuple[int, str]]] = {}
    mem_to_diskkey: dict[tuple[str, tuple[str, ...]], str] = {}
    mem_to_cands: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    mem_to_text: dict[tuple[str, tuple[str, ...]], str] = {}

    for i, (txt, cands, fb) in enumerate(zip(texts, top_candidates, argmax_labels)):
        cands_list = [str(c) for c in cands]
        if len(cands_list) < 2:
            out[i] = str(fb)
            _n_skip += 1
            continue

        _txt = str(txt or "")
        mem_key = (_txt, tuple(cands_list))

        if mem_key not in mem_groups:
            mem_groups[mem_key] = []
            mem_to_cands[mem_key] = cands_list
            mem_to_text[mem_key] = _txt

        mem_groups[mem_key].append((i, str(fb)))

    # Resolve each unique (text, candidates) key from caches first;
    # leftover keys become work_items.
    work_items: list[tuple[tuple[str, tuple[str, ...]], str, list[str], str, str]] = []
    for mem_key, members in mem_groups.items():
        _txt = mem_to_text[mem_key]
        cands_list = mem_to_cands[mem_key]
        _fb = members[0][1]

        disk_key = _cache_key(_txt, cands_list, provider, model) if use_disk_cache else ""
        mem_to_diskkey[mem_key] = disk_key
        resolved: str | None = None

        if use_disk_cache:
            cached = _cache_read(disk_key)
            if cached is not None and cached in cands_list:
                resolved = cached
                _n_disk_cache += len(members)

        if resolved is not None:
            mem_cache[mem_key] = resolved
            for row_i, _ in members:
                out[row_i] = resolved
            continue

        work_items.append((mem_key, _txt, cands_list, _fb, disk_key))

    def _call_llm(
        job: tuple[tuple[str, tuple[str, ...]], str, list[str], str, str],
    ) -> tuple[tuple[str, tuple[str, ...]], str, bool]:
        mem_key, _txt, cands_list, fb, disk_key = job
        user_prompt = _build_rerank_user(_txt, cands_list, class_examples)
        try:
            resp = LLMClient.complete_text(
                provider=provider,
                model=model,
                api_key=api_key,
                system_prompt=_RERANK_SYSTEM,
                user_prompt=user_prompt,
                max_tokens=32,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                temperature=temperature,
            )
            chosen = _parse_rerank_response(resp, cands_list, fallback=fb)
            if use_disk_cache and disk_key:
                _cache_write(disk_key, chosen)
            return mem_key, chosen, True
        except FeatureBuildError as e:
            _log.warning("llm rerank failed for key %s: %s", mem_key[0][:40], e)
            return mem_key, fb, False
        except Exception as e:  # noqa: BLE001 — per-group safety net; one bad group must not abort the whole batch
            _log.warning("llm rerank unexpected error for key %s: %s", mem_key[0][:40], e)
            return mem_key, fb, False

    # Track fan-out: members who did not hit disk cache but share a key
    if work_items:
        workers = max(1, int(max_workers if max_workers is not None else _DEFAULT_MAX_WORKERS))
        workers = min(workers, len(work_items))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mem_key, label, ok in ex.map(_call_llm, work_items):
                members = mem_groups[mem_key]
                # Only the first row in each group counts as «fresh LLM call»;
                # the rest are in-batch cache hits.
                first_row = members[0][0]
                with mem_cache_lock:
                    mem_cache[mem_key] = label
                for row_i, _fb in members:
                    out[row_i] = label
                with _counters_lock:
                    if ok:
                        _n_ok += 1
                    else:
                        _n_fail += 1
                    if len(members) > 1:
                        _n_mem_cache += len(members) - 1
                del first_row

    if log_fn:
        log_fn(
            f"[LLM-ре-ранк] rerank'нуто={_n_ok} | из disk-кэша={_n_disk_cache} | "
            f"из in-batch кэша={_n_mem_cache} | ошибок={_n_fail} | "
            f"пропущено={_n_skip} (всего строк={n})"
        )
    return out


def build_class_examples_from_training(
    X_texts: Sequence[str],
    y_labels: Sequence[str],
    n_per_class: int = 3,
    max_chars: int = 260,
) -> dict[str, list[str]]:
    """Отбирает до n_per_class текстов на класс (первые encountered).

    ВНИМАНИЕ: упрощённый отбор по порядку появления, НЕ по центроидности.
    Для продакшен-использования предпочитайте
    `ml_diagnostics.find_cluster_representative_texts(labels=y_labels, ...)` —
    качество few-shot у LLM заметно выше с репрезентативными примерами.
    """
    _log.info(
        "build_class_examples_from_training: упрощённый отбор — "
        "для лучшего качества few-shot используйте "
        "ml_diagnostics.find_cluster_representative_texts"
    )
    out: dict[str, list[str]] = {}
    for x, y in zip(X_texts, y_labels):
        _y = str(y)
        if _y not in out:
            out[_y] = []
        if len(out[_y]) < n_per_class:
            _t = str(x or "")[:max_chars]
            if _t:
                out[_y].append(_t)
    return out
