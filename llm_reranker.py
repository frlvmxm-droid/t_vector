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

import re
from collections.abc import Sequence
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

def _parse_rerank_response(
    response: str,
    candidates: Sequence[str],
    fallback: str,
) -> str:
    """Мапит свободный ответ LLM к одному из допустимых кандидатов."""
    r = (response or "").strip()
    if not r:
        return fallback
    # Снимаем кавычки, точки, маркеры
    r = re.sub(r'^[\s"«\'`*\-•]+|[\s"»\'`*\-•.]+$', "", r).strip()
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
# Public API
# ---------------------------------------------------------------------------

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
    """
    if not (len(texts) == len(top_candidates) == len(argmax_labels)):
        raise ValueError("texts, top_candidates, argmax_labels must have equal length")

    out: list[str] = []
    _n_ok = 0
    _n_fail = 0
    _n_skip = 0

    for i, (txt, cands, fb) in enumerate(zip(texts, top_candidates, argmax_labels)):
        cands_list = [str(c) for c in cands]
        if len(cands_list) < 2:
            # Нечего пересортировывать
            out.append(str(fb))
            _n_skip += 1
            continue

        user_prompt = _build_rerank_user(txt, cands_list, class_examples)
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
            chosen = _parse_rerank_response(resp, cands_list, fallback=str(fb))
            out.append(chosen)
            _n_ok += 1
        except FeatureBuildError as e:
            _log.warning("llm rerank failed on row %d: %s", i, e)
            out.append(str(fb))
            _n_fail += 1
        except Exception as e:  # noqa: BLE001 — per-row safety net; one bad row must not abort the whole batch (fallback to argmax)
            _log.warning("llm rerank unexpected error on row %d: %s", i, e)
            out.append(str(fb))
            _n_fail += 1

    if log_fn:
        log_fn(
            f"[LLM-ре-ранк] rerank'нуто={_n_ok} | ошибок={_n_fail} | пропущено={_n_skip} "
            f"(всего строк={len(texts)})"
        )
    return out


def build_class_examples_from_training(
    X_texts: Sequence[str],
    y_labels: Sequence[str],
    n_per_class: int = 3,
    max_chars: int = 260,
) -> dict[str, list[str]]:
    """Отбирает до n_per_class текстов на класс (первые encountered).

    Упрощённый отбор — не ищет ближайшие к центроиду. Для качественного отбора
    используй ml_diagnostics.find_cluster_representative_texts с labels=y_labels
    и предобученными векторами.
    """
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
