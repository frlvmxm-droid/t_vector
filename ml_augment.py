# -*- coding: utf-8 -*-
"""
ml_augment — LLM-based data augmentation for rare training classes.

Generates paraphrases of existing examples using an LLM to augment
under-represented classes before training.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


_SYS_PROMPT = (
    "Ты — специалист по аугментации текстовых данных для банковских чат-ботов. "
    "Сгенерируй перефразировки клиентского обращения. "
    "Каждая перефразировка — на отдельной строке, начинается с «- ». "
    "Сохраняй смысл и тему, но меняй формулировку, порядок слов, синонимы. "
    "Используй разговорный стиль, возможны опечатки и сокращения. "
    "Не нумеруй строки, не добавляй пояснений."
)


def _parse_lines(text: str) -> List[str]:
    """Extract paraphrase lines from LLM response."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-•*").strip()
        if line:
            lines.append(line)
    return lines


def generate_class_paraphrases(
    examples: List[str],
    n_paraphrases: int,
    llm_complete_fn: Callable[..., str],
    provider: str,
    model: str,
    api_key: str,
    cancel_event: Optional[Any] = None,
) -> List[str]:
    """Generate n_paraphrases paraphrases for the given examples using an LLM.

    Returns a flat list of generated texts (may be fewer than requested if
    the LLM returns shorter output or on error).
    """
    if not examples:
        return []

    # Use up to 3 examples as seeds to give the LLM variety
    seeds = examples[:3]
    user_prompt = (
        f"Сгенерируй {n_paraphrases} перефразировок для каждого из следующих обращений.\n\n"
        + "\n".join(f"Пример {i+1}: {ex[:200]}" for i, ex in enumerate(seeds))
    )

    try:
        if cancel_event is not None and cancel_event.is_set():
            return []
        response = llm_complete_fn(
            provider=provider,
            model=model,
            api_key=api_key,
            system_prompt=_SYS_PROMPT,
            user_prompt=user_prompt,
            max_tokens=400,
        )
        return _parse_lines(response)
    except Exception:
        return []


def augment_rare_classes(
    X: List[str],
    y: List[str],
    min_samples_threshold: int,
    n_paraphrases: int,
    llm_complete_fn: Callable[..., str],
    provider: str,
    model: str,
    api_key: str,
    log_fn: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[Any] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Augment classes with fewer than min_samples_threshold examples.

    For each rare class, calls the LLM to generate paraphrases of existing
    examples and appends them to X, y.

    Returns:
        X_aug: original + generated texts
        y_aug: original + generated labels
        report: {"classes_augmented": int, "rows_added": int, "skipped": List[str]}
    """
    from collections import Counter

    counts = Counter(y)
    rare_classes = sorted(
        [cls for cls, cnt in counts.items() if cnt < min_samples_threshold]
    )

    if not rare_classes:
        return list(X), list(y), {"classes_augmented": 0, "rows_added": 0, "skipped": []}

    X_aug = list(X)
    y_aug = list(y)
    rows_added = 0
    skipped: List[str] = []

    # Build index: class → list of texts
    class_texts: Dict[str, List[str]] = {}
    for text, label in zip(X, y):
        class_texts.setdefault(label, []).append(text)

    for cls in rare_classes:
        if cancel_event is not None and cancel_event.is_set():
            break

        existing = class_texts.get(cls, [])
        if not existing:
            skipped.append(cls)
            continue

        if log_fn:
            log_fn(
                f"  LLM-аугментация: «{cls}» ({len(existing)} примеров → "
                f"генерирую {n_paraphrases} перефразировок)"
            )

        generated = generate_class_paraphrases(
            examples=existing,
            n_paraphrases=n_paraphrases,
            llm_complete_fn=llm_complete_fn,
            provider=provider,
            model=model,
            api_key=api_key,
            cancel_event=cancel_event,
        )

        if not generated:
            skipped.append(cls)
            if log_fn:
                log_fn(f"  ⚠ LLM-аугментация: «{cls}» — не удалось получить ответ, пропускаем")
            continue

        for text in generated:
            X_aug.append(text)
            y_aug.append(cls)
            rows_added += 1

        if log_fn:
            log_fn(f"  ✅ LLM-аугментация: «{cls}» +{len(generated)} строк")

    report = {
        "classes_augmented": len(rare_classes) - len(skipped),
        "rows_added": rows_added,
        "skipped": skipped,
    }
    return X_aug, y_aug, report
