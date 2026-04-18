# -*- coding: utf-8 -*-
"""
Построение текстового признака для TF-IDF:
- build_feature_text        — сборка секций с весами
- choose_row_profile_weights — выбор весов на основе заполненности полей строки
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

from constants import PRESET_WEIGHTS, SECTION_PREFIX


# ---------------------------------------------------------------------------
# Ограничение глубины диалога
# ---------------------------------------------------------------------------
# Суть проблемы — в первых репликах клиента.
# Результат/развязка (согласился / отказался / перевёл) — в последних.
# Берём обе части: голову + хвост, разделённые маркером «…» если они не
# перекрываются. Промежуток выбрасывается — он содержит рутинные Q&A.
MAX_CLIENT_LINES_HEAD   = 5   # первые N реплик клиента  (суть обращения)
MAX_CLIENT_LINES_TAIL   = 4   # последние N реплик клиента (итог)
MAX_OPERATOR_LINES_HEAD = 3   # первые N реплик оператора (диагностика)
MAX_OPERATOR_LINES_TAIL = 3   # последние N реплик оператора (решение)

# Компилируем один раз на уровне модуля (не внутри функции)
_HTAIL_TOK_RE = re.compile(r'[А-ЯЁа-яёA-Za-z]{3,}')


def _head_tail_lines(text: str, n_head: int, n_tail: int) -> str:
    """Возвращает первые n_head строк + n_tail наиболее информативных из хвоста.

    Алгоритм:
      1. Голова (первые n_head реплик) — сохраняется всегда: содержит суть обращения.
      2. Из оставшихся реплик отбираются n_tail с наибольшим локальным IDF-score:
         score(line) = Σ (1 / freq_in_dialog(token)) для уникальных токенов строки.
         Строки с редкими (в рамках диалога) словами получают высокий score.
      3. Между голова и хвостом вставляется «…».

    Если строк ≤ n_head + n_tail — возвращаем все без дублирования.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    total = len(lines)
    if total <= n_head + n_tail:
        return "\n".join(lines)

    head = lines[:n_head]
    rest = lines[n_head:]

    # Локальный IDF только по хвостовым кандидатам (rest), а не по всему диалогу.
    # Если считать по lines (включая head), то слова из head занижают IDF-score
    # хвостовых строк с теми же словами — выбор становится смещённым.
    tok_doc_freq: Dict[str, int] = {}
    for line in rest:
        for tok in set(_HTAIL_TOK_RE.findall(line.lower())):
            tok_doc_freq[tok] = tok_doc_freq.get(tok, 0) + 1

    def _score(line: str) -> float:
        toks = set(_HTAIL_TOK_RE.findall(line.lower()))
        if not toks:
            return 0.0
        N = len(rest)
        return sum(math.log1p(N / tok_doc_freq.get(t, 1)) for t in toks)

    # Сортируем «хвост» по убыванию score, берём top n_tail
    scored = sorted(enumerate(rest), key=lambda iv: _score(iv[1]), reverse=True)
    chosen_idx = sorted(idx for idx, _ in scored[:n_tail])
    tail = [rest[i] for i in chosen_idx]

    return "\n".join(head) + "\n…\n" + "\n".join(tail)


# ---------------------------------------------------------------------------
# Сборка feature-текста
# ---------------------------------------------------------------------------

def build_feature_text(
    channel: str,
    desc: str,
    client_text: str,
    operator_text: str,
    summary: str,
    ans_short: str,
    ans_full: str,
    weights: Dict[str, int],
    normalize_entities: bool = False,
) -> str:
    """
    Собирает итоговый текстовый признак с префиксами секций и весами (0..3).
    Вес реализован как повторение секции — совместимо с TF-IDF.

    Для client_text и operator_text берутся только первые MAX_CLIENT_LINES /
    MAX_OPERATOR_LINES реплик: суть проблемы всегда в начале диалога.
    """
    if normalize_entities:
        from entity_normalizer import normalize_entities as _ne
        desc         = _ne(desc)
        client_text  = _ne(client_text)
        operator_text = _ne(operator_text)
        summary      = _ne(summary)
        ans_short    = _ne(ans_short)
        ans_full     = _ne(ans_full)

    parts: List[str] = [f"{SECTION_PREFIX['CHANNEL']}\n{channel}\n"]

    def add(tag: str, txt: str, w: int) -> None:
        if not txt or w <= 0:
            return
        block = f"{SECTION_PREFIX[tag]}\n{txt}\n"
        parts.append(("\n" + block) * int(w))

    add("DESC",         desc,
        weights.get("w_desc", 0))
    add("CLIENT",       _head_tail_lines(client_text,
                                         MAX_CLIENT_LINES_HEAD,
                                         MAX_CLIENT_LINES_TAIL),
        weights.get("w_client", 0))
    add("OPERATOR",     _head_tail_lines(operator_text,
                                         MAX_OPERATOR_LINES_HEAD,
                                         MAX_OPERATOR_LINES_TAIL),
        weights.get("w_operator", 0))
    add("SUMMARY",      summary,
        weights.get("w_summary", 0))
    add("ANSWER_SHORT", ans_short,
        weights.get("w_answer_short", 0))
    add("ANSWER_FULL",  ans_full,
        weights.get("w_answer_full", 0))

    x = "\n".join([p for p in parts if p]).strip()
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x


# ---------------------------------------------------------------------------
# Выбор пресета весов для строки
# ---------------------------------------------------------------------------

def choose_row_profile_weights(
    base: Dict[str, int],
    auto_profile: str,
    has_desc: bool,
    has_dialog: bool,
    roles_found: bool,
    has_summary: bool,
    has_ans_s: bool,
    has_ans_f: bool,
) -> Dict[str, int]:
    """
    Адаптирует веса под конкретную строку в зависимости от заполненности полей.

    Логика выбора пресета:
        • dialog + summary, без ответов → no_answers (summary усилена до 4, клиент/оператор повышены)
        • dialog + summary              → balanced   (все источники есть, оптимальный баланс)
        • только dialog                 → client     (клиент — главный источник)
        • только summary                → summary    (суммаризация — главный источник)
        • только ответы                 → answers    (ответ банка — основной сигнал)
        • иначе                         → balanced   (нет явной специфики)

    Режимы auto_profile:
        off    — только базовые веса пользователя
        smart  — берёт max(base, preset) для каждого поля
        strict — полностью заменяет base на подходящий preset
    """
    w = dict(base)

    # Всегда обнуляем пустые поля
    if not has_desc:    w["w_desc"] = 0
    if not has_summary: w["w_summary"] = 0
    if not has_ans_s:   w["w_answer_short"] = 0
    if not has_ans_f:   w["w_answer_full"] = 0

    if auto_profile == "off":
        return w

    # Выбор пресета с учётом набора доступных источников
    has_roles = has_dialog and roles_found

    if has_roles and has_summary and not has_ans_s and not has_ans_f and not has_desc:
        # Консультации: диалог + суммаризация, нет desc и ответов — специализированный пресет
        preset = "consultation"
    elif has_roles and has_summary and not has_ans_s and not has_ans_f:
        # Диалог + суммаризация, ответы не выбраны → суммаризация заменяет ответы
        preset = "no_answers"
    elif has_roles and has_summary:
        # Все ключевые источники присутствуют → сбалансированный пресет
        preset = "balanced"
    elif has_roles:
        # Есть диалог, нет суммаризации → фокус на клиенте
        preset = "client"
    elif has_summary:
        # Нет живого диалога → суммаризация + desc + ответ
        preset = "summary"
    elif has_ans_s or has_ans_f:
        # Нет диалога и суммаризации → опора на ответы
        preset = "answers"
    else:
        preset = "balanced"

    pw = PRESET_WEIGHTS[preset]
    if auto_profile == "smart":
        out = {k: max(int(w.get(k, 0)), int(pw.get(k, 0))) for k in pw}
    else:  # strict
        out = dict(pw)

    # Повторно обнуляем пустые поля после применения пресета
    if not has_desc:    out["w_desc"] = 0
    if not has_summary: out["w_summary"] = 0
    if not has_ans_s:   out["w_answer_short"] = 0
    if not has_ans_f:   out["w_answer_full"] = 0

    return out
