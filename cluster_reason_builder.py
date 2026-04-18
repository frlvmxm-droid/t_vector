# -*- coding: utf-8 -*-
"""Эвристики описания причин кластера."""
from __future__ import annotations

from typing import List


class ClusterReasonBuilder:
    """Эвристический генератор описания причин кластера (fallback без LLM)."""

    @staticmethod
    def build_reason(cluster_name: str, keywords: str, examples: List[str]) -> str:
        tokens = []
        for raw in (keywords or "").replace(";", ",").split(","):
            w = raw.strip().lower()
            if w and w not in tokens:
                tokens.append(w)
        top = tokens[:5]

        topic_rules = [
            (("оплат", "комисс", "списан"), "проблемами оплаты и списаний"),
            (("перевод", "зачисл", "реквиз"), "переводами и зачислением средств"),
            (("бонус", "кэшбек", "балл"), "бонусами/кэшбеком и начислениями"),
            (("пенси",), "вопросами по пенсионным выплатам"),
            (("карт", "пин", "банкомат"), "операциями по банковским картам"),
            (("кредит", "ставк", "платеж"), "кредитными продуктами и платежами"),
            (("смс", "код", "подтвержд"), "подтверждением операций и SMS-кодами"),
        ]
        joined = " ".join(top)
        theme = ""
        for needles, label in topic_rules:
            if any(n in joined for n in needles):
                theme = label
                break
        if not theme:
            theme = "похожими причинами обращений клиентов"

        title_part = f"«{cluster_name.strip()}»" if cluster_name else "этого кластера"
        kw_part = ", ".join(top[:3]) if top else "ключевые слова не выделены"
        ex_hint = f" Часто встречаются формулировки про: {examples[0][:90]}." if examples else ""
        return (
            f"Обращения {title_part} в основном связаны с {theme}. "
            f"Наиболее характерные маркеры: {kw_part}.{ex_hint}"
        ).strip()
