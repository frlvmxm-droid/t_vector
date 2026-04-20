"""Headless port of `app_cluster._cluster_step_llm_naming`.

Pure-Python helper that turns a (labels, texts) pair into ``{cluster_id:
human_name}`` by asking an LLM to summarise each cluster's representative
texts. The desktop closure at ``app_cluster.py:2076`` does the same, but
mixes UI-thread hooks (``self.after``, ``self.log_cluster``) into the
loop. This module strips them out and accepts an optional
``log_cb(message: str)`` instead.

Offline mode: when ``BRT_LLM_PROVIDER=offline`` the call returns the
deterministic stub from ``LLMClient.complete_text`` (ADR-0004), so unit
tests run without network access.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from app_logger import get_logger

_log = get_logger(__name__)

LogCB = Callable[[str], None]

# Mirrors the prompt baked into `_cluster_step_llm_naming`. Few-shot
# examples in Russian + a temperature=0.2 default give reproducible,
# concrete cluster titles ("Списание комиссии за обслуживание") instead
# of generic ones ("Вопрос по карте").
DEFAULT_SYSTEM_PROMPT = (
    "Ты — аналитик клиентских обращений банка. "
    "Сформулируй краткое название (3–6 слов) на русском языке, "
    "отражающее КОНКРЕТНУЮ причину или проблему обращения клиента. "
    "Используй глагол или существительное-проблему, "
    "избегай общих слов («вопрос», «проблема» без уточнения). "
    "Отвечай только названием, без пояснений и без кавычек.\n\n"
    "Примеры корректных названий:\n"
    "• «Задержка зачисления платежа»\n"
    "• «Блокировка карты при оплате»\n"
    "• «Списание комиссии за обслуживание»\n"
    "• «Оспаривание операции по карте»\n"
    "• «Невозможно войти в мобильное приложение»\n\n"
    "Плохие (слишком общие) названия: "
    "«Вопрос по карте», «Обращение клиента», «Проблема с банком»."
)


def _representative_texts(
    cluster_labels: Any,
    texts: Sequence[str],
    vectors: Any | None,
    rep_k: int,
) -> dict[int, list[str]]:
    """Top-K texts nearest to each cluster centroid (cosine).

    Falls back to the first ``rep_k`` rows per cluster when ``vectors`` is
    None or `find_cluster_representative_texts` raises (e.g. degenerate
    cluster). Mirrors the desktop fallback at ``app_cluster.py:2089``.
    """
    import numpy as _np

    labels_arr = _np.asarray(cluster_labels)
    if vectors is not None:
        try:
            from ml_diagnostics import find_cluster_representative_texts

            return find_cluster_representative_texts(
                list(texts), labels_arr, vectors, n_top=rep_k,
            )
        except Exception as exc:  # noqa: BLE001 — fallback is fine
            _log.debug("find_cluster_representative_texts failed: %s", exc)

    out: dict[int, list[str]] = {}
    for cid in sorted(set(int(c) for c in labels_arr.tolist()) - {-1}):
        idx = [j for j, c in enumerate(labels_arr) if int(c) == cid][:rep_k]
        out[cid] = [str(texts[j])[:300] for j in idx if j < len(texts)]
    return out


def _build_user_prompt(keywords: str, examples: Sequence[str]) -> str:
    return (
        f"Ключевые признаки: {keywords}\n\n"
        "Типичные обращения:\n"
        + "\n".join(f"• {ex}" for ex in examples)
    )


def name_clusters_with_llm(
    cluster_labels: Any,
    texts: Sequence[str],
    *,
    keywords: dict[int, str] | Sequence[str],
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: str = "",
    vectors: Any | None = None,
    rep_k: int = 5,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 64,
    temperature: float = 0.2,
    complete_fn: Callable[..., str] | None = None,
    log_cb: LogCB | None = None,
) -> dict[int, str]:
    """Return ``{cluster_id: short_name}`` by asking ``provider/model``.

    ``keywords`` may be either a dict ``{cluster_id: "kw1, kw2, ..."}`` or
    a sequence indexed by cluster id (matches ``kw[ci]`` in the desktop
    closure). Clusters with empty keyword strings are skipped.

    ``complete_fn`` is dependency-injected for tests; defaults to
    ``LLMClient.complete_text`` (which honours ``BRT_LLM_PROVIDER=offline``).

    ``vectors`` is the clustering matrix used to find representative texts
    via ``ml_diagnostics.find_cluster_representative_texts``. If omitted,
    the first ``rep_k`` rows per cluster are used as fallback.
    """
    if complete_fn is None:
        from llm_client import LLMClient

        complete_fn = LLMClient.complete_text

    rep_texts = _representative_texts(cluster_labels, texts, vectors, rep_k)

    if isinstance(keywords, dict):
        kw_iter = keywords.items()
    else:
        kw_iter = list(enumerate(keywords))

    name_map: dict[int, str] = {}
    for cid, kw_str in kw_iter:
        if not kw_str:
            continue
        cid_i = int(cid)
        examples = rep_texts.get(cid_i, [])
        user_msg = _build_user_prompt(str(kw_str), examples)
        try:
            name = complete_fn(
                provider=provider,
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_msg,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 — per-row failure is non-fatal
            if log_cb is not None:
                log_cb(f"⚠️ LLM кластер {cid_i}: {exc}")
            continue
        clean = (name or "").strip().strip('"').strip("«»").strip()
        if clean:
            name_map[cid_i] = clean
    if log_cb is not None:
        log_cb(f"LLM-нейминг: названы {len(name_map)} кластеров ✅")
    return name_map
