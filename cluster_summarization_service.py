"""Headless port of `app_cluster._cluster_step_t5`.

Pure-Python wrapper around ``T5RussianSummarizer`` that turns
``{cluster_id: [texts...]}`` into ``{cluster_id: summary}``. The desktop
closure at ``app_cluster.py:2413`` does the same but with UI-thread
hooks (``self.after``, ``ui_prog``); this module accepts an optional
``progress_cb(frac, msg)`` instead.

When ``transformers`` / ``torch`` are missing, ``T5RussianSummarizer``
raises ``ImportError`` from its lazy ``load()``; we catch it and fall
back to ``"(disabled — transformers not installed)"`` so the call site
stays safe in CI.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from app_logger import get_logger

_log = get_logger(__name__)

ProgressCB = Callable[[float, str], None]
LogCB = Callable[[str], None]


def _join_texts(texts: Sequence[str], max_chars: int = 3000) -> str:
    """Concatenate cluster texts into one bullet list, truncated for T5."""
    parts: list[str] = []
    for t in texts:
        s = str(t).strip()
        if s:
            parts.append(f"• {s}")
    joined = "\n".join(parts)
    return joined[:max_chars]


def summarize_clusters_with_t5(
    texts_by_cluster: Mapping[int, Sequence[str]],
    *,
    model_name: str = "UrukHan/t5-russian-summarization",
    max_input_length: int = 512,
    max_target_length: int = 128,
    batch_size: int = 4,
    device: str = "auto",
    progress_cb: ProgressCB | None = None,
    log_cb: LogCB | None = None,
) -> dict[int, str]:
    """Summarise each cluster with T5; return ``{cluster_id: summary}``.

    Returns an empty dict if ``transformers`` is unavailable. Per-cluster
    failures are logged via ``log_cb`` and the cluster is skipped — the
    overall summary still completes.
    """
    if not texts_by_cluster:
        return {}

    try:
        from t5_summarizer import T5RussianSummarizer
    except ImportError as exc:
        if log_cb is not None:
            log_cb(f"⚠️ T5 пропущен — {exc}")
        else:
            _log.info("T5 unavailable: %s", exc)
        return {}

    summarizer = T5RussianSummarizer(
        model_name=model_name,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        batch_size=batch_size,
        device=device,
        log_cb=log_cb,
        progress_cb=None,  # we drive a per-cluster progress instead
    )
    try:
        summarizer.load()
    except ImportError as exc:
        # transformers/torch missing — same as the import-time guard above
        if log_cb is not None:
            log_cb(f"⚠️ T5 пропущен — {exc}")
        return {}
    except Exception as exc:  # noqa: BLE001 — model download / OOM
        if log_cb is not None:
            log_cb(f"⚠️ T5 ошибка загрузки: {exc}")
        return {}

    cluster_ids = sorted(int(cid) for cid in texts_by_cluster.keys() if int(cid) >= 0)
    inputs = [_join_texts(texts_by_cluster[cid]) for cid in cluster_ids]
    if not inputs:
        return {}

    out: dict[int, str] = {}
    try:
        summaries = summarizer.summarize(inputs)
    except Exception as exc:  # noqa: BLE001 — fall back to per-cluster
        if log_cb is not None:
            log_cb(f"⚠️ T5 batch-режим упал ({exc}), fallback по одному кластеру")
        summaries = []
        for i, text in enumerate(inputs):
            try:
                summaries.append(summarizer.summarize([text])[0])
            except Exception as exc2:  # noqa: BLE001 — skip cluster
                if log_cb is not None:
                    log_cb(f"⚠️ T5 кластер {cluster_ids[i]}: {exc2}")
                summaries.append("")

    for i, cid in enumerate(cluster_ids):
        summary = (summaries[i] if i < len(summaries) else "").strip()
        if summary:
            out[cid] = summary
        if progress_cb is not None:
            progress_cb((i + 1) / len(cluster_ids), f"T5 cluster {cid}")
    if log_cb is not None:
        log_cb(f"T5: суммаризировано {len(out)} кластеров ✅")
    return out
