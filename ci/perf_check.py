#!/usr/bin/env python3
"""Nightly perf-regression gate.

Запускает несколько микро-бенчмарков, сравнивает медианы с
``ci/perf_baseline.json`` и выходит с кодом 1, если любая метрика
ухудшилась более чем на ``TOLERANCE`` (по умолчанию +5 %).

Дизайн:
  * Один процесс, без CLI-флагов — гейт.
  * Каждый бенчмарк — N≥7 измерений; берём median, не mean (устойчиво
    к одиночным GC/CI-spikes).
  * Метрики только pure-CPU (без I/O, без LLM, без сетевых вызовов) —
    иначе шум CI ломает ±5 %.
  * Baseline обновляется отдельным MR после планомерного ускорения.

Запуск:

    PYTHONPATH=. python ci/perf_check.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_PATH = ROOT / "ci" / "perf_baseline.json"
TOLERANCE = 1.05  # +5 % regress max
N_RUNS = 7


def _bench_prepare_inputs() -> float:
    from app_cluster_pipeline import prepare_inputs

    snap = {
        "cluster_algo": "kmeans",
        "cluster_vec_mode": "tfidf",
        "call_col": "call",
        "chat_col": "chat",
    }
    files = [f"f_{i}.xlsx" for i in range(200)]
    t0 = time.perf_counter()
    prepare_inputs(files, snap)
    return time.perf_counter() - t0


def _bench_build_t5() -> float:
    from app_cluster_pipeline import build_t5_source_text

    snap = {"call_col": "call", "chat_col": "chat"}
    cluster_snap = {"ignore_chatbot": True}
    header = ["id", "call", "chat"]
    header_index = {"id": 0, "call": 1, "chat": 2}
    row = [1, "перевод между счетами", "статус операции"]
    t0 = time.perf_counter()
    for _ in range(2000):
        build_t5_source_text(row, header, snap, cluster_snap, header_index=header_index)
    return time.perf_counter() - t0


BENCHMARKS: Dict[str, Callable[[], float]] = {
    "prepare_inputs_sec_median": _bench_prepare_inputs,
    "build_t5_source_text_sec_median": _bench_build_t5,
}


def _run_with_warmup(fn: Callable[[], float]) -> float:
    fn()  # warmup, не учитываем
    samples: List[float] = [fn() for _ in range(N_RUNS)]
    return statistics.median(samples)


def main() -> int:
    if not BASELINE_PATH.is_file():
        print(f"[perf-check] no baseline at {BASELINE_PATH}, skipping gate")
        return 0
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    failed: List[str] = []
    for name, bench in BENCHMARKS.items():
        observed = _run_with_warmup(bench)
        ref = baseline.get(name)
        if ref is None:
            print(f"[perf-check] {name}: observed={observed:.4f}s "
                  f"(no baseline — informational)")
            continue
        ratio = observed / float(ref)
        verdict = "OK" if ratio <= TOLERANCE else f"REGRESS x{ratio:.2f}"
        print(f"[perf-check] {name}: observed={observed:.4f}s "
              f"baseline={ref:.4f}s ratio={ratio:.2f} → {verdict}")
        if ratio > TOLERANCE:
            failed.append(name)

    if failed:
        print(f"\n[perf-check] FAIL — regressed metrics: {failed}")
        print(f"[perf-check] Tolerance: +{int((TOLERANCE - 1) * 100)} %")
        print("[perf-check] Update ci/perf_baseline.json only after intentional speed-up.")
        return 1
    print("\n[perf-check] PASS — all metrics within ±5 % of baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
