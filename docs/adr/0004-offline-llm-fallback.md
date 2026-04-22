# ADR-0004: Offline LLM fallback for tests and CI

## Status
Proposed (target: Wave 4 CI integration, Wave 5 coverage gate).

## Context
Several features depend on a live LLM provider:
- `llm_reranker.rerank_top_k` — picks one of top-K candidates on
  low-confidence rows.
- Cluster auto-naming in `app_cluster_pipeline` postprocess stage.
- Active-learning rank explanations.

CI must be deterministic and runnable without network access or API
keys. Running the real providers in CI causes three problems:
1. Flakiness from upstream rate-limits and transient 5xx errors.
2. Hidden cost and key-rotation overhead.
3. Non-determinism that makes coverage and mutation gates unsafe.

## Decision
1. Introduce an explicit environment switch:
   `BRT_LLM_PROVIDER=offline`. When set, `llm_client.complete_text`
   short-circuits to a local stub that reads a deterministic JSONL
   replay file at `tests/fixtures/llm_replay.jsonl`.
2. The replay file is keyed by `(provider, model, user_prompt hash,
   temperature)` (the same key used in `_cache_key`). Missing keys
   return a stable default chosen by the test, with the first
   candidate echoed back for `rerank_top_k`.
3. Each test that exercises an LLM-touching code path pins
   `BRT_LLM_PROVIDER=offline` in a fixture, and records replay
   entries via a `record-golden` mode gated by another env
   (`BRT_LLM_RECORD_GOLDEN=1`) run manually against a real provider.
4. The offline provider lives in `llm_client.py` alongside the real
   providers — not in a separate module — so cache-key logic stays
   in one place.
5. Production code never checks `BRT_LLM_PROVIDER` directly; only
   `llm_client.complete_text` branches on it.

## Consequences
+ CI becomes fully deterministic and key-free.
+ `rerank_top_k` tests in `test_wave4_quick_wins.py` and broader
  coverage of `ml_training` (Wave 5, target ≥ 70 % branch coverage)
  become feasible without patching internals.
+ The JSONL replay file is reviewable: any change to LLM wording is
  visible in the diff.
− Replay files must be curated and kept in sync with prompt changes.
  Each change to `_build_rerank_user` invalidates entries; CI prints
  a warning listing stale hashes.
− The `record-golden` pass still requires a real key; run it locally
  before shipping LLM-touching changes.

## References
- `llm_client.py` `complete_text` (branch target)
- `llm_reranker.py` (primary consumer)
- Roadmap: Wave 4 "offline back-translate fallback" and Wave 5
  "coverage ≥70 %" in `/root/.claude/plans/graceful-cooking-taco.md`
