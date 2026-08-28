# SOTA v2 Wave 1 — Benchmark Validation (2026-08-28)

Model: `github_copilot/claude-sonnet-4.5` (all tiers). 16 internal tasks, sequential, 5s inter-task delay. Raw results: `benchmarks/results/sota-v2-wave1/`.

## Runs

| Run | Code | Config | Pass | Total latency | Avg/task |
|-----|------|--------|------|--------------|----------|
| Before | `142e98d` (pre-wave-1, worktree) | default | **16/16 (100%)** | 872s | 54.5s |
| After | `main` | default (new flags off, `thread_invariants` on) | **16/16 (100%)** | 746s | 46.6s |
| After+flags | `main` | `stable_prompt: true`, `sandbox.enabled: true` (workspace-write, network allowed) | **16/16 (100%)** | 772s | 48.2s |

`norn bench diff`: no regressions, both comparisons.

## Findings

- **Zero functional regression** from wave 1, both with default config and with the new flags enabled.
- **Thread invariant (W1.2)**: active on every run (`thread_invariants` defaults on) — **0 `invariant.violation` events** across ~314 LLM dispatches.
- **Sandbox (workstream C)**: 55 `sandbox.decision` events in the flags run, all `granted` under seatbelt `workspace-write`; no task needed writes outside the workspace, no escalation triggered. Latency overhead vs default run: ~+3.5% total — negligible.
- **Streaming `llm.complete` (workstream D)**: interactive/headless sessions now appear in the JSONL logs (they previously emitted nothing) — this validation itself relied on it.
- **Cache metrics: not measurable on Copilot.** Two provider limits, not wave-1 bugs:
  1. `github_copilot/*` is not in `_CACHE_ELIGIBLE_PREFIXES`, so `cache_control` markers (and the `stable_prompt` split) are never sent — the CACHE_BREAK sentinel is correctly stripped.
  2. litellm's github_copilot streaming returns no usage object, so token/cache counts are 0 in `llm.complete` for this provider.
  Measuring the `stable_prompt` cache gain requires a run against Anthropic direct (or OpenRouter/Anthropic). Follow-up.
- **W3.3 guard baseline initialised**: 5/5 easy tasks, stored in `benchmarks/results/guard/`. `norn bench guard` is now armed after each `/reflect`.

## Caveats

- The internal suite is saturated at 100% with claude-sonnet-4.5 — it no longer discriminates at this model tier. External comparability (terminal-bench via Harbor) or harder tasks needed for future deltas.
- Single run per config (n=1); latency deltas within run-to-run noise.
