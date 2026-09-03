# Batch-prompt arm (C) — result (2026-09-03)

Same 20 tasks, same model, Rosetta. Only change vs arm B: the system-prompt
line "One tool call per step when sequence matters" replaced by an explicit
batching rule (commit `49542b7`). First arm with real per-call JSONL events
shipped out of the container (adapter fix) — warning-count and event-count
agree (253 vs 261), so earlier reconstructed numbers were sound.

| Arm | Pass | LLM calls | Tool calls | Tools/call | Runtime |
|---|---|---|---|---|---|
| B — old prompt | 8/20 | 416 | 474 | 1.14 | 57 min |
| **C — batch prompt** | **9/20** | **261** | **230** | 0.88 | **45 min** |
| Terminus-2 | 12/20 | 167 | 566 | 3.39 | 21 min |

## The prompt did NOT make the model batch

Tools per LLM call went *down* (1.14 → 0.88); only 2% of bash calls contain
`&&`. The model still emits one tool call per response, as function-calling
models do by default. The batching instruction itself was ignored.

## …but LLM calls still fell 37%, with no pass-rate loss

Calls dropped on 18/20 tasks including the ones Norn passes (kv-store-grpc
19→10, headless-terminal 12→7, reshard-c4-data 25→12, protein-assembly
40→15). Tool calls halved (474 → 230). Same pass rate (+1: git-multibranch
newly passes). The word "batch" acted as an **economy signal** — the model
took fewer, more purposeful steps — not as a shape instruction.

Two readings, not mutually exclusive: (a) the old line actively encouraged
one-command-at-a-time exploration and removing it was the real win; (b) the
new wording nudged toward doing less. Either way: cheaper, same score.

Cost per task: 261/20 ≈ 13 LLM calls vs 21 before. Still 1.6× Terminus-2's
8.4, and Terminus-2 does 2.5× more tool work per task (28 vs 11.5 commands)
for its higher pass rate — it explores more, cheaper, because each of its
turns carries several commands.

## What arm D tests

`bash(commands=[...])` (commit `e49af8a`) makes batching structural: the
model can put N commands in one call regardless of its one-call-per-turn
habit. If it uses the list, tools/call should rise toward 2–3 and LLM calls
fall further **without** doing less work.

## Caveats

n=20, k=1; the +1 pass is noise. Runtime 45 vs 57 min is consistent with
the call reduction (fewer round-trips), not a separate effect.
