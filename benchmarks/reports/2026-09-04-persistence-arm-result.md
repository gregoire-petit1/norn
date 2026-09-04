# Persistence arm (E) — result: no gain (2026-09-04)

Arm E = arm D (batch prompt + bash list tool) + empty-response retry
(`ed232aa`) + adaptive_rounds max 2 (`f78a9d5`). Same 20 tasks, gpt-5.3-codex,
Rosetta.

| Arm | Pass | LLM calls | Commands/task | Wall |
|---|---|---|---|---|
| D | 10/20 | 238 | 30.8 | 42 min |
| **E** | **9/20** | **309** | **35.0** | 34 min |
| T2 | 12/20 | 167 | 24.0 | 21 min |

**E did not improve the score (9 vs 10, within n=20 noise) and cost 30% more
LLM calls.** E-only pass: dna-insert. D-only: fix-code-vulnerability,
query-optimize. Pure churn, no net capability change.

## The empty-retry fix works — the task is just hard

circuit-fibsqrt (the motivating case): 9 empty `stop,0-token` responses,
6 retries fired. The agent **no longer abandons** — after each nudge it
resumes with real tool calls (31, 50, 50 completion tokens) and now passes
**2 of 3 subtests** (was 0). It fails `test_sqrt_fib` on the merits, not by
giving up. So the fix does exactly what it should; it just doesn't turn this
particular hard task green. Keep it — it converts silent give-ups into real
attempts, which is correct behaviour regardless of this one task.

## adaptive_rounds bought effort, not passes

Commands/task rose 30.8 → 35.0 (toward T2's 24 — actually past it), i.e. the
agent did grind more. But more grinding on tasks it fails the *substance* of
did not convert. The exploration gap was a symptom, not the cause: Norn
stops early on compute tasks because it cannot crack them, not because a
budget cap cut it off. Extending the budget just spends more to fail.

## Decision

- **Keep**: batch prompt, list tool, empty-retry — each is a correctness or
  efficiency win on its own merits (D is the best cost/score point).
- **Do not enable adaptive_rounds by default**: +30% cost, no pass gain here.
  Leave the flag off; revisit only if a future task set shows budget-capped
  give-ups (progress detected at the cap), which this set does not.
- The remaining Norn↔T2 gap (10 vs 12) is **substance on hard compute tasks**
  (raman-fitting numerics, torch parallelism, circuit sim), not harness
  mechanics. That is a model-capability frontier; harness tuning has reached
  diminishing returns on this subset.

## Caveats

n=20, k=1: the 9-vs-10 difference is noise; the honest read is "no change".
The cost increase (+30% calls) is real and consistent (adaptive_rounds by
construction adds rounds).
