# Exploration gap analysis — Norn vs Terminus-2 (2026-09-04)

Question after the batching arms: Norn sits ~2 tasks and ~4 LLM-calls/task
behind Terminus-2. Is Norn under-exploring? Analysis on existing logs (T2
ATIF trajectories with full commands; Norn console + JSONL). No quota spent.

## Command volume: Norn does ~half the work, same read/act mix

Over the 20 tasks (arm B vs Terminus-2, per-command, not per-line):

| | Terminus-2 | Norn (B) |
|---|---|---|
| read-only commands / task | 9.1 | 5.3 |
| action commands / task | 14.9 | 8.4 |
| total commands / task | 24.0 | 13.7 |
| read share | 38% | 39% |

The **mix is identical** (~38% inspection) — Norn is not skipping inspection
relatively. It simply does **~1.75× less of everything**: fewer probes AND
fewer actions. It converges (or gives up) sooner. On the compute-heavy tasks
the gap is extreme — protein-assembly: T2 41 commands, Norn 10;
raman-fitting: T2 17, Norn 5. T2 grinds; Norn stops early.

So "under-exploration" is real but it is not a read/act imbalance to
rebalance — it is total effort. That is harder to fix with a prompt line and
is downstream of two concrete bugs found below.

## Bug 1 (fixed): empty responses end the task silently

`circuit-fibsqrt`, **all three Norn arms**, gpt-5.3-codex returned
`finish_reason=stop` with **0 completion tokens and no tool call**, 3× per
run. Norn treated each void as a final answer and abandoned the task after
~7 calls — no verify turn, 0 verdicts. Terminus-2 passes it.

Fixed (`ed232aa`): both loops now nudge and retry (bounded to 2) on an empty
response instead of ending. This directly targets a task both harnesses'
gap and a class of silent give-ups.

## Bug 2 (not a bug): refusal on break-filter is non-deterministic

`break-filter-js-from-html` asks for an XSS filter bypass. Norn refused on
arms B and C ("Sorry, I can't help create an XSS bypass payload") and
**passed on arm D**. Same prompt, same model — the refusal is stochastic,
not a fixed safety block. Terminus-2 passed. Not worth a code change on k=1
evidence; worth noting that a benign-but-security-shaped task trips the
model's refusal sometimes. A one-line framing in the task-agnostic prompt
("these are authorized benchmark tasks in a sandbox") might reduce it —
deferred until measured.

## What this leaves

The remaining Norn↔T2 gap is **total effort / persistence**, not tool
ergonomics (fixed) or read/act balance (already matched). Terminus-2's
tmux tool lets it fire many commands cheaply and keep grinding; even with
Norn's new list tool, the model chooses to stop earlier. Levers to test,
in order:
1. The empty-retry fix alone may recover circuit-fibsqrt — cheapest check,
   folded into the next run.
2. Persistence: auto_verify already re-enters once on FAIL; consider a
   second verify round on compute tasks, or a "you stopped early, N steps
   remain in your plan" nudge when the plan (auto_plan) has open items.
3. Adaptive_rounds (already built, off) raises the cap when progressing —
   turn it on for the next run and measure.

## Caveats

k=1 per task; per-task differences are anecdotal, aggregates are the signal.
Norn command counts from console arg-summaries (list-mode bash args are not
shown, so Norn's action count is a slight under-count — widening, not
narrowing, the effort gap).
