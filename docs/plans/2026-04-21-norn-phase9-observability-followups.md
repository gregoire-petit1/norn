# Phase 9 — Observability Polish & Follow-ups

**Status:** Draft (created at Phase 8 close-out, 2026-04-21)
**Predecessor:** Phase 8 — Observability (closed at commit `28cc63c`, 485 tests passing)

## Context

Phase 8 shipped a complete observability foundation: 6 structured event types
(`agent.run`, `llm.complete`, `tool.call`, `permission.decision`,
`routing.decision`, `fallback`) emitted to a dual-sink (rich console + daily
rotating JSONL), correlated by session ID via `contextvars`, with fail-open
semantics throughout.

The final code review identified a backlog of non-blocking follow-ups, plus
two CLI commands (`norn logs tail/stats`) explicitly deferred from Phase 8 to
keep that POC minimal.

This document tracks all of these as Phase 9 candidates so we can pick a
subset and scope properly when we start the next iteration.

---

## Follow-up backlog (from Phase 8 reviews)

Items are grouped by theme. Each lists the originating review, the affected
files, and a sketch of the fix. Effort estimates are agent-time
order-of-magnitude (S = <30 min, M = 30-90 min, L = >90 min).

### A. Documentation drift

#### A1. `tool.call` schema mismatch (S)
**Source:** T6 quality review.
**Where:** `docs/plans/2026-04-21-norn-phase8-observability-design.md` §3.2.
**Drift:** Doc shows `"error": null`; implementation emits the harmonized
`error_type` + `error_message` pair (introduced in T5 fix cycle commit
`0295c18`). Update the schema example to match reality.

#### A2. `permission.decision.reason` nullability (S)
**Source:** T6 quality review nit #2 + T7 final review.
**Where:** Same design doc §3.2 + `src/norn/permissions/checker.py:65`.
**Drift:** Doc shows `"reason": "user_approved"`; implementation emits
`reason=decision.reason` unconditionally, which is `None` for auto-approved
LOW/MEDIUM risks. **Two viable fixes:**
- Fix the doc: change schema to `reason: string | null`.
- Fix the code: synthesize a default reason at emission time (`"auto_approved"`,
  `"yolo"`, `"user_approved"`, `"denied"`) so the field is always populated.
The latter is more useful for `jq` consumers and downstream dashboards. Pair
with A4 (test coverage of denied path) when implementing.

#### A3. Stale `TODO(T6)` in `processors.py:54` (XS)
**Source:** T7 final review.
**Where:** `src/norn/observability/processors.py` line 54 (likely refers to
recursive redaction not yet implemented).
**Fix:** Either implement the deferred behavior or re-tag the comment to
`TODO(phase9)` with a brief note on what's pending.

### B. Information preservation

#### B1. `error_type="ToolExecutionError"` literal loses real exception class (M)
**Source:** T6 quality review nit #1 + T7 final review.
**Where:** `src/norn/core/agent.py:136` (in `_execute_tool` instrumentation).
**Problem:** `_execute_tool_inner` catches all exceptions and wraps them as
`ToolResult(error=f"Tool execution error: {e}")`, losing the original
exception class. The instrumentation then emits a generic `"ToolExecutionError"`
literal regardless of whether the failure was an `OSError`, `ValidationError`,
`PermissionError`, etc. Three distinct error paths (`UnknownTool`,
`PermissionDenied`, real exception) all get the same `error_type`.
**Fix:** Plumb the original exception class through `ToolResult` (e.g. add
`error_class: str | None` field) or, simpler, parse `result.error` prefix at
emission time to disambiguate (`"Unknown tool:" → "UnknownTool"`,
`"Permission denied:" → "PermissionDenied"`, otherwise → `type(e).__name__`).
Adds real value for post-mortem analysis (`jq '. | select(.error_type ==
"PermissionError")'`).

### C. Performance & UX

#### C1. `norn version` loads full config + initializes logging (S)
**Source:** T4 quality review nit #2.
**Where:** `src/norn/cli/main.py` `version` command.
**Problem:** `version` is conceptually a metadata query but now triggers
config YAML loading + log dir creation. Adds latency to a command often called
in shell prompts / CI scripts.
**Fix options:**
- Skip `_bootstrap_logging` for `version`.
- Or hardcode `enabled=False` for `version` only.
- Or accept as documented behavior and move on.

#### C2. `_provider_from_model` heuristic for unprefixed models (S)
**Source:** T5 quality review nit #1.
**Where:** `src/norn/core/llm.py` `_provider_from_model`.
**Problem:** Unprefixed model names (`"gpt-4"`, `"claude-3"`) all default to
`provider="openai"` because that's litellm's own convention for bare ids. Bare
Anthropic models would be misreported as OpenAI in events. Production code
always goes through `build_litellm_provider` (which prefixes), so this only
affects ad-hoc instantiation, but worth tightening: store the original
provider on the instance, or extend the heuristic with a prefix-table for
common bare-id models.

### D. Test hygiene & coverage gaps

#### D1. Concurrent-test patching of `litellm.acompletion` is fragile (S)
**Source:** T7 final review test quality nit #1.
**Where:** `tests/test_observability/test_e2e.py` test #7
(`test_concurrent_agent_runs_do_not_cross_contaminate_sessions`).
**Problem:** Two `with patch("litellm.acompletion", ...)` blocks nested inside
concurrent asyncio tasks patch the same global attribute. Currently passes
because both mocks return identical payloads, but a future change with
divergent mocks per task could produce confusing failures.
**Fix:** Patch at the provider instance level, or use a single top-level patch
with a shared dispatcher.

#### D2. Test #3 cost assertion is non-discriminating (S)
**Source:** T7 final review test quality nit #2.
**Where:** Same file, `test_cost_included_when_enabled`.
**Problem:** Asserts `cost_usd is None OR isinstance(..., float)` — a
regression where `litellm.completion_cost` always raised would still pass
(cost_usd would consistently be `None`). The processor unit test already
covers happy path, so this is technically defensive duplication, but the E2E
test should at minimum assert that one *known-priced* model produces a numeric
cost.

#### D3. Missing coverage paths (M)
**Source:** T7 final review coverage gaps.
**Add tests for:**
- Permission **denied** path (only `granted=True` covered today).
- `agent.run` end with `success=False` (real failure E2E, error field
  populated).
- `cost_usd: null` explicit assertion when model is unsupported by litellm.
- Multiple tool calls in one turn (parallel execution within single LLM
  response).
- Empty `messages=[]` edge case.

#### D4. JSONL read while file handle still open (XS, OS-portability)
**Source:** T7 final review test quality nit #4.
**Where:** Most E2E tests except #5.
**Problem:** Tests read `~/tmp/.../YYYY-MM-DD.jsonl` while the file handler is
still attached to the root logger. POSIX append-mode tolerates this; Windows
CI might fail (locked file).
**Fix:** Adopt the test #5 pattern uniformly — call `_close_handlers()` before
reading. Alternative: refactor into a fixture that handles teardown.

### E. CI & cross-platform

#### E1. End-to-end CI smoke test (M)
**Source:** T7 final review post-Phase-8 follow-ups.
**Goal:** Add a CI job that runs `norn run "echo hi"` (or equivalent
deterministic command) and greps `~/.norn/logs/*.jsonl` for the 6 expected
event names. Catches regressions where `init_logging` isn't wired into a new
CLI command — a class of bug no unit test would surface.

#### E2. Windows CI coverage (M)
**Source:** Same as D4.
**Goal:** Add a Windows runner to the CI matrix to surface file-locking
issues introduced by the JSONL-while-open pattern, ContextVar quirks on
ProactorEventLoop, etc. May be deferred indefinitely if cross-platform isn't
on the roadmap.

### F. New CLI commands (originally deferred from Phase 8)

#### F1. `norn logs tail` (M)
Stream the current day's JSONL with optional filters: `--event tool.call`,
`--session <uuid>`, `--since 5m`, `--level error`. Use `rich` for pretty
formatting in interactive mode, `--json` for raw passthrough.

#### F2. `norn logs stats` (M-L)
Aggregate today's logs into a summary:
- Event counts by type
- Tool usage frequency
- Total LLM cost (USD)
- Average latency per tier
- Permission decision distribution (granted/denied)
- Active sessions (count + duration)
Implementation hint: leverage DuckDB to query the JSONL files directly (no
ETL needed — DuckDB's `read_json_auto` handles this natively, and matches
the data-engineering stack guidance from AGENTS.md).

#### F3. `norn cost report` (M)
Cost-focused subset of `logs stats`. Aggregates `cost_usd` from
`llm.complete` events, breakdown by model/tier/session/day.

---

## Suggested Phase 9 scope (proposal)

**Tier 1 — Quick wins (S only, ~1.5 hours total):**
- A1, A2 (doc drift)
- A3 (stale TODO)
- C1 (version latency)
- D1 (test #7 hardening)
- D2 (cost test discrimination)

**Tier 2 — Information preservation:**
- B1 (real exception class in tool.call) — high value for debugging

**Tier 3 — CLI commands (the bulk of Phase 9 if we go ambitious):**
- F1 (`norn logs tail`)
- F2 (`norn logs stats`) — DuckDB-backed
- F3 (`norn cost report`)

**Tier 4 — Optional / opportunistic:**
- C2 (provider heuristic)
- D3 (coverage gaps)
- D4 + E2 (Windows portability) — defer unless Windows is needed
- E1 (CI smoke test) — depends on existing CI setup

A reasonable Phase 9 v1 would bundle Tier 1 + B1 + F1, leaving F2/F3 for v2.

---

## Notes

- This tracker is a living document — add new items as they emerge during
  Phase 9 brainstorming.
- Each Phase 9 task should produce its own implementation plan
  (`docs/plans/2026-04-XX-norn-phase9-<topic>-implementation.md`) following
  the same TDD + subagent-driven cadence as Phase 8.
- Phase 8 commits range: `0b4261e..28cc63c` (14 atomic commits, all on
  `main`).
