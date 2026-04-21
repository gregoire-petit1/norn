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

## Phase 9 v1 — Completed (2026-04-22)

Scope delivered: Tier 1 + B1 + F1 (8 tasks). All commits on `main`, no push.

### Items closed

| Item | Status | Commit(s) | Notes |
|---|---|---|---|
| A1 | ✅ | `8bb36c3` | Design doc updated with harmonised `tool.call` schema (success omits error fields, failure emits `error_type` + `error_message`). |
| A2 | ✅ | `31fbf05` + `396b617` | Initial fix synthesised reasons; review fixup added `PermissionDecisionReason` StrEnum (`yolo`, `auto_approved`, `user_approved`, `user_denied`, `destructive_denied`, `prompt_required_no_handler`), e2e log assertions in `test_instrumentation_periphery.py`, and a "Reason taxonomy" subsection in the Phase 8 design doc. |
| A3 | ✅ | `d8aaac3` | `TODO(T6)` retagged to `TODO(phase9-v2)` with cross-reference. |
| C1 | ✅ | `be3aede` | `norn version` is now a 4-line trivial print: no `NornConfig.load()`, no `_bootstrap_logging()`, no `verbose` flag. |
| D2 | ✅ | `b8eaf4f` | Tightened the e2e cost assertion to require numeric `> 0` for `gpt-3.5-turbo`. **Bonus discovery**: revealed a silent litellm API drift bug (`completion_cost` no longer accepts `prompt_tokens`/`completion_tokens` kwargs) → switched the cost processor to `litellm.cost_per_token`. |
| D1 | ✅ | `234e59b` | `LiteLLMProvider` now accepts an optional `completion_fn` injected at construction time, lazily resolved from `litellm.acompletion` if `None`. The concurrent test (`test_concurrent_agent_runs_do_not_cross_contaminate_sessions`) injects per-instance `AsyncMock`s instead of patching the module global, with `assert_awaited_once()` proving isolation. |
| B1 | ✅ | `7848bd1` | Added `error_type: str \| None` field to `ToolResult`. The 3 systemic origin sites in `agent.py` now emit `"UnknownTool"`, `"PermissionDenied"`, `type(e).__name__`. The ~30 per-tool `ToolResult(error=...)` sites in `src/norn/tools/**/*.py` keep falling through to the `"ToolExecutionError"` fallback (deferred to v2). |
| F1 | ✅ | `5a000da` (plumbing) + `2fc45f3` (CLI) | `new_session()` persists to `~/.norn/state/last_session` (atomic, fail-open). New `norn logs tail` command in `src/norn/cli/logs.py` mounted as a `logs` Typer sub-app. Filters: `--event` (repeatable, OR), `--session` (UUID or `last`/`current` alias), `--json` passthrough. Default rich format: `HH:MM:SS [LEVEL] event.name k=v ...` with level colours. Test conftest extended with `_redirect_state_dir` session-scoped autouse fixture to protect the user's real state file. |

### Bottom-line metrics

- **Tests:** 485 (Phase 8 close) → **507 passed** (+22 net: +6 A2, -1 C1, +3 B1, +13 F1, +1 e2e log A2 review)
- **Ruff:** 37 → **38 errors** (+1 E402 on the `from norn.cli.logs import logs_app` line in `main.py`, structurally identical to the 28 pre-existing E402s caused by `load_dotenv()` running before the norn import block — accepted as baseline drift)
- **Commits:** 11 commits on `main` (`8bb36c3..2fc45f3`), no push
- **Plans/design docs added:**
  - `docs/plans/2026-04-22-norn-phase9-logs-tail-design.md`
  - `docs/plans/2026-04-22-norn-phase9-v1-implementation.md`

### Follow-ups carried forward to Phase 9 v2

The A2 code review surfaced 4 "Important" issues; 3 of them were absorbed into the A2 fixup commit (`396b617`). The remaining one — granularity unification of auto-approve reasons — was decided in scope (collapsed to `AUTO_APPROVED`).

For B1: per-tool granular `error_type` (replacing the `"ToolExecutionError"` fallback for the ~30 sites in `src/norn/tools/**/*.py`) is a Phase 9 v2 candidate if downstream consumers need it.

For F1: `--follow` (live tail), `--since 5m`, `--level error`, `--lines N`, multi-day reading, and pager-style pagination are explicitly deferred to v2 per the design doc §1.4.

Other unaddressed items from this tracker:
- **C2** — `_provider_from_model` heuristic (deferred)
- **D3** — Coverage gaps (denied path, agent.run failure E2E, multi-tool turn, empty messages)
- **D4 + E2** — Windows portability
- **E1** — CI smoke test
- **F2** — `norn logs stats` (DuckDB-backed)
- **F3** — `norn cost report`

---

## Phase 9 v2 — Completed (2026-04-22)

Scope delivered: E1 (CI), D3 (coverage gaps), B1.5 (ToolErrorType taxonomy across all tool families), G (prompt caching). All commits on `main`, no push.

### Items closed

| Item | Status | Commit(s) | Notes |
|---|---|---|---|
| E1 | ✅ | `115ebbc` + `5a4da3b` | Cleaned up the F401/N806/SIM105 ruff violations that would have failed `ruff check` in CI; added `.github/workflows/ci.yml` running `ruff check` + `pytest` on Ubuntu / Python 3.11 with `uv sync --all-extras`. Smoke test on every push & PR. |
| D3.1 | ✅ | `33f0c9b` | E2E test for the denied destructive command path: STRICT mode + `bash` HIGH-risk tool → `permission.decision` event with `reason=destructive_denied`, `granted=false`. |
| D3.2 | ✅ | `6b56d37` | E2E test for `agent.run` failure propagation: tool that raises → `agent.run(success=false, error_type, error_message, duration_ms)` end event, exception re-raised to caller. |
| D3.3 | ✅ | `de00f31` (+ `6fbc5bf` SIM117 fixup) | E2E test for the multi-tool turn: single LLM response with 2 native tool_calls → 2 `tool.call` events both within one turn. |
| D3.4 | ✅ | `e21ccb9` | `LiteLLMProvider.complete()` now guards against empty `messages=[]` with a deterministic `ValueError("messages cannot be empty")` before any network call. |
| B1.5.0 | ✅ | `f02bc9b` | Introduced `ToolErrorType` StrEnum in `src/norn/tools/base.py`: `FILE_NOT_FOUND`, `PERMISSION_DENIED`, `INVALID_ARGUMENT`, `TIMEOUT`, `NETWORK_ERROR`, `HTTP_ERROR`, `PARSE_ERROR`, `NOT_SUPPORTED`, `EXECUTION_ERROR`, `RESOURCE_EXHAUSTED`. Mirrors `PermissionDecisionReason` from v1. |
| B1.5.1 | ✅ | `08cf80d` | Applied taxonomy to `file_read` / `file_write` / `file_edit` (5 sites). |
| B1.5.2 | ✅ | `001c7b6` | Applied to `bash` (3 sites: `INVALID_ARGUMENT`, `TIMEOUT`, `EXECUTION_ERROR`). |
| B1.5.3 | ✅ | `9f46e74` | Applied to `grep` + `glob` (3 sites). |
| B1.5.4 | ✅ | `1de5179` | Applied to `web_fetch` + `web_search` (6 sites: `NOT_SUPPORTED` for missing httpx, `HTTP_ERROR` for 4xx/5xx, `NETWORK_ERROR` catch-all). |
| B1.5.5 | ✅ | `40c561e` | Applied to ML tools — `dataset_inspector`, `model_inspector`, `model_card`, `model_eval`, `tensor_inspector` (15 sites). |
| G.1 | ✅ | `9740d63` | Pure helpers `_supports_prompt_cache(model)` (allowlist: `anthropic/`, `openrouter/anthropic/`, `openai/gpt-4o`, `openai/gpt-4.1`) and `_apply_cache_markers(messages, tools, enabled)` (tags system prompt + last tool schema with `cache_control: ephemeral`). Pure-function, no side effects. |
| G.2 | ✅ | `a6b72bb` | Wired into `LiteLLMProvider.__init__(prompt_cache: bool = True)` (gates immediately on `_supports_prompt_cache(model)` so the hot path stays a single bool check) and `complete()` (applies markers + emits `cache_read_tokens` + `cache_creation_tokens` on the `llm.complete` observability event). |
| G.3 | ✅ | `8526700` | Added `prompt_cache: bool = True` to `LLMConfig`, forwarded through `build_litellm_provider(..., prompt_cache=...)` and `cli/main.py:_build_provider`, documented in `configs/default.yaml` with a comment block explaining the behaviour. |

### Bottom-line metrics

- **Tests:** 507 (Phase 9 v1 close) → **560 passed** (+53 net: +5 D3, +1 B1.5.0, +5 B1.5.1, +3 B1.5.2, +5 B1.5.3, +6 B1.5.4, +13 B1.5.5, +11 G.1, +4 G.2, +2 G.3, with some absorbed into existing files)
- **Ruff:** 38 → **29 errors** (-9 from E1 cleanup; baseline maintained throughout B1.5 and G)
- **Commits:** 16 commits on `main` (`115ebbc..8526700`), no push
- **CI:** `.github/workflows/ci.yml` exercises both `ruff check` and `pytest` on every push & PR (Ubuntu / Python 3.11)

### Follow-ups

The taxonomy work covers ~32 sites across all 5 tool families. The generic `"ToolExecutionError"` literal in `agent.py` still applies for tools that don't override `error_type`; this is now a per-tool deferral rather than a systemic gap.

Items from the original tracker still unaddressed:
- **C2** — `_provider_from_model` heuristic
- **D4 + E2** — Windows portability
- **F2** — `norn logs stats` (DuckDB-backed)
- **F3** — `norn cost report`

Phase 9 v2 design notes:
- `docs/plans/2026-04-22-norn-phase9-v2-design.md`
- `docs/plans/2026-04-22-norn-phase9-v2-implementation.md`

---

## Notes

- This tracker is a living document — add new items as they emerge during
  Phase 9 brainstorming.
- Each Phase 9 task should produce its own implementation plan
  (`docs/plans/2026-04-XX-norn-phase9-<topic>-implementation.md`) following
  the same TDD + subagent-driven cadence as Phase 8.
- Phase 8 commits range: `0b4261e..28cc63c` (14 atomic commits, all on
  `main`).
