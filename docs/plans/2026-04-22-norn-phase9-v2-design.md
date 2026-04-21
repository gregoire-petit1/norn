# Norn — Phase 9 v2 Design (Robustness & Quality)

**Date:** 2026-04-22
**Status:** Approved, ready for implementation planning
**Predecessor:** Phase 9 v1 (`docs/plans/2026-04-22-norn-phase9-v1-implementation.md`)
**Tracker:** `docs/plans/2026-04-21-norn-phase9-observability-followups.md`

---

## 1. Goal

Solidify the foundations laid by Phases 8 and 9 v1 before stacking more
observability features. Four workstreams, executed linearly:

1. **E1** — Minimal CI on GitHub Actions (protective net first).
2. **D3** — Four end-to-end coverage gaps (denied path, agent.run failure,
   multi-tool turn, empty messages).
3. **B1.5** — `ToolErrorType` taxonomy + application to the ~30 per-tool
   `ToolResult(error=...)` sites.
4. **G** — Prompt caching (litellm `cache_control`) on system prompt + tool
   schemas to reduce input token cost on multi-turn agent loops.

The ordering is deliberate: ship the CI net first so that every subsequent
test addition and refactor is automatically validated. Workstream G runs
last because it is a behavioural optimisation that benefits from the
strengthened test suite.

## 2. Out of scope

Deferred to Phase 9 v3 or later:

- F1.5 (`norn logs tail` advanced flags: `--follow`, `--since`, `--level`,
  `--lines`, multi-day, pager)
- F2 (`norn logs stats` with DuckDB)
- F3 (`norn cost report`)
- C2 (`_provider_from_model` heuristic)
- D4 + E2 (Windows portability)
- `obsidian_rag` tool (planned as Phase 10 dedicated workstream)
- Internal benchmark / mini-eval (planned as Phase 11)
- Watch mode (file-trigger Dream/Coordinator)
- Repo map / tree-sitter integration (Aider-style)
- Scratchpad concurrent bottleneck (note for when multi-session arrives)

Rationale: each of these is a feature; Phase 9 v2 deliberately invests in
non-feature quality work plus one targeted cost optimisation (G).

---

## 3. Workstream E1 — Minimal CI

### 3.1 Scope

A single GitHub Actions workflow that runs on every push to `main` and on
every pull request targeting `main`. It runs ruff and pytest on a single
matrix point (Ubuntu, Python 3.11).

### 3.2 File

`.github/workflows/ci.yml`

### 3.3 Steps

1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v3` with built-in cache enabled
3. `uv sync --all-extras`
4. `uv run ruff check .`
5. `uv run pytest`

### 3.4 Ruff baseline decision

The current ruff baseline is 38 errors:

- 28× E402 (caused by `load_dotenv()` running before the norn import block in
  `src/norn/cli/main.py` — structural, accepted)
- 1× F401, 4× N806, 4× SIM105 (legitimate cleanup candidates)

**Decision:** before enabling CI, fix the 1 F401 + 4 N806 + 4 SIM105 in a
single preparatory commit, leaving only the 28 E402s (which are an accepted
design constraint). The CI then runs `uv run ruff check .` with no flags and
must pass.

### 3.5 Success criteria

- Workflow file committed
- First push to `main` after the workflow lands shows a green check
- A deliberately broken test or ruff violation in a follow-up branch turns
  the check red

### 3.6 Risks

- **uv cache invalidation** between runs: mitigated by `astral-sh/setup-uv@v3`
  which handles `uv.lock` hashing automatically.
- **Hidden test flakiness on Linux** (we develop on macOS): if a test fails
  only on Linux, we accept fixing it as part of this workstream.

---

## 4. Workstream D3 — Coverage gaps

### 4.1 Test 1: Denied path E2E

**Location:** `tests/test_observability/test_e2e.py`

**Scenario:**
- Mock provider returns a `tool_call` for `bash` with `command="rm -rf /tmp/foo"`
- Mock permission handler returns `deny`
- Agent loop continues for one more turn (mock provider then returns a final
  text response acknowledging the denial)

**Assertions:**
- The `tool.call` log event has `error_type == "PermissionDenied"`
- The `tool.call` log event has the appropriate permission decision reason
  (from `PermissionDecisionReason`)
- The session closes cleanly via `session.end`
- No uncaught exception escapes `agent.run()`

### 4.2 Test 2: agent.run failure E2E

**Location:** `tests/test_observability/test_e2e.py`

**Scenario:**
- Mock provider succeeds on turn 1 (returns a tool_call)
- Tool executes successfully
- Mock provider raises `RuntimeError("provider 500")` on turn 2

**Assertions:**
- `agent.run()` re-raises the `RuntimeError` to the caller
- The session is closed (a final `session.end` event is emitted, possibly
  with a failure marker — to be specified during implementation)
- The error is logged at error level with full context
- No file handle / async task is leaked

### 4.3 Test 3: Multi-tool turn

**Location:** `tests/test_observability/test_e2e.py`

**Scenario:**
- Mock provider returns 2 `tool_calls` in a single turn (e.g.
  `file_read("/tmp/a")` + `glob("**/*.py")`)
- Both tools succeed

**Assertions:**
- Both tools execute (sequentially, in declared order)
- Two distinct `tool.call` log events are emitted, in order
- Both events share the same `turn_id` / parent context
- Both tool results are passed back to the provider in the next turn
- If one of the two tools fails, the other still runs and emits its event

**Pre-flight check:** verify that the existing mock provider in the test
suite supports returning multiple `tool_calls` per turn. If not, extend it
(small change, no production impact).

### 4.4 Test 4: Empty messages edge case

**Location:** `tests/test_core/test_llm.py` (or new file if cleaner)

**Scenario:**
- `LiteLLMProvider.complete(messages=[])` is invoked

**Assertions:**
- The provider raises a clear `ValueError` (or whatever litellm raises) — to
  be specified once we observe actual behaviour during implementation
- No silent passthrough that would later produce a confusing API error

This is the lowest-priority test; if implementation reveals litellm already
handles it gracefully, we add a single assertion documenting the behaviour
and move on.

---

## 5. Workstream B1.5 — `ToolErrorType` taxonomy

### 5.1 Enum definition

**File:** `src/norn/tools/base.py` (next to `ToolResult`)

```python
from enum import StrEnum


class ToolErrorType(StrEnum):
    """Semantic error categories for ToolResult.error_type.

    Used by tool implementations to tag failures with stable, queryable
    categories. Mirrors the taxonomy approach introduced for permissions
    in Phase 9 v1 (PermissionDecisionReason).
    """

    FILE_NOT_FOUND = "FileNotFound"
    PERMISSION_DENIED = "PermissionDenied"
    INVALID_ARGUMENT = "InvalidArgument"
    TIMEOUT = "Timeout"
    NETWORK_ERROR = "NetworkError"
    HTTP_ERROR = "HttpError"
    PARSE_ERROR = "ParseError"
    NOT_SUPPORTED = "NotSupported"
    EXECUTION_ERROR = "ExecutionError"
    RESOURCE_EXHAUSTED = "ResourceExhausted"
```

10 values, extensible. The `ToolResult.error_type` field stays typed as
`str | None` (not `ToolErrorType | None`) to preserve compatibility with the
systemic path in `agent.py`, which passes `type(e).__name__` for unknown
tools and unhandled exceptions. Tools that opt into the taxonomy pass
`error_type=ToolErrorType.X.value`.

### 5.2 Inventory & mapping

Step 1 of implementation: run `rg "ToolResult\(.*error="` across
`src/norn/tools/**/*.py` to enumerate the exact sites and current error
patterns.

Conservative mapping rules:

| Source pattern | `ToolErrorType` |
|---|---|
| `FileNotFoundError`, `IsADirectoryError`, missing path checks | `FILE_NOT_FOUND` |
| `PermissionError`, OS-level access denied | `PERMISSION_DENIED` |
| Pydantic validation, missing/invalid args, schema mismatch | `INVALID_ARGUMENT` |
| `asyncio.TimeoutError`, bash hard timeout, web fetch timeout | `TIMEOUT` |
| `httpx.ConnectError`, `socket.gaierror`, DNS failures | `NETWORK_ERROR` |
| HTTP 4xx/5xx responses (web tools) | `HTTP_ERROR` |
| JSON/YAML/XML parse failures | `PARSE_ERROR` |
| Tool/feature explicitly unavailable on this platform | `NOT_SUPPORTED` |
| OOM, disk full, GPU OOM | `RESOURCE_EXHAUSTED` |
| Anything else | `EXECUTION_ERROR` |

### 5.3 Commit granularity

To keep diffs reviewable, B1.5 lands as one commit per tool family:

1. `feat(tools): apply ToolErrorType to file_* tools` (file_read, file_write, file_edit)
2. `feat(tools): apply ToolErrorType to bash tool`
3. `feat(tools): apply ToolErrorType to grep + glob tools`
4. `feat(tools): apply ToolErrorType to web_* tools` (web_fetch, web_search)
5. `feat(tools): apply ToolErrorType to ML tools` (dataset_inspector, model_*, tensor_inspector)
6. `feat(tools): introduce ToolErrorType enum` (the enum itself, landed first)

Order of execution: enum first (commit 6 above becomes commit 1), then each
family. Each commit ships its own tests.

### 5.4 Tests

For each tool modified, add or extend a test that:
- Forces the error path (e.g. read a nonexistent file)
- Asserts `result.error_type == ToolErrorType.FILE_NOT_FOUND.value`

Estimated test additions: +20 to +30, with consolidation where multiple
tools share an error path.

### 5.5 Backward compatibility

- `ToolResult.error_type` stays `str | None` → no breaking change for the
  fallback path in `agent.py`
- Consumers querying logs by `error_type` start seeing semantic values
  instead of `"ToolExecutionError"` for tool-level failures
- The taxonomy is additive; tools that don't tag continue to fall through to
  the systemic `"ToolExecutionError"` fallback in `agent.py`

---

## 6. Workstream G — Prompt caching (litellm `cache_control`)

### 6.1 Motivation

In a multi-turn agent loop, the system prompt and tool schemas are
re-transmitted on every turn. For long sessions or large tool catalogs,
this dominates the input token bill. Anthropic, OpenAI (automatic on
gpt-4o family), and a few OpenRouter models support **prompt caching**: a
provider-side hash of a prefix block, with cached reads billed at 10%
(Anthropic) or 50% (OpenAI) of the normal input rate.

litellm exposes Anthropic-style caching via per-message
`cache_control: {"type": "ephemeral"}` markers. OpenAI's automatic caching
requires no markers but only kicks in for prefixes ≥ 1024 tokens.

### 6.2 Scope

Add cache markers to two stable prefix blocks per `LiteLLMProvider.complete`
call:

1. **System prompt** — first message, role=system, never mutates within a
   session
2. **Tool schemas** — passed via `tools=` parameter, stable across turns of
   the same agent run

The user-conversation messages remain uncached (they grow each turn).

### 6.3 Implementation sketch

In `src/norn/core/llm.py`:

```python
def _apply_cache_markers(
    messages: list[dict],
    tools: list[dict] | None,
    enabled: bool,
) -> tuple[list[dict], list[dict] | None]:
    """Tag system prompt + tool schemas with cache_control: ephemeral.

    No-op if `enabled` is False or if the model/provider does not support
    caching (litellm silently ignores unknown fields, but we gate to be
    explicit and to avoid masking real errors).
    """
    if not enabled:
        return messages, tools

    new_messages = list(messages)
    if new_messages and new_messages[0].get("role") == "system":
        # Anthropic format: content becomes a list of blocks
        sys_msg = dict(new_messages[0])
        content = sys_msg.get("content", "")
        if isinstance(content, str):
            sys_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        new_messages[0] = sys_msg

    new_tools = tools
    if tools:
        # Mark the tools block as cacheable; litellm propagates this to
        # provider-specific shapes.
        new_tools = list(tools)
        if new_tools:
            new_tools[-1] = {
                **new_tools[-1],
                "cache_control": {"type": "ephemeral"},
            }

    return new_messages, new_tools
```

Configuration knob:

```yaml
# configs/default.yaml
llm:
  prompt_cache: true   # default true; harmless on non-supporting providers
```

### 6.4 Provider eligibility

Conservative gate: enable only when the model id starts with
`anthropic/`, `openrouter/anthropic/`, or `openai/gpt-4o`. Other models
ignore the field but we want explicit opt-in to keep diagnostics clean.

A small helper in `llm.py`:

```python
_CACHE_ELIGIBLE_PREFIXES = (
    "anthropic/",
    "openrouter/anthropic/",
    "openai/gpt-4o",
    "openai/gpt-4.1",
)


def _supports_prompt_cache(model: str) -> bool:
    return model.startswith(_CACHE_ELIGIBLE_PREFIXES)
```

### 6.5 Observability

The `llm.complete` log event already records `prompt_tokens`. To verify
caching works, add two fields when the provider returns them (Anthropic
exposes `cache_creation_input_tokens` and `cache_read_input_tokens` on the
usage object):

```python
event["cache_read_tokens"] = getattr(usage, "cache_read_input_tokens", 0) or 0
event["cache_creation_tokens"] = getattr(usage, "cache_creation_input_tokens", 0) or 0
```

This makes the cache hit rate observable via `norn logs tail --event llm.complete`.

### 6.6 Tests

- Unit: `_apply_cache_markers` with system + tools, with system only,
  without system, disabled flag, edge cases (empty messages, no tools).
- Unit: `_supports_prompt_cache` truth table.
- Integration: mock `litellm.acompletion` to return a usage object with
  `cache_read_input_tokens=42` and assert the log event carries the field.

Estimated test additions: +6 to +8.

### 6.7 Risks

- **Anthropic minimum block size** — caching only kicks in for blocks
  ≥ 1024 tokens. Small system prompts won't benefit. Acceptable; we ship
  the markers anyway and let the provider decide.
- **Tool schema mutation breaks the cache** — if the tool list changes
  between turns (e.g. MCP load), the cache invalidates. Acceptable, this
  is a feature not a bug.
- **Stale cache data** — `ephemeral` lasts ~5 minutes on Anthropic. Long
  idle sessions pay the cache-creation premium twice. Document, don't fix.
- **Test pollution** — mock providers in tests must accept the
  `cache_control` markers without choking. Verify in test harness.

### 6.8 Success criteria

- New config flag `llm.prompt_cache` defaults to `true`
- A run against an Anthropic model emits `llm.complete` events with
  `cache_creation_tokens > 0` on turn 1 and `cache_read_tokens > 0` on
  turn 2+
- All existing tests still pass
- New unit tests for marker application and eligibility pass

---

## 7. Bottom-line targets

| Metric | Phase 9 v1 close | Phase 9 v2 target |
|---|---|---|
| Tests passing | 507 | ~553–563 (+46 to +56) |
| Ruff errors | 38 | 29 (28 E402 + 1 acceptable margin) |
| CI status | none | green on `main` |
| Commits on `main` | 12 | +12 to +14 |
| Push to remote | no | no |

## 8. Execution order

1. **E1** (CI) — 2-step:
   1. Ruff cleanup commit (fix F401, N806, SIM105)
   2. Workflow file commit
2. **D3** (4 tests) — one commit per test, or grouped (TBD by implementer)
3. **B1.5** — enum first, then 5 family commits
4. **G** (prompt caching) — single workstream, ~3 commits:
   1. Helper functions + unit tests (`_apply_cache_markers`,
      `_supports_prompt_cache`)
   2. Wire into `LiteLLMProvider.complete` + observability fields
   3. Config knob + integration test

Each major step ends with a `pytest` + `ruff check` verification before
moving to the next. Sub-agent driven development for heavier code commits;
trivial edits inline.

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| CI fails on Linux for macOS-specific tests | Fix the test as part of E1; do not skip silently |
| Multi-tool turn test reveals mock provider limitation | Extend the mock; small test-only change |
| B1.5 large diff hard to review | Family-by-family commits (≤6 files each) |
| Mapping a pre-existing error to wrong category | Document the mapping in code comments at each site; reviewer catches |
| New ruff rule triggered by new code | Fix immediately; do not let baseline drift |
| Prompt caching markers break a non-Anthropic provider | Eligibility gate (`_supports_prompt_cache`) keeps the markers off unsupported models |
| Cache observability fields missing on non-Anthropic providers | `getattr(..., 0) or 0` fallback ensures the field is always present (zero) |

## 10. Open questions (to resolve during implementation)

1. **`session.end` failure marker** — Should `agent.run()` emit a different
   event (`session.failed`?) or annotate `session.end` with a status field?
   Decide when implementing D3 test 2.
2. **Empty messages behaviour** — Observe litellm's default before deciding
   whether to add validation in `LiteLLMProvider`.
3. **CI cache key** — Default `setup-uv` behaviour vs. explicit
   `cache-dependency-glob`? Trial defaults first.
4. **Cache marker placement on tools** — Anthropic docs are ambiguous on
   whether to mark each tool or only the last one. We mark the last one
   (so the entire `tools` array becomes a single cache block) and verify
   empirically with `cache_creation_tokens` on first call.

---

## 11. Approval

Design approved by user (FR session, 2026-04-22). Workstream G (prompt
caching) added in second pass after external code-scanner review surfaced
the missing token cache as a real cost gap. Next step: invoke
`writing-plans` skill to produce the detailed implementation plan in
`docs/plans/2026-04-22-norn-phase9-v2-implementation.md`.
