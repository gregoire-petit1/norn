# Norn — Phase 9 v2 Design (Robustness & Quality)

**Date:** 2026-04-22
**Status:** Approved, ready for implementation planning
**Predecessor:** Phase 9 v1 (`docs/plans/2026-04-22-norn-phase9-v1-implementation.md`)
**Tracker:** `docs/plans/2026-04-21-norn-phase9-observability-followups.md`

---

## 1. Goal

Solidify the foundations laid by Phases 8 and 9 v1 before stacking more
observability features. Three workstreams, executed linearly:

1. **E1** — Minimal CI on GitHub Actions (protective net first).
2. **D3** — Four end-to-end coverage gaps (denied path, agent.run failure,
   multi-tool turn, empty messages).
3. **B1.5** — `ToolErrorType` taxonomy + application to the ~30 per-tool
   `ToolResult(error=...)` sites.

The ordering is deliberate: ship the CI net first so that every subsequent
test addition and refactor is automatically validated.

## 2. Out of scope

Deferred to Phase 9 v3 or later:

- F1.5 (`norn logs tail` advanced flags: `--follow`, `--since`, `--level`,
  `--lines`, multi-day, pager)
- F2 (`norn logs stats` with DuckDB)
- F3 (`norn cost report`)
- C2 (`_provider_from_model` heuristic)
- D4 + E2 (Windows portability)

Rationale: each of these is a feature; Phase 9 v2 deliberately invests in
non-feature quality work.

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

## 6. Bottom-line targets

| Metric | Phase 9 v1 close | Phase 9 v2 target |
|---|---|---|
| Tests passing | 507 | ~545–555 (+38 to +48) |
| Ruff errors | 38 | 29 (28 E402 + 1 acceptable margin) |
| CI status | none | green on `main` |
| Commits on `main` | 12 | +10 to +12 |
| Push to remote | no | no |

## 7. Execution order

1. **E1** (CI) — 2-step:
   1. Ruff cleanup commit (fix F401, N806, SIM105)
   2. Workflow file commit
2. **D3** (4 tests) — one commit per test, or grouped (TBD by implementer)
3. **B1.5** — enum first, then 5 family commits

Each major step ends with a `pytest` + `ruff check` verification before
moving to the next. Sub-agent driven development (implementer → spec
reviewer → code quality reviewer) for the heavier code commits; trivial
edits inline.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| CI fails on Linux for macOS-specific tests | Fix the test as part of E1; do not skip silently |
| Multi-tool turn test reveals mock provider limitation | Extend the mock; small test-only change |
| B1.5 large diff hard to review | Family-by-family commits (≤6 files each) |
| Mapping a pre-existing error to wrong category | Document the mapping in code comments at each site; reviewer catches |
| New ruff rule triggered by new code | Fix immediately; do not let baseline drift |

## 9. Open questions (to resolve during implementation)

1. **`session.end` failure marker** — Should `agent.run()` emit a different
   event (`session.failed`?) or annotate `session.end` with a status field?
   Decide when implementing D3 test 2.
2. **Empty messages behaviour** — Observe litellm's default before deciding
   whether to add validation in `LiteLLMProvider`.
3. **CI cache key** — Default `setup-uv` behaviour vs. explicit `cache-dependency-glob`?
   Trial defaults first.

---

## 10. Approval

Design approved by user (FR session, 2026-04-22). Next step: invoke
`writing-plans` skill to produce the detailed implementation plan in
`docs/plans/2026-04-22-norn-phase9-v2-implementation.md`.
