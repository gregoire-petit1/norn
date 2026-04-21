# Norn — Phase 9 v2 Implementation Plan (Robustness & Quality)

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task.

**Goal:** Solidify foundations with CI, coverage gaps, and a unified `ToolErrorType` taxonomy before stacking more observability features.

**Architecture:** Three workstreams executed linearly: E1 (CI net) → D3 (4 E2E tests) → B1.5 (taxonomy + ~30 sites). Each workstream is split into bite-sized commits. All work on `main`, no push.

**Tech Stack:** Python 3.11, uv, pytest, ruff, GitHub Actions, Pydantic, Typer, litellm.

**Design doc:** `docs/plans/2026-04-22-norn-phase9-v2-design.md`

---

## Pre-flight checks

Before starting any task, verify:

```bash
git status                          # working tree clean
git log --oneline -1                # HEAD is 2668d63 or later
uv run pytest 2>&1 | tail -1        # 507 passed
uv run ruff check . 2>&1 | tail -3  # 38 errors baseline
```

Expected: clean tree, 507 tests pass, 38 ruff errors.

---

## Workstream E1 — Minimal CI

### Task E1.1: Ruff cleanup (F401, N806, SIM105)

**Files:** TBD by inventory

**Step 1: Inventory the 9 non-E402 violations**

Run: `uv run ruff check . --select F401,N806,SIM105`

Expected output: 9 errors across some files, with `[*]` autofixable markers
on the SIM105 ones (`contextlib.suppress` rewrite).

**Step 2: Apply autofix where safe**

Run: `uv run ruff check . --select SIM105 --fix`

Expected: 4 fewer SIM105 errors. `try/except: pass` becomes `with contextlib.suppress(...)`.

**Step 3: Manual fixes for F401 (1) and N806 (4)**

For F401: remove the unused import.

For N806 (lowercase variable names in functions): rename to `snake_case`.
Each rename is a single-file change. Verify the renamed variable is local
to its function (no module-level usage).

**Step 4: Verify ruff baseline shrunk**

Run: `uv run ruff check . 2>&1 | tail -3`

Expected: `Found 29 errors.` (37 original − 9 cleaned + 1 from F1 = 29).

Wait: actual baseline is 38 (37 + 1 F1 drift), so target after cleanup is **29**.

**Step 5: Verify tests still pass**

Run: `uv run pytest 2>&1 | tail -1`

Expected: `507 passed`.

**Step 6: Commit**

```bash
git add -u
git commit -m "chore(ruff): clean up F401/N806/SIM105 violations before CI"
```

---

### Task E1.2: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Verify directory does not exist yet**

Run: `ls .github/workflows/ 2>/dev/null || echo "does not exist"`

Expected: `does not exist` (Norn has no CI yet).

**Step 2: Create the workflow file**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Set up Python 3.11
        run: uv python install 3.11

      - name: Sync dependencies
        run: uv sync --all-extras

      - name: Lint with ruff
        run: uv run ruff check .

      - name: Run tests
        run: uv run pytest
```

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add minimal GitHub Actions workflow (ruff + pytest)"
```

**Step 4: Verify locally that the steps would pass**

Run sequentially:
```bash
uv sync --all-extras 2>&1 | tail -3
uv run ruff check . 2>&1 | tail -3
uv run pytest 2>&1 | tail -1
```

Expected: sync OK, 29 ruff errors, 507 tests pass.

Note: we do not push, so CI does not actually run. The workflow will trigger
on the first future push. This is acceptable per Phase 9 v2 constraints.

---

## Workstream D3 — Coverage gaps

### Task D3.1: Inventory mock provider capabilities

**Files (read-only):**
- `tests/test_observability/test_e2e.py`
- `tests/conftest.py`
- Any existing mock provider in `tests/`

**Step 1: Locate the mock provider**

Run: `rg "class.*Provider" tests/ -l`

Identify the mock used in `test_e2e.py`. Read its `complete()` /
`acompletion()` implementation.

**Step 2: Verify multi-tool_call support**

Check whether the mock can return a response with multiple `tool_calls` in a
single turn. If yes, document; if no, note that D3.3 must extend it first.

**Step 3: Document findings inline (no commit)**

Just take notes for use in D3.2–D3.4.

---

### Task D3.2: Test — Denied path E2E

**Files:**
- Modify: `tests/test_observability/test_e2e.py`

**Step 1: Write the failing test**

```python
async def test_denied_destructive_command_logs_permission_denied(
    tmp_path, mock_provider, ...
):
    """E2E: when user denies a destructive command, tool.call logs
    error_type=PermissionDenied and the agent loop continues cleanly."""
    # Mock provider: turn 1 = bash rm -rf, turn 2 = final text response
    mock_provider.set_turns([
        make_tool_call_turn("bash", {"command": "rm -rf /tmp/foo"}),
        make_text_turn("Understood, command was denied."),
    ])

    # Mock permission handler that denies
    deny_handler = lambda *_args, **_kwargs: PermissionDecision.DENY

    # Run agent
    log_path = tmp_path / "logs.jsonl"
    async with capture_logs(log_path):
        await agent.run(
            user_message="please rm -rf",
            permission_handler=deny_handler,
        )

    # Assertions
    events = parse_jsonl(log_path)
    tool_calls = [e for e in events if e["event"] == "tool.call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["error_type"] == "PermissionDenied"
    assert tool_calls[0]["permission_reason"] in {
        "user_denied",
        "destructive_denied",
    }
    session_ends = [e for e in events if e["event"] == "session.end"]
    assert len(session_ends) == 1
```

Adapt names to the actual fixtures and helpers found in D3.1.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_observability/test_e2e.py::test_denied_destructive_command_logs_permission_denied -v`

Expected: FAIL — likely either fixture missing or assertion mismatch.

**Step 3: Adjust until the test reflects actual production behaviour**

If `error_type` for permission denial is not currently set: that is a
production bug and must be fixed (likely a 1-line tweak in
`src/norn/core/agent.py` permission denial handler). Phase 9 v1's B1
already covers this; verify by inspection.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_observability/test_e2e.py::test_denied_destructive_command_logs_permission_denied -v`

Expected: PASS.

**Step 5: Run full suite**

Run: `uv run pytest 2>&1 | tail -1`

Expected: `508 passed`.

**Step 6: Commit**

```bash
git add tests/test_observability/test_e2e.py
git commit -m "test(observability): add E2E for denied destructive command (D3.1)"
```

---

### Task D3.3: Test — agent.run failure E2E

**Files:**
- Modify: `tests/test_observability/test_e2e.py`
- Possibly: `src/norn/core/agent.py` (if a `session.failed` event needs to be added)

**Step 1: Decide on session lifecycle for failures**

Open question from design §9.1: does the session end with `session.end` +
status field, or with a separate `session.failed` event?

Read `src/norn/core/agent.py` around the `agent.run()` exception handler.
Whatever the current behaviour is, the test asserts it. If no lifecycle
event is emitted on failure, this is a bug — fix it as part of this task
(emit `session.end` with a `status: "failed"` field, or add `session.failed`).

**Step 2: Write the failing test**

```python
async def test_agent_run_failure_emits_session_end_and_propagates(
    tmp_path, mock_provider, ...
):
    """E2E: when the provider raises mid-flight, agent.run propagates
    the exception and the session is closed in the log."""
    mock_provider.set_turns([
        make_tool_call_turn("file_read", {"path": str(tmp_path / "a.txt")}),
        ProviderError("provider 500"),  # raise on turn 2
    ])
    (tmp_path / "a.txt").write_text("hello")

    log_path = tmp_path / "logs.jsonl"
    with pytest.raises(ProviderError, match="provider 500"):
        async with capture_logs(log_path):
            await agent.run(user_message="read a.txt")

    events = parse_jsonl(log_path)
    session_ends = [e for e in events if e["event"] == "session.end"]
    assert len(session_ends) == 1
    # Whatever the failure marker is, assert it
    assert session_ends[0].get("status") == "failed"
```

**Step 3: Run test to verify it fails**

Expected: FAIL.

**Step 4: Implement the fix in `agent.py`**

Wrap the agent loop in try/except, emit `session.end` with `status: "failed"`
on exception, then re-raise. Keep the change minimal.

**Step 5: Run test, then full suite**

Run: `uv run pytest 2>&1 | tail -1`

Expected: `509 passed`.

**Step 6: Commit**

```bash
git add tests/test_observability/test_e2e.py src/norn/core/agent.py
git commit -m "feat(observability): emit session.end with failure status on agent.run error (D3.2)"
```

---

### Task D3.4: Test — Multi-tool turn

**Files:**
- Modify: `tests/test_observability/test_e2e.py`
- Possibly: mock provider helper if it does not support multi-tool_calls

**Step 1: Extend mock provider if needed**

If D3.1 found the mock cannot return multi-tool_call responses, extend it
with a helper like `make_multi_tool_call_turn([(name1, args1), (name2, args2)])`.

**Step 2: Write the failing test**

```python
async def test_multi_tool_turn_executes_in_order_and_logs_separately(
    tmp_path, mock_provider, ...
):
    """E2E: a single LLM turn returning 2 tool_calls runs them sequentially
    and emits 2 distinct tool.call events."""
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")

    mock_provider.set_turns([
        make_multi_tool_call_turn([
            ("file_read", {"path": str(tmp_path / "a.txt")}),
            ("file_read", {"path": str(tmp_path / "b.txt")}),
        ]),
        make_text_turn("Done."),
    ])

    log_path = tmp_path / "logs.jsonl"
    async with capture_logs(log_path):
        await agent.run(user_message="read both files")

    events = parse_jsonl(log_path)
    tool_calls = [e for e in events if e["event"] == "tool.call"]
    assert len(tool_calls) == 2
    # Assert order
    assert "a.txt" in tool_calls[0]["args"]["path"]
    assert "b.txt" in tool_calls[1]["args"]["path"]
    # Assert both succeed
    assert all(tc.get("error_type") is None for tc in tool_calls)
```

**Step 3: Run test, fix what needs fixing, re-run**

Expected after fixes: PASS.

**Step 4: Run full suite**

Run: `uv run pytest 2>&1 | tail -1`

Expected: `510 passed`.

**Step 5: Commit**

```bash
git add tests/test_observability/test_e2e.py
# also include mock extension if added
git commit -m "test(observability): add E2E for multi-tool turn (D3.3)"
```

---

### Task D3.5: Test — Empty messages edge case

**Files:**
- Modify: `tests/test_core/test_llm.py` (or create if missing)

**Step 1: Observe current behaviour first**

Write a one-shot script or interactive test that calls
`LiteLLMProvider.complete(messages=[])` and observe what happens
(litellm error? clean ValueError? silent passthrough?).

**Step 2: Write a test that asserts the documented behaviour**

If litellm raises cleanly: assert `pytest.raises(SomeError)`.
If silent passthrough: add a guard in `LiteLLMProvider.complete()` that
raises `ValueError("messages cannot be empty")` and assert that.

**Step 3: Run test, full suite**

Expected: `511 passed`.

**Step 4: Commit**

```bash
git add tests/test_core/test_llm.py  # + src/norn/core/llm.py if guard added
git commit -m "test(llm): document empty messages behaviour for LiteLLMProvider (D3.4)"
```

---

## Workstream B1.5 — `ToolErrorType` taxonomy

### Task B1.5.1: Introduce `ToolErrorType` enum

**Files:**
- Modify: `src/norn/tools/base.py`
- Modify: `tests/test_tools/test_base.py` (or create)

**Step 1: Write the failing test**

```python
from norn.tools.base import ToolErrorType


def test_tool_error_type_values():
    assert ToolErrorType.FILE_NOT_FOUND.value == "FileNotFound"
    assert ToolErrorType.PERMISSION_DENIED.value == "PermissionDenied"
    assert ToolErrorType.INVALID_ARGUMENT.value == "InvalidArgument"
    assert ToolErrorType.TIMEOUT.value == "Timeout"
    assert ToolErrorType.NETWORK_ERROR.value == "NetworkError"
    assert ToolErrorType.HTTP_ERROR.value == "HttpError"
    assert ToolErrorType.PARSE_ERROR.value == "ParseError"
    assert ToolErrorType.NOT_SUPPORTED.value == "NotSupported"
    assert ToolErrorType.EXECUTION_ERROR.value == "ExecutionError"
    assert ToolErrorType.RESOURCE_EXHAUSTED.value == "ResourceExhausted"


def test_tool_error_type_is_str_enum():
    # StrEnum members ARE strings
    assert ToolErrorType.FILE_NOT_FOUND == "FileNotFound"
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `ToolErrorType` not defined.

**Step 3: Implement the enum**

Add to `src/norn/tools/base.py`:

```python
from enum import StrEnum


class ToolErrorType(StrEnum):
    """Semantic error categories for ToolResult.error_type.

    Used by tool implementations to tag failures with stable, queryable
    categories. Mirrors PermissionDecisionReason (Phase 9 v1).
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

**Step 4: Run test, full suite**

Expected: `513 passed` (+2 for the two new tests).

**Step 5: Commit**

```bash
git add src/norn/tools/base.py tests/test_tools/test_base.py
git commit -m "feat(tools): introduce ToolErrorType enum (B1.5.0)"
```

---

### Task B1.5.2: Inventory the ~30 sites

**Files (read-only):** `src/norn/tools/**/*.py`

**Step 1: Run grep**

Run: `rg "ToolResult\(.*error=" src/norn/tools/ -n`

**Step 2: Categorise each site**

Build a table (in a scratch buffer, not committed) mapping each site to its
target `ToolErrorType` per the design doc §5.2 mapping.

**Step 3: Group by family**

Confirm the 5 families from design §5.3:
- file_* (file_read, file_write, file_edit)
- bash
- grep + glob
- web_* (web_fetch, web_search)
- ML (dataset_inspector, model_*, tensor_inspector)

**No commit**, this is research.

---

### Task B1.5.3: Apply taxonomy to file_* tools

**Files:**
- Modify: `src/norn/tools/file_read.py`, `file_write.py`, `file_edit.py`
- Modify: `tests/test_tools/test_file_read.py`, `test_file_write.py`, `test_file_edit.py`

**Step 1: Write the failing tests** (one per error path per tool)

Example for `file_read`:

```python
async def test_file_read_missing_file_sets_error_type():
    result = await file_read_tool(path="/nonexistent/path/xyz.txt")
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND.value


async def test_file_read_permission_denied_sets_error_type(tmp_path):
    f = tmp_path / "secret.txt"
    f.write_text("data")
    f.chmod(0o000)
    try:
        result = await file_read_tool(path=str(f))
        assert result.error_type == ToolErrorType.PERMISSION_DENIED.value
    finally:
        f.chmod(0o644)
```

Repeat the pattern for `file_write` (path not writable, parent missing) and
`file_edit` (file missing, edit conflict).

**Step 2: Run tests to verify they fail**

Expected: all FAIL.

**Step 3: Update each tool's `ToolResult(error=...)` calls**

In `file_read.py`, replace e.g.:

```python
except FileNotFoundError as e:
    return ToolResult(error=str(e))
```

with:

```python
except FileNotFoundError as e:
    return ToolResult(
        error=str(e),
        error_type=ToolErrorType.FILE_NOT_FOUND.value,
    )
```

Repeat for each error path in the 3 file tools.

**Step 4: Run tests, full suite**

Expected: tests pass, full suite at `~520 passed`.

**Step 5: Commit**

```bash
git add src/norn/tools/file_*.py tests/test_tools/test_file_*.py
git commit -m "feat(tools): apply ToolErrorType to file_* tools (B1.5.1)"
```

---

### Task B1.5.4: Apply taxonomy to bash tool

**Files:**
- Modify: `src/norn/tools/bash.py`
- Modify: `tests/test_tools/test_bash.py`

**Step 1: Write failing tests for the error paths**

- Timeout: command exceeds the timeout → `TIMEOUT`
- Non-zero exit: keep current behaviour but tag → likely `EXECUTION_ERROR`
- Working directory does not exist → `INVALID_ARGUMENT` or `FILE_NOT_FOUND`

**Step 2: Run, fail, implement, pass.**

Same pattern as B1.5.3.

**Step 3: Commit**

```bash
git add src/norn/tools/bash.py tests/test_tools/test_bash.py
git commit -m "feat(tools): apply ToolErrorType to bash tool (B1.5.2)"
```

---

### Task B1.5.5: Apply taxonomy to grep + glob

**Files:**
- Modify: `src/norn/tools/grep.py`, `glob.py`
- Modify: `tests/test_tools/test_grep.py`, `test_glob.py`

Same pattern. Likely error types:
- Invalid regex → `INVALID_ARGUMENT` (grep)
- Path missing → `FILE_NOT_FOUND` (glob, grep)

**Step 1–4: TDD cycle.**

**Step 5: Commit**

```bash
git add src/norn/tools/grep.py src/norn/tools/glob.py tests/test_tools/test_grep.py tests/test_tools/test_glob.py
git commit -m "feat(tools): apply ToolErrorType to grep + glob tools (B1.5.3)"
```

---

### Task B1.5.6: Apply taxonomy to web_* tools

**Files:**
- Modify: `src/norn/tools/web/web_fetch.py`, `web_search.py`
- Modify: `tests/test_tools/test_web/*.py`

Likely error types:
- HTTP 4xx/5xx → `HTTP_ERROR`
- DNS / connection refused → `NETWORK_ERROR`
- Timeout → `TIMEOUT`
- Malformed response → `PARSE_ERROR`

**Step 1–4: TDD cycle.** Mock httpx responses for deterministic tests.

**Step 5: Commit**

```bash
git add src/norn/tools/web/ tests/test_tools/test_web/
git commit -m "feat(tools): apply ToolErrorType to web_* tools (B1.5.4)"
```

---

### Task B1.5.7: Apply taxonomy to ML tools

**Files:**
- Modify: `src/norn/tools/ml/*.py` (dataset_inspector, model_inspector, model_eval, model_card, tensor_inspector, registration)
- Modify: `tests/test_tools/test_ml/*.py`

Likely error types:
- Missing dataset/model file → `FILE_NOT_FOUND`
- Unsupported format → `NOT_SUPPORTED`
- OOM during eval → `RESOURCE_EXHAUSTED`
- Invalid args → `INVALID_ARGUMENT`

**Step 1–4: TDD cycle.**

**Step 5: Commit**

```bash
git add src/norn/tools/ml/ tests/test_tools/test_ml/
git commit -m "feat(tools): apply ToolErrorType to ML tools (B1.5.5)"
```

---

## Workstream G — Prompt caching (litellm `cache_control`)

### Task G.1: Helper functions + unit tests

**Files:**
- Modify: `src/norn/core/llm.py`
- Modify (or create): `tests/test_core/test_llm_caching.py`

**Step 1: Write failing tests**

```python
from norn.core.llm import _apply_cache_markers, _supports_prompt_cache


def test_supports_prompt_cache_anthropic_direct():
    assert _supports_prompt_cache("anthropic/claude-sonnet-4")


def test_supports_prompt_cache_anthropic_via_openrouter():
    assert _supports_prompt_cache("openrouter/anthropic/claude-sonnet-4")


def test_supports_prompt_cache_openai_4o():
    assert _supports_prompt_cache("openai/gpt-4o")


def test_supports_prompt_cache_unsupported():
    assert not _supports_prompt_cache("ollama/qwen2.5-coder:14b")
    assert not _supports_prompt_cache("openrouter/stepfun/step-3.5-flash:free")


def test_apply_cache_markers_disabled_is_noop():
    msgs = [{"role": "system", "content": "you are an agent"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    out_msgs, out_tools = _apply_cache_markers(msgs, tools, enabled=False)
    assert out_msgs == msgs
    assert out_tools == tools


def test_apply_cache_markers_tags_system_prompt():
    msgs = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "hi"},
    ]
    out_msgs, _ = _apply_cache_markers(msgs, None, enabled=True)
    assert isinstance(out_msgs[0]["content"], list)
    assert out_msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out_msgs[0]["content"][0]["text"] == "you are an agent"
    # User message untouched
    assert out_msgs[1] == msgs[1]


def test_apply_cache_markers_tags_last_tool():
    tools = [
        {"type": "function", "function": {"name": "bash"}},
        {"type": "function", "function": {"name": "file_read"}},
    ]
    _, out_tools = _apply_cache_markers([], tools, enabled=True)
    assert out_tools[0] == tools[0]  # untouched
    assert out_tools[1]["cache_control"] == {"type": "ephemeral"}


def test_apply_cache_markers_no_system_no_crash():
    msgs = [{"role": "user", "content": "hi"}]
    out_msgs, _ = _apply_cache_markers(msgs, None, enabled=True)
    assert out_msgs == msgs


def test_apply_cache_markers_empty_messages():
    out_msgs, out_tools = _apply_cache_markers([], None, enabled=True)
    assert out_msgs == []
    assert out_tools is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_llm_caching.py -v`

Expected: FAIL — symbols not defined.

**Step 3: Implement the helpers in `src/norn/core/llm.py`**

```python
_CACHE_ELIGIBLE_PREFIXES = (
    "anthropic/",
    "openrouter/anthropic/",
    "openai/gpt-4o",
    "openai/gpt-4.1",
)


def _supports_prompt_cache(model: str) -> bool:
    """Return True if the model is known to support prompt caching markers.

    Conservative gate: better to skip caching on a supporting model than
    to send unrecognised fields to a fragile provider.
    """
    return model.startswith(_CACHE_ELIGIBLE_PREFIXES)


def _apply_cache_markers(
    messages: list[dict],
    tools: list[dict] | None,
    enabled: bool,
) -> tuple[list[dict], list[dict] | None]:
    """Tag system prompt + last tool schema with cache_control: ephemeral.

    No-op when `enabled` is False. Caller must also gate on
    `_supports_prompt_cache(model)` to avoid sending markers to providers
    that don't understand them.
    """
    if not enabled:
        return messages, tools

    new_messages = list(messages)
    if new_messages and new_messages[0].get("role") == "system":
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
        new_tools = list(tools)
        new_tools[-1] = {
            **new_tools[-1],
            "cache_control": {"type": "ephemeral"},
        }

    return new_messages, new_tools
```

**Step 4: Run tests, full suite**

Run: `uv run pytest 2>&1 | tail -1`

Expected: `~553 passed` (507 baseline + previous v2 tasks + 8 new cache helper tests).

**Step 5: Commit**

```bash
git add src/norn/core/llm.py tests/test_core/test_llm_caching.py
git commit -m "feat(llm): add prompt-cache marker helpers (G.1)"
```

---

### Task G.2: Wire helpers into `LiteLLMProvider.complete` + observability

**Files:**
- Modify: `src/norn/core/llm.py`
- Modify: `tests/test_observability/test_e2e.py` or `tests/test_core/test_llm_caching.py`

**Step 1: Decide on the config plumbing approach**

Two options:
- (a) Add `prompt_cache: bool = True` field on `LiteLLMProvider.__init__`,
  default True
- (b) Read from a module-level flag

Choose **(a)** for testability and explicitness.

**Step 2: Write the failing integration test**

```python
from unittest.mock import AsyncMock, MagicMock

async def test_litellm_provider_emits_cache_observability_fields():
    """When the provider returns cache usage stats, llm.complete event
    carries cache_read_tokens and cache_creation_tokens."""
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ok"
    fake_response.choices[0].message.tool_calls = None
    fake_response.choices[0].finish_reason = "stop"
    fake_response.usage = MagicMock(
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
        cache_read_input_tokens=800,
        cache_creation_input_tokens=200,
    )
    fake_completion = AsyncMock(return_value=fake_response)

    provider = LiteLLMProvider(
        model="anthropic/claude-sonnet-4",
        prompt_cache=True,
        completion_fn=fake_completion,
    )

    # Capture log via existing test harness or an observability fixture
    with capture_log_events() as events:
        await provider.complete(
            messages=[Message(role=Role.SYSTEM, content="you are an agent"),
                      Message(role=Role.USER, content="hi")],
            tools=None,
        )

    llm_events = [e for e in events if e["event"] == "llm.complete"]
    assert len(llm_events) == 1
    assert llm_events[0]["cache_read_tokens"] == 800
    assert llm_events[0]["cache_creation_tokens"] == 200


async def test_litellm_provider_passes_cache_markers_to_litellm():
    """When prompt_cache=True and model is eligible, litellm receives
    messages with cache_control markers."""
    fake_response = _minimal_fake_response()
    fake_completion = AsyncMock(return_value=fake_response)
    provider = LiteLLMProvider(
        model="anthropic/claude-sonnet-4",
        prompt_cache=True,
        completion_fn=fake_completion,
    )
    await provider.complete(
        messages=[Message(role=Role.SYSTEM, content="hello"),
                  Message(role=Role.USER, content="hi")],
        tools=None,
    )
    call_kwargs = fake_completion.await_args.kwargs
    sys_content = call_kwargs["messages"][0]["content"]
    assert isinstance(sys_content, list)
    assert sys_content[0]["cache_control"] == {"type": "ephemeral"}


async def test_litellm_provider_skips_markers_on_unsupported_model():
    """When model is not eligible, no cache_control markers are sent."""
    fake_completion = AsyncMock(return_value=_minimal_fake_response())
    provider = LiteLLMProvider(
        model="ollama/qwen2.5-coder:14b",
        prompt_cache=True,
        completion_fn=fake_completion,
    )
    await provider.complete(
        messages=[Message(role=Role.SYSTEM, content="hello"),
                  Message(role=Role.USER, content="hi")],
        tools=None,
    )
    call_kwargs = fake_completion.await_args.kwargs
    sys_content = call_kwargs["messages"][0]["content"]
    assert sys_content == "hello"  # untouched string
```

(Adapt `_minimal_fake_response` and `capture_log_events` helpers to the
existing test infrastructure; reuse what `test_e2e.py` already exposes.)

**Step 3: Run tests to verify they fail**

Expected: FAIL.

**Step 4: Implement the wiring in `LiteLLMProvider`**

Update `__init__`:

```python
def __init__(
    self,
    model: str,
    api_base: str | None = None,
    *,
    completion_fn: Callable[..., Awaitable[Any]] | None = None,
    prompt_cache: bool = True,
) -> None:
    self.model = model
    self.api_base = api_base
    self._completion_fn = completion_fn
    self._prompt_cache = prompt_cache and _supports_prompt_cache(model)
    litellm.suppress_debug_info = True
```

Update `complete()` to:
- call `_apply_cache_markers(_messages_to_dicts(messages), tool_schemas, self._prompt_cache)`
- pass the marked messages and tools to `completion(**kwargs)`
- after the call, extract `cache_read_input_tokens` and
  `cache_creation_input_tokens` from `usage` and add them to the event:

```python
event["cache_read_tokens"] = (
    getattr(usage, "cache_read_input_tokens", 0) or 0
    if usage is not None else 0
)
event["cache_creation_tokens"] = (
    getattr(usage, "cache_creation_input_tokens", 0) or 0
    if usage is not None else 0
)
```

**Step 5: Run tests, full suite**

Expected: `~556 passed` (+3 integration tests).

**Step 6: Commit**

```bash
git add src/norn/core/llm.py tests/test_core/test_llm_caching.py
git commit -m "feat(llm): wire prompt cache markers into LiteLLMProvider + observability (G.2)"
```

---

### Task G.3: Config knob + verify no regression

**Files:**
- Modify: `src/norn/core/config.py` (add `prompt_cache: bool = True` to LLM config block)
- Modify: `src/norn/core/router.py` (forward `prompt_cache` to underlying providers if router uses LiteLLMProvider)
- Modify: `configs/default.yaml` (document the knob)
- Modify: `src/norn/cli/main.py` `_build_provider` to pass `prompt_cache=config.llm.prompt_cache`

**Step 1: Inventory how `LiteLLMProvider` is constructed**

Run: `rg "LiteLLMProvider\(" src/norn/ -n`

Identify every call site. Likely: `cli/main.py` (`_build_provider`),
`core/router.py` (`build_litellm_provider`).

**Step 2: Write failing test for config plumbing**

```python
def test_config_default_prompt_cache_is_true():
    config = NornConfig.load_default()
    assert config.llm.prompt_cache is True


def test_config_prompt_cache_can_be_disabled():
    config = NornConfig.load_from_dict({
        "llm": {
            "provider": "anthropic",
            "model": "anthropic/claude-sonnet-4",
            "prompt_cache": False,
        },
    })
    assert config.llm.prompt_cache is False
```

**Step 3: Run, fail, implement, pass.**

Add `prompt_cache: bool = True` to the LLM config Pydantic model.
Update `_build_provider` and `build_litellm_provider` to forward the flag.
Document in `configs/default.yaml` with a comment block.

**Step 4: Run full suite**

Expected: `~558 passed`.

**Step 5: Commit**

```bash
git add src/norn/core/config.py src/norn/core/router.py src/norn/cli/main.py configs/default.yaml tests/test_core/test_config.py
git commit -m "feat(config): add llm.prompt_cache knob (G.3)"
```

---

## Final verification

### Task FINAL: Tracker update + full verification

**Files:**
- Modify: `docs/plans/2026-04-21-norn-phase9-observability-followups.md`

**Step 1: Run the full test suite**

Run: `uv run pytest 2>&1 | tail -1`

Expected: ~553–563 passed.

**Step 2: Run ruff**

Run: `uv run ruff check . 2>&1 | tail -3`

Expected: 29 errors (the 28 E402 + nothing new).

**Step 3: Append "Phase 9 v2 — Completed" section to the tracker**

Mirror the structure used for v1: items table with commit SHAs, bottom-line
metrics, follow-ups carried forward to v3.

**Step 4: Commit**

```bash
git add docs/plans/2026-04-21-norn-phase9-observability-followups.md
git commit -m "docs(phase9-v2): mark v2 items complete in tracker"
```

**Step 5: Final state check**

```bash
git log --oneline 2668d63..HEAD
git status
```

Expected: ~10–12 commits since the design doc, working tree clean, no push.

---

## Summary

| Workstream | Tasks | Estimated commits |
|---|---|---|
| E1 | E1.1, E1.2 | 2 |
| D3 | D3.1 (research), D3.2, D3.3, D3.4, D3.5 | 4 |
| B1.5 | B1.5.1, B1.5.2 (research), B1.5.3, B1.5.4, B1.5.5, B1.5.6, B1.5.7 | 6 |
| G | G.1, G.2, G.3 | 3 |
| FINAL | tracker update | 1 |
| **Total** | **~17 tasks** | **~16 commits** |

## Skills referenced

- `@superpowers:test-driven-development` — every code task follows RED-GREEN-REFACTOR
- `@superpowers:subagent-driven-development` — execution model for this plan
- `@superpowers:verification-before-completion` — required before each commit
- `@superpowers:systematic-debugging` — if any test fails unexpectedly
