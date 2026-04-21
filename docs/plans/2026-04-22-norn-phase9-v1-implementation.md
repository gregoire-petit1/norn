# Phase 9 v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Land 8 Phase 9 v1 items (Tier 1 quick wins A1/A2/A3/C1/D1/D2 + B1 error_type granularity + F1 `norn logs tail`) on `main` with zero test regression and zero ruff baseline drift.

**Architecture:** Linear execution, one task per commit. Quick wins first (~1.5h) to validate the workflow, then B1 (~30min, touches `agent.py` + tests), then F1 (~2h, new module + state-file plumbing in `logger.py`). All tasks remain local (no push).

**Tech Stack:** Python 3.11+, structlog, Typer + Rich, pytest, ruff, uv. Conventions: `from __future__ import annotations`, `Annotated[...]` Typer options, `StrEnum`, `TYPE_CHECKING`, conventional commits.

**Baselines (HEAD=`b804ffa`):**
- Tests: **485 passed** (target after this plan: **498 passed**)
- Ruff: **37 errors** (28 E402 + 1 F401 + 4 N806 + 4 SIM105) — must not change
- Branch: `main`, working tree clean, no remote push

**References:**
- Phase 9 tracker: `docs/plans/2026-04-21-norn-phase9-observability-followups.md`
- Phase 8 design: `docs/plans/2026-04-21-norn-phase8-observability-design.md`
- F1 design: `docs/plans/2026-04-22-norn-phase9-logs-tail-design.md`

---

## Task 1 (A1): Update Phase 8 design doc — `tool.call` error schema

**Files:**
- Modify: `docs/plans/2026-04-21-norn-phase8-observability-design.md` (§3.2 `tool.call` block)

**Step 1: Locate the current `tool.call` schema in §3.2**

Run: `rg -n 'tool\.call' docs/plans/2026-04-21-norn-phase8-observability-design.md`
Expected: a JSON-like block listing `{event: "tool.call", ..., error: null}` (or similar).

**Step 2: Replace the `error: null` field with the harmonised pair**

Edit the schema block so failure events show:
```
"success": false,
"error_type": "<exception class name or category>",
"error_message": "<single-line message>"
```
And success events show neither key (omit, do not set null).

Add a one-sentence note: "Failure events emit the harmonised `error_type` / `error_message` pair (see Phase 9 follow-up A1)."

**Step 3: Verify no other doc references `error: null` for `tool.call`**

Run: `rg -n 'tool\.call' docs/plans/`
Expected: no orphan stale references.

**Step 4: Commit**

```bash
git add docs/plans/2026-04-21-norn-phase8-observability-design.md
git commit -m "docs(phase8): harmonise tool.call error schema (A1)"
```

---

## Task 2 (A2): Synthesise `permission.decision.reason` for auto-approve paths

**Files:**
- Modify: `src/norn/permissions/checker.py:65` (and the auto-approve branches in `_apply_mode`)
- Test: `tests/test_permissions/` (extend or add a focused test)

**Step 1: Inspect current behaviour**

Run: `rg -n 'PermissionDecision\(approved=True' src/norn/permissions/checker.py`
Expected: at least 2 sites where `approved=True` is returned without `reason` (e.g. yolo mode L78-79, auto-approve safe paths).

**Step 2: Write the failing test**

Create or extend in `tests/test_permissions/test_checker.py`:
```python
@pytest.mark.asyncio
async def test_yolo_mode_logs_reason_yolo(caplog) -> None:
    checker = PermissionChecker(mode=PermissionMode.YOLO, ...)
    await checker.check(PermissionRequest(tool_name="x", risk_level="low", arguments={}))
    # capture the structlog event and assert reason == "yolo"
    ...

@pytest.mark.asyncio
async def test_safe_auto_approve_logs_reason_auto_approved(caplog) -> None:
    ...
```
(Use the same caplog/structlog capture pattern as existing tests in `tests/test_permissions/`.)

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_permissions/test_checker.py -k 'reason' -v`
Expected: FAIL — `reason` is `None` in the emitted event.

**Step 4: Implement — synthesise reason in auto-approve branches**

In `src/norn/permissions/checker.py`, update each `PermissionDecision(approved=True)` site to carry an explicit reason:
- yolo mode → `reason="yolo"`
- safe auto-approve in strict/normal → `reason="auto_approved"`
- any other implicit `True` → `reason="<descriptive_snake_case>"`

The log call at line 59-66 already passes `reason=decision.reason`, so synthesising on the decision side is enough.

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_permissions/ -v`
Expected: PASS, no regression in existing permission tests.

**Step 6: Run full suite + ruff**

Run: `uv run pytest -q && uv run ruff check src tests 2>&1 | tail -3`
Expected: 487+ passed (485 baseline + ≥2 new tests), ruff still 37 errors.

**Step 7: Commit**

```bash
git add src/norn/permissions/checker.py tests/test_permissions/
git commit -m "fix(permissions): synthesise reason for auto-approve paths (A2)"
```

---

## Task 3 (A3): Re-tag stale `TODO(T6)` in processors.py

**Files:**
- Modify: `src/norn/observability/processors.py:54`

**Step 1: Verify the comment**

Run: `sed -n '53,55p' src/norn/observability/processors.py`
Expected: `# TODO(T6): add recursive redaction if events become nested.`

**Step 2: Re-tag to `TODO(phase9-v2)` with rationale**

Replace the line with:
```
nested. TODO(phase9-v2): add recursive redaction if events become nested.
Tracked in docs/plans/2026-04-21-norn-phase9-observability-followups.md (item A3).
```

**Step 3: Verify no other stale `TODO(T<n>)` markers exist**

Run: `rg -n 'TODO\(T[0-9]+\)' src/`
Expected: no matches.

**Step 4: Commit**

```bash
git add src/norn/observability/processors.py
git commit -m "chore(observability): re-tag stale TODO(T6) to phase9-v2 (A3)"
```

---

## Task 4 (C1): Skip `_bootstrap_logging` for `norn version`

**Files:**
- Modify: `src/norn/cli/main.py:451-457`
- Test: `tests/test_cli/` (existing test file or new)

**Step 1: Write the failing test**

In `tests/test_cli/test_logging_integration.py` (or a new `test_version.py`):
```python
def test_version_command_does_not_init_logging(monkeypatch) -> None:
    called = {"v": False}
    def fake_bootstrap(*a, **kw):
        called["v"] = True
    monkeypatch.setattr("norn.cli.main._bootstrap_logging", fake_bootstrap)
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "norn 0.1.0" in result.stdout
    assert called["v"] is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli/ -k 'version_command_does_not_init' -v`
Expected: FAIL — `called["v"] is True`.

**Step 3: Implement — strip the bootstrap call**

Edit `src/norn/cli/main.py:451-457`:
```python
@app.command()
def version() -> None:
    """Show Norn version (no config load, no logging init)."""
    console.print("norn 0.1.0")
```
Remove the `verbose` param entirely (it was only used for `_bootstrap_logging`). If a future need arises, the version string can come from `importlib.metadata`.

**Step 4: Run test + full suite**

Run: `uv run pytest -q`
Expected: 488+ passed, 0 failed.

**Step 5: Run ruff**

Run: `uv run ruff check src tests 2>&1 | tail -3`
Expected: 37 errors (no delta).

**Step 6: Commit**

```bash
git add src/norn/cli/main.py tests/test_cli/
git commit -m "perf(cli): skip logging bootstrap for norn version (C1)"
```

---

## Task 5 (D1): Patch `litellm.acompletion` per-instance in concurrent test

**Files:**
- Modify: `tests/test_observability/test_e2e.py:486-545` (`test_concurrent_agent_runs_do_not_cross_contaminate_sessions`)

**Step 1: Read current implementation**

Run: `sed -n '486,545p' tests/test_observability/test_e2e.py`
Note where `patch("litellm.acompletion", ...)` is used and how concurrent tasks share that patch.

**Step 2: Identify the LLM client instances**

Each agent in the test owns a `LiteLLMProvider` (or similar) with a method that calls `litellm.acompletion`. We want to monkeypatch the bound method on each instance, not the global module.

**Step 3: Refactor the test to patch per instance**

Replace the global `patch("litellm.acompletion", ...)` with per-instance `monkeypatch.setattr(provider, "_acompletion_callable", side_effect_fn)` (or whatever the call boundary is). Each agent gets its own AsyncMock with deterministic, distinct responses so cross-contamination would be visible.

If the provider hard-codes `litellm.acompletion`, introduce a thin attribute (`self._completion = litellm.acompletion`) used at call time, then patch `self._completion` per instance. Document the indirection with a one-line comment.

**Step 4: Run the test**

Run: `uv run pytest tests/test_observability/test_e2e.py::test_concurrent_agent_runs_do_not_cross_contaminate_sessions -v`
Expected: PASS, deterministic output regardless of task scheduling.

**Step 5: Run full suite**

Run: `uv run pytest -q`
Expected: same count as before D1 (no new tests added), 0 failed.

**Step 6: Commit**

```bash
git add tests/test_observability/test_e2e.py src/norn/llm/  # if provider touched
git commit -m "test(observability): patch litellm per-instance for concurrent test (D1)"
```

---

## Task 6 (D2): Assert numeric cost for known-priced model

**Files:**
- Modify: `tests/test_observability/test_e2e.py:351-385` (`test_cost_included_when_enabled`)

**Step 1: Read current assertion**

Run: `sed -n '370,380p' tests/test_observability/test_e2e.py`
Current: `assert ev["cost_usd"] is None or isinstance(ev["cost_usd"], int | float)` — too lenient.

**Step 2: Pick a model with a known litellm price entry**

Use `gpt-4o-mini` or another model present in `litellm.model_cost`. Adjust the mocked response usage (`prompt_tokens`, `completion_tokens`) so the computed cost > 0.

**Step 3: Tighten the assertion**

Replace lines 376-377 with:
```python
assert "cost_usd" in ev, "cost_usd key missing when include_cost=True"
assert isinstance(ev["cost_usd"], (int, float)), "cost_usd must be numeric for a priced model"
assert ev["cost_usd"] > 0, f"cost_usd must be positive for {MODEL_NAME}, got {ev['cost_usd']}"
```

**Step 4: Run the test**

Run: `uv run pytest tests/test_observability/test_e2e.py::test_cost_included_when_enabled -v`
Expected: PASS with strict numeric assertion.

**Step 5: Run full suite**

Run: `uv run pytest -q`
Expected: 488+ passed, 0 failed.

**Step 6: Commit**

```bash
git add tests/test_observability/test_e2e.py
git commit -m "test(observability): assert numeric cost for priced model (D2)"
```

---

## Task 7 (B1): Granular `error_type` for `tool.call` failures

**Files:**
- Modify: `src/norn/core/agent.py:128-143` (and `_execute_tool_inner` if needed)
- Modify: `src/norn/core/tools.py` or wherever `ToolResult` lives (only if a sibling field is needed)
- Test: `tests/test_observability/test_instrumentation_core.py`

**Step 1: Inventory current error origin sites**

Run: `rg -n 'ToolResult\(error=' src/norn/`
Expected: at least 3 origin sites:
- `agent.py:149` → `f"Unknown tool: {call.name}"` → should map to `error_type="UnknownTool"`
- `permissions/...` integration → permission denied → `error_type="PermissionDenied"`
- generic exception path → `error_type=type(exc).__name__`

**Step 2: Decide plumbing strategy**

Two options:
- **(a) Parse prefix** of `result.error` string in `_execute_tool` (simple, no schema change, fragile if messages drift)
- **(b) Add `error_type: str | None = None` field on `ToolResult`** and set it at each origin site (proper, schema change, ~6 touch points)

**Choose (b)** — it's a minor field, fully backwards-compatible (default `None`), and removes the brittle prefix-matching. If `error_type` is `None` at log time, fall back to `"ToolExecutionError"` literal.

**Step 3: Write failing tests**

In `tests/test_observability/test_instrumentation_core.py`, add:
```python
async def test_tool_call_error_type_unknown_tool(...): ...   # invokes a non-existent tool
async def test_tool_call_error_type_permission_denied(...): ... # strict mode + risky tool
async def test_tool_call_error_type_exception_uses_class_name(...): ... # tool raises ValueError
```
Each asserts the `tool.call` event payload has the expected `error_type`.

**Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_observability/test_instrumentation_core.py -k 'error_type' -v`
Expected: FAIL — all return `"ToolExecutionError"`.

**Step 5: Implement**

a. Add field to `ToolResult`:
```python
@dataclass
class ToolResult:
    output: ... = None
    error: str | None = None
    error_type: str | None = None  # NEW
```

b. Update origin sites:
- `agent.py:149`: `return ToolResult(error=f"Unknown tool: {call.name}", error_type="UnknownTool")`
- Permission-denied path (likely in `_execute_tool_inner`): `error_type="PermissionDenied"`
- Generic exception wrapper (around tool invocation): `error_type=type(exc).__name__`

c. Update `agent.py:136`:
```python
payload["error_type"] = result.error_type or "ToolExecutionError"
payload["error_message"] = result.error
```

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_observability/test_instrumentation_core.py -v`
Expected: PASS, including the 3 new tests.

**Step 7: Run full suite + ruff**

Run: `uv run pytest -q && uv run ruff check src tests 2>&1 | tail -3`
Expected: 491+ passed, ruff 37 errors (no delta).

**Step 8: Commit**

```bash
git add src/norn/core/agent.py src/norn/core/tools.py tests/test_observability/test_instrumentation_core.py
git commit -m "feat(observability): granular error_type for tool.call failures (B1)"
```

---

## Task 8 (F1): `norn logs tail` command

**Reference:** Full design at `docs/plans/2026-04-22-norn-phase9-logs-tail-design.md`. Read it before starting.

**Files:**
- Create: `src/norn/cli/logs.py`
- Modify: `src/norn/observability/logger.py` (add `_STATE_DIR`, `_LAST_SESSION_FILE`, `_persist_last_session`)
- Modify: `src/norn/cli/main.py` (mount `logs_app` sub-Typer)
- Test: `tests/test_cli/test_logs.py` (NEW, 11 tests)
- Test: `tests/test_observability/test_logger.py` (extend, +2 tests)

### Sub-task 8a: Plumbing `last_session` state file

**Step 1: Write failing tests**

In `tests/test_observability/test_logger.py`, add:
```python
def test_new_session_persists_last_session_file(tmp_path, monkeypatch):
    monkeypatch.setattr("norn.observability.logger._STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(
        "norn.observability.logger._LAST_SESSION_FILE",
        tmp_path / "state" / "last_session",
    )
    sid = new_session()
    assert (tmp_path / "state" / "last_session").read_text(encoding="utf-8").strip() == sid


def test_new_session_fails_open_on_oserror(tmp_path, monkeypatch):
    def boom(*a, **kw): raise OSError("disk full")
    monkeypatch.setattr("pathlib.Path.mkdir", boom)
    sid = new_session()  # must not raise
    assert sid  # uuid still returned
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_observability/test_logger.py -k 'persists_last_session or fails_open_on_oserror' -v`
Expected: FAIL — module symbols don't exist.

**Step 3: Implement in `src/norn/observability/logger.py`**

Add after line 17:
```python
_STATE_DIR = Path.home() / ".norn" / "state"
_LAST_SESSION_FILE = _STATE_DIR / "last_session"
```

Add `from pathlib import Path` at top (after stdlib imports, before structlog).

Modify `new_session()`:
```python
def new_session() -> str:
    """Generate a new session UUID, bind to contextvar, persist to state file."""
    sid = str(uuid.uuid4())
    session_id_var.set(sid)
    _persist_last_session(sid)
    return sid


def _persist_last_session(sid: str) -> None:
    """Best-effort write of the latest session id; never raises."""
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _LAST_SESSION_FILE.with_suffix(".tmp")
        tmp.write_text(sid, encoding="utf-8")
        tmp.replace(_LAST_SESSION_FILE)
    except OSError:
        # fail-open: state file is a convenience, not a correctness requirement
        pass
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_observability/test_logger.py -v`
Expected: PASS, no regression.

**Step 5: Verify nothing else broke (state file leakage in tests)**

Run: `uv run pytest -q`
Expected: 491+ passed. If failures appear due to state file persistence across tests, add a session-scoped autouse fixture in `tests/conftest.py` redirecting `_STATE_DIR` to `tmp_path_factory`.

**Step 6: Commit**

```bash
git add src/norn/observability/logger.py tests/test_observability/test_logger.py tests/conftest.py
git commit -m "feat(observability): persist last_session for CLI alias resolution (F1 plumbing)"
```

### Sub-task 8b: `norn logs tail` command + tests

**Step 1: Write failing tests**

Create `tests/test_cli/test_logs.py` with the 11 tests from §6.1 of the F1 design doc. Use `CliRunner` and seed JSONL fixtures via a helper:

```python
def _seed_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(d) for d in lines) + "\n", encoding="utf-8")
```

Pattern per test (see existing `tests/test_cli/test_logging_integration.py` for config monkeypatching):
```python
def test_tail_default_emits_all_events(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    today = datetime.now(UTC).date().isoformat()
    _seed_jsonl(log_dir / f"{today}.jsonl", [
        {"event": "agent.run", "level": "info", "timestamp": "2026-04-22T10:00:00Z", "session_id": "abc"},
        {"event": "tool.call", "level": "info", "timestamp": "2026-04-22T10:00:01Z", "session_id": "abc"},
        {"event": "llm.complete", "level": "info", "timestamp": "2026-04-22T10:00:02Z", "session_id": "abc"},
    ])
    monkeypatch.setattr(
        "norn.cli.logs._today_log_path",
        lambda: log_dir / f"{today}.jsonl",
    )
    result = CliRunner().invoke(app, ["logs", "tail"])
    assert result.exit_code == 0
    assert result.stdout.count("\n") == 3
```

**Step 2: Run tests to verify they fail (collection error or import error)**

Run: `uv run pytest tests/test_cli/test_logs.py -v`
Expected: FAIL — `norn.cli.logs` does not exist.

**Step 3: Implement `src/norn/cli/logs.py`**

```python
"""`norn logs tail` and future log-inspection commands."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Iterator

import typer
from rich.console import Console
from rich.text import Text

from norn.core.config import NornConfig
from norn.observability.logger import _LAST_SESSION_FILE

logs_app = typer.Typer(name="logs", help="Inspect Norn structured logs.")
_console = Console()
_err_console = Console(stderr=True)
_SESSION_DISPLAY_LEN = 8
_HIDDEN_KEYS = {"event", "level", "timestamp", "logger"}
_LEVEL_STYLE = {
    "debug": "dim",
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
}


def _today_log_path() -> Path:
    """Resolve today's JSONL log file path from config."""
    cfg = NornConfig.load()
    return Path(cfg.logging.file_dir).expanduser() / f"{datetime.now(UTC).date().isoformat()}.jsonl"


def _resolve_session(value: str) -> str:
    """Resolve --session, accepting 'last'/'current' aliases."""
    if value in {"last", "current"}:
        try:
            return _LAST_SESSION_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise typer.BadParameter("No previous session found.") from exc
    return value


def _iter_filtered_events(
    path: Path,
    events_filter: set[str] | None,
    session_filter: str | None,
) -> Iterator[tuple[str, dict]]:
    """Yield (raw_line, parsed_dict) pairs that pass filters. Tracks malformed via stderr."""
    malformed = 0
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if events_filter and parsed.get("event") not in events_filter:
                continue
            if session_filter and parsed.get("session_id") != session_filter:
                continue
            yield raw, parsed
    if malformed:
        _err_console.print(f"Skipped {malformed} malformed line(s).", style="yellow")


def _format_rich_line(ev: dict) -> Text:
    """Compact one-line rich rendering of an event."""
    ts = ev.get("timestamp", "")
    hhmmss = ts[11:19] if len(ts) >= 19 else ts
    level = (ev.get("level") or "info").lower()
    name = ev.get("event", "<no-event>")
    style = _LEVEL_STYLE.get(level, "white")
    text = Text()
    text.append(f"{hhmmss} ", style="dim")
    text.append(f"[{level.upper()}] ", style=style)
    text.append(name, style="bold")
    rest = sorted((k, v) for k, v in ev.items() if k not in _HIDDEN_KEYS)
    for k, v in rest:
        if k == "session_id" and isinstance(v, str):
            v = v[:_SESSION_DISPLAY_LEN]
        if isinstance(v, (dict, list)):
            v = json.dumps(v, separators=(",", ":"))
        text.append(f" {k}={v}", style="dim")
    return text


@logs_app.command("tail")
def tail(
    event: Annotated[
        list[str] | None,
        typer.Option("--event", "-e", help="Filter by event name (repeatable, OR logic)."),
    ] = None,
    session: Annotated[
        str | None,
        typer.Option(
            "--session",
            "-s",
            help="Filter by session UUID, or 'last'/'current' for the most recent session.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit raw JSONL passthrough instead of rich formatting."),
    ] = False,
) -> None:
    """Tail today's Norn log file with optional filters."""
    path = _today_log_path()
    if not path.exists():
        _err_console.print(f"No logs for today ({path}).")
        raise typer.Exit(0)
    events_filter = set(event) if event else None
    session_filter = _resolve_session(session) if session else None
    for raw, parsed in _iter_filtered_events(path, events_filter, session_filter):
        if json_output:
            sys.stdout.write(raw + "\n")
        else:
            _console.print(_format_rich_line(parsed))
```

**Step 4: Mount sub-app in `main.py`**

Add to `src/norn/cli/main.py` (after `app = typer.Typer(...)` declaration around line 48):
```python
from norn.cli.logs import logs_app
app.add_typer(logs_app, name="logs")
```
(Place the import in the existing import block to respect E402 baseline; the `app.add_typer` call goes right after `app` is defined.)

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli/test_logs.py -v`
Expected: 11 PASS.

**Step 6: Run full suite + ruff**

Run: `uv run pytest -q && uv run ruff check src tests 2>&1 | tail -3`
Expected: **498 passed**, ruff 37 errors (no delta).

**Step 7: Smoke-test manually**

Run: `uv run norn logs --help`
Expected: Typer prints `tail` sub-command help.

Run: `uv run norn logs tail --help`
Expected: 3 options listed (`--event`, `--session`, `--json`).

**Step 8: Commit**

```bash
git add src/norn/cli/logs.py src/norn/cli/main.py tests/test_cli/test_logs.py
git commit -m "feat(cli): add 'norn logs tail' command (F1)"
```

---

## Final Verification

**Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: **498 passed**, 0 failed, 0 errors.

**Step 2: Ruff baseline**

Run: `uv run ruff check src tests 2>&1 | tail -3`
Expected: **37 errors** (28 E402 + 1 F401 + 4 N806 + 4 SIM105). No delta.

**Step 3: Commit log review**

Run: `git log --oneline b804ffa..HEAD`
Expected: 8 commits in order:
```
<sha> feat(cli): add 'norn logs tail' command (F1)
<sha> feat(observability): persist last_session for CLI alias resolution (F1 plumbing)
<sha> feat(observability): granular error_type for tool.call failures (B1)
<sha> test(observability): assert numeric cost for priced model (D2)
<sha> test(observability): patch litellm per-instance for concurrent test (D1)
<sha> perf(cli): skip logging bootstrap for norn version (C1)
<sha> chore(observability): re-tag stale TODO(T6) to phase9-v2 (A3)
<sha> fix(permissions): synthesise reason for auto-approve paths (A2)
<sha> docs(phase8): harmonise tool.call error schema (A1)
```

**Step 4: Update Phase 9 tracker**

Mark items A1, A2, A3, B1, C1, D1, D2, F1 as completed in `docs/plans/2026-04-21-norn-phase9-observability-followups.md`. Single commit:

```bash
git add docs/plans/2026-04-21-norn-phase9-observability-followups.md
git commit -m "docs(phase9): mark v1 items complete in tracker"
```

**Step 5: Confirm no push**

Run: `git status && git log -1 --oneline`
Expected: working tree clean, HEAD ahead of `origin/main` by 9+ commits, no `git push` issued.

---

## Definition of Done

- [ ] All 8 tasks committed in order on `main`
- [ ] `pytest -q` → 498 passed
- [ ] `ruff check` → 37 errors (baseline)
- [ ] Tracker updated, marked as Phase 9 v1 complete
- [ ] No push to remote (per user instruction)
- [ ] Manual smoke: `uv run norn logs tail --help` works
