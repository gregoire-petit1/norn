# Norn Phase 8 — Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans OR superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add structured-logs foundation (6 events, dual-sink console+JSONL, session correlation, cost estimation) via structlog, with zero coupling to engines.

**Architecture:** New `src/norn/observability/` module (events, logger, processors, sinks) wired via `init_logging()` at CLI startup. Session UUID via `contextvars` auto-injected. Instrumentation at 6 points (agent, llm, router×2, tool, permission). See `docs/plans/2026-04-21-norn-phase8-observability-design.md`.

**Tech Stack:** structlog 24.1+, litellm (existing, for `completion_cost`), rich (existing, for console rendering), Python 3.11+ contextvars, Pydantic v2, pytest.

**Reference:** [Phase 8 Design Doc](./2026-04-21-norn-phase8-observability-design.md)

---

## Pre-flight

**Step 0.1: Verify branch and clean state**

```bash
cd ~/norn
git status
git log --oneline -5
```

Expected: clean working tree, HEAD at `1aa7c16` (design doc commit).

**Step 0.2: Add structlog dependency**

Edit `pyproject.toml`: add `"structlog>=24.1"` to `dependencies`.

```bash
uv sync
```

Expected: structlog installed, lockfile updated.

**Step 0.3: Commit dep**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add structlog>=24.1 dependency for Phase 8"
```

---

## Task 1 — Foundation: LoggingConfig + EventName

**Files:**
- Modify: `src/norn/core/config.py` (add `LoggingConfig`, add `logging` field to `NornConfig`)
- Create: `src/norn/observability/__init__.py`
- Create: `src/norn/observability/events.py`
- Modify: `configs/default.yaml` (add `logging:` block)
- Create: `tests/test_observability/__init__.py`
- Create: `tests/test_observability/test_events.py`
- Create: `tests/test_core/test_logging_config.py`

### Step 1.1: Write failing test for LoggingConfig defaults

Create `tests/test_core/test_logging_config.py`:

```python
"""Tests for LoggingConfig."""
from norn.core.config import LoggingConfig, NornConfig


def test_logging_config_defaults():
    cfg = LoggingConfig()
    assert cfg.enabled is True
    assert cfg.level == "INFO"
    assert cfg.output == "both"
    assert cfg.file_dir == "~/.norn/logs"
    assert cfg.include_cost is True
    assert "api_key" in cfg.redact_keys


def test_norn_config_includes_logging():
    cfg = NornConfig()
    assert cfg.logging.enabled is True
    assert cfg.logging.level == "INFO"


def test_logging_config_rejects_invalid_level():
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        LoggingConfig(level="TRACE")


def test_logging_config_rejects_invalid_output():
    import pydantic
    import pytest
    with pytest.raises(pydantic.ValidationError):
        LoggingConfig(output="syslog")
```

### Step 1.2: Run test — expect failure

```bash
uv run pytest tests/test_core/test_logging_config.py -v
```

Expected: 4 FAILED (`LoggingConfig` not importable).

### Step 1.3: Implement LoggingConfig

Edit `src/norn/core/config.py`, add imports and class after existing models (before `NornConfig`):

```python
# Add to imports at top
from typing import Literal

# Add class before NornConfig
class LoggingConfig(BaseModel):
    enabled: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    output: Literal["console", "file", "both"] = "both"
    file_dir: str = "~/.norn/logs"
    include_cost: bool = True
    redact_keys: list[str] = Field(
        default_factory=lambda: ["api_key", "authorization", "token", "password"]
    )
```

Add field to `NornConfig`:

```python
class NornConfig(BaseModel):
    # ... existing fields
    logging: LoggingConfig = LoggingConfig()
```

### Step 1.4: Run tests — expect pass

```bash
uv run pytest tests/test_core/test_logging_config.py -v
```

Expected: 4 PASSED.

### Step 1.5: Write failing test for EventName enum

Create `tests/test_observability/__init__.py` (empty).

Create `tests/test_observability/test_events.py`:

```python
"""Tests for EventName enum."""
from norn.observability.events import EventName


def test_event_names_stable():
    # Stable contract for downstream log consumers
    assert EventName.AGENT_RUN == "agent.run"
    assert EventName.LLM_COMPLETE == "llm.complete"
    assert EventName.TOOL_CALL == "tool.call"
    assert EventName.PERMISSION_DECISION == "permission.decision"
    assert EventName.ROUTING_DECISION == "routing.decision"
    assert EventName.FALLBACK == "fallback"


def test_event_name_is_str():
    # StrEnum so structlog can serialize directly
    assert isinstance(EventName.AGENT_RUN, str)
    assert EventName.AGENT_RUN.value == "agent.run"
```

### Step 1.6: Run test — expect failure

```bash
uv run pytest tests/test_observability/test_events.py -v
```

Expected: 2 FAILED (module not found).

### Step 1.7: Implement events.py

Create `src/norn/observability/__init__.py`:

```python
"""Structured observability for Norn."""
from norn.observability.events import EventName

__all__ = ["EventName"]
```

Create `src/norn/observability/events.py`:

```python
"""Event name constants for structured logging."""
from __future__ import annotations

from enum import StrEnum


class EventName(StrEnum):
    """Canonical event names emitted by Norn's observability layer.

    Treat values as a stable contract: downstream log consumers
    (jq queries, dashboards, metrics) depend on exact strings.
    """

    AGENT_RUN = "agent.run"
    LLM_COMPLETE = "llm.complete"
    TOOL_CALL = "tool.call"
    PERMISSION_DECISION = "permission.decision"
    ROUTING_DECISION = "routing.decision"
    FALLBACK = "fallback"
```

### Step 1.8: Run tests — expect pass

```bash
uv run pytest tests/test_observability/test_events.py -v
```

Expected: 2 PASSED.

### Step 1.9: Update default.yaml

Edit `configs/default.yaml`, append:

```yaml
logging:
  enabled: true
  level: "INFO"          # DEBUG | INFO | WARNING | ERROR
  output: "both"         # console | file | both
  file_dir: "~/.norn/logs"
  include_cost: true
```

### Step 1.10: Verify yaml loads

```bash
uv run python -c "from norn.core.config import NornConfig; c = NornConfig.load(); print(c.logging)"
```

Expected: prints `LoggingConfig(enabled=True, level='INFO', ...)`.

### Step 1.11: Run full test suite — no regressions

```bash
uv run pytest
```

Expected: 420 existing + 6 new = 426 PASSED.

### Step 1.12: Commit

```bash
git add src/norn/core/config.py src/norn/observability/ tests/test_core/test_logging_config.py tests/test_observability/ configs/default.yaml
git commit -m "feat(observability): add LoggingConfig and EventName foundation"
```

---

## Task 2 — Logger setup: structlog config + session contextvar + processors

**Files:**
- Create: `src/norn/observability/logger.py`
- Create: `src/norn/observability/processors.py`
- Create: `tests/test_observability/test_processors.py`
- Create: `tests/test_observability/test_logger.py`
- Modify: `src/norn/observability/__init__.py`

### Step 2.1: Write failing test for session_id contextvar

Create `tests/test_observability/test_logger.py`:

```python
"""Tests for logger setup and session_id contextvar."""
import asyncio

import pytest

from norn.core.config import LoggingConfig
from norn.observability.logger import (
    get_logger,
    init_logging,
    new_session,
    session_id_var,
)


@pytest.fixture(autouse=True)
def reset_session():
    """Reset session_id_var between tests."""
    token = session_id_var.set(None)
    yield
    session_id_var.reset(token)


def test_new_session_generates_uuid():
    sid = new_session()
    assert isinstance(sid, str)
    assert len(sid) == 36  # UUID4 format


def test_new_session_sets_contextvar():
    sid = new_session()
    assert session_id_var.get() == sid


def test_session_id_propagates_across_await():
    sid_holder = {}

    async def inner():
        sid_holder["inner"] = session_id_var.get()

    async def outer():
        sid = new_session()
        sid_holder["outer"] = sid
        await inner()

    asyncio.run(outer())
    assert sid_holder["outer"] == sid_holder["inner"]


def test_init_logging_idempotent(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    init_logging(cfg)  # Should not raise or duplicate handlers


def test_get_logger_returns_bound_logger():
    log = get_logger("test")
    assert hasattr(log, "info")
    assert hasattr(log, "bind")


def test_init_logging_disabled_no_op(tmp_path):
    cfg = LoggingConfig(enabled=False, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    log = get_logger("test")
    log.info("test.event", foo="bar")
    # No file should be created
    files = list(tmp_path.iterdir())
    assert files == []
```

### Step 2.2: Write failing test for processors

Create `tests/test_observability/test_processors.py`:

```python
"""Tests for structlog processors."""
import pytest

from norn.observability.logger import new_session, session_id_var
from norn.observability.processors import (
    add_session_id,
    add_timestamp_iso,
    redact_secrets,
)


@pytest.fixture(autouse=True)
def reset_session():
    token = session_id_var.set(None)
    yield
    session_id_var.reset(token)


def test_add_session_id_when_set():
    sid = new_session()
    out = add_session_id(None, "info", {"event": "test"})
    assert out["session_id"] == sid


def test_add_session_id_absent_when_unset():
    out = add_session_id(None, "info", {"event": "test"})
    assert "session_id" not in out


def test_add_timestamp_iso_format():
    out = add_timestamp_iso(None, "info", {"event": "test"})
    assert "timestamp" in out
    # ISO-8601 UTC, e.g. "2026-04-21T14:23:01.123456Z"
    assert out["timestamp"].endswith("Z")
    assert "T" in out["timestamp"]


def test_redact_secrets_strips_sensitive_fields():
    event = {
        "event": "llm.complete",
        "api_key": "sk-secret",
        "authorization": "Bearer xyz",
        "token": "abc",
        "password": "hunter2",
        "model": "gpt-4",
    }
    out = redact_secrets(None, "info", event, redact_keys=["api_key", "authorization", "token", "password"])
    assert out["api_key"] == "[REDACTED]"
    assert out["authorization"] == "[REDACTED]"
    assert out["token"] == "[REDACTED]"
    assert out["password"] == "[REDACTED]"
    assert out["model"] == "gpt-4"  # untouched
    assert out["event"] == "llm.complete"


def test_redact_secrets_case_insensitive():
    event = {"API_KEY": "secret", "Authorization": "bearer"}
    out = redact_secrets(None, "info", event, redact_keys=["api_key", "authorization"])
    assert out["API_KEY"] == "[REDACTED]"
    assert out["Authorization"] == "[REDACTED]"
```

### Step 2.3: Run tests — expect failures

```bash
uv run pytest tests/test_observability/test_logger.py tests/test_observability/test_processors.py -v
```

Expected: all FAILED (imports missing).

### Step 2.4: Implement processors.py

Create `src/norn/observability/processors.py`:

```python
"""Custom structlog processors."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from norn.observability.events import EventName

# Imported lazily in add_session_id to avoid circular import
from norn.observability.logger import session_id_var


def add_session_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject current session_id (if any) into the event."""
    sid = session_id_var.get()
    if sid is not None:
        event_dict["session_id"] = sid
    return event_dict


def add_timestamp_iso(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject ISO-8601 UTC timestamp ending in 'Z'."""
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    # Python emits "+00:00"; normalize to trailing "Z" for log ecosystems
    event_dict["timestamp"] = now.replace("+00:00", "Z")
    return event_dict


def make_redact_processor(redact_keys: list[str]):
    """Factory: bind redact keys at init_logging time."""
    def redact(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return redact_secrets(logger, method_name, event_dict, redact_keys=redact_keys)
    return redact


def redact_secrets(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
    redact_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Replace sensitive field values with [REDACTED] (case-insensitive match)."""
    keys = redact_keys or []
    lowered = [k.lower() for k in keys]
    for key in list(event_dict.keys()):
        if any(pattern in key.lower() for pattern in lowered):
            event_dict[key] = "[REDACTED]"
    return event_dict


def make_cost_processor(enabled: bool):
    """Factory: emit cost_usd on llm.complete events when enabled."""
    def add_cost(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        if not enabled:
            return event_dict
        if event_dict.get("event") != EventName.LLM_COMPLETE.value:
            return event_dict
        try:
            from litellm import completion_cost
            cost = completion_cost(
                model=event_dict.get("model", ""),
                prompt_tokens=event_dict.get("prompt_tokens", 0),
                completion_tokens=event_dict.get("completion_tokens", 0),
            )
            event_dict["cost_usd"] = round(float(cost), 6)
        except Exception:
            event_dict["cost_usd"] = None
        return event_dict
    return add_cost
```

### Step 2.5: Implement logger.py

Create `src/norn/observability/logger.py`:

```python
"""structlog configuration and session management."""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from norn.core.config import LoggingConfig

session_id_var: ContextVar[str | None] = ContextVar("norn_session_id", default=None)

_configured: bool = False


def new_session() -> str:
    """Generate a new session UUID and bind to contextvar."""
    sid = str(uuid.uuid4())
    session_id_var.set(sid)
    return sid


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger for the given module."""
    return structlog.get_logger(name)


def init_logging(config: LoggingConfig, cli_level_override: str | None = None) -> None:
    """Configure structlog + stdlib logging for Norn.

    Idempotent: safe to call multiple times; subsequent calls replace handlers.
    If config.enabled is False, logging is silenced (no handlers attached).
    """
    global _configured

    # Avoid duplicate handler registration on re-init
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    if not config.enabled:
        root.setLevel(logging.CRITICAL + 1)  # silence everything
        structlog.configure(
            processors=[structlog.dev.ConsoleRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL + 1),
            cache_logger_on_first_use=True,
        )
        _configured = True
        return

    # Resolve level: CLI override > env > config
    import os
    level_str = (
        cli_level_override
        or os.environ.get("NORN_LOG_LEVEL")
        or config.level
    ).upper()
    level = getattr(logging, level_str, logging.INFO)
    root.setLevel(level)

    # Lazy imports to avoid circulars at module load
    from norn.observability.processors import (
        add_session_id,
        add_timestamp_iso,
        make_cost_processor,
        make_redact_processor,
    )
    from norn.observability.sinks import (
        build_console_handler,
        build_file_handler,
    )

    # Shared processor chain (runs before final rendering)
    shared_processors: list = [
        structlog.stdlib.add_log_level,
        add_timestamp_iso,
        add_session_id,
        make_cost_processor(config.include_cost),
        make_redact_processor(config.redact_keys),
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Attach sinks
    if config.output in {"console", "both"}:
        root.addHandler(build_console_handler(shared_processors))
    if config.output in {"file", "both"}:
        root.addHandler(build_file_handler(config.file_dir, shared_processors))

    _configured = True
```

Note: this references `sinks.py` which is Task 3. For now, provide a stub:

Create `src/norn/observability/sinks.py` (stub, full impl in Task 3):

```python
"""Output sinks — console (rich) and file (JSONL). Full impl in Task 3."""
from __future__ import annotations

import logging


def build_console_handler(shared_processors: list) -> logging.Handler:
    h = logging.StreamHandler()
    return h


def build_file_handler(file_dir: str, shared_processors: list) -> logging.Handler:
    # Stub: in-memory handler to let Task 2 tests pass
    return logging.NullHandler()
```

### Step 2.6: Update __init__.py

Edit `src/norn/observability/__init__.py`:

```python
"""Structured observability for Norn."""
from norn.observability.events import EventName
from norn.observability.logger import (
    get_logger,
    init_logging,
    new_session,
    session_id_var,
)

__all__ = [
    "EventName",
    "get_logger",
    "init_logging",
    "new_session",
    "session_id_var",
]
```

### Step 2.7: Run Task 2 tests — expect pass

```bash
uv run pytest tests/test_observability/test_logger.py tests/test_observability/test_processors.py -v
```

Expected: all PASSED (~15 tests).

### Step 2.8: Run full suite — no regressions

```bash
uv run pytest
```

Expected: all PASSED.

### Step 2.9: Commit

```bash
git add src/norn/observability/ tests/test_observability/
git commit -m "feat(observability): add structlog logger, session contextvar, processors"
```

---

## Task 3 — Sinks: console (rich) + file (JSONL rotating)

**Files:**
- Modify: `src/norn/observability/sinks.py` (replace stubs with real impl)
- Create: `tests/test_observability/test_sinks.py`

### Step 3.1: Write failing tests

Create `tests/test_observability/test_sinks.py`:

```python
"""Tests for console and file sinks."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from norn.core.config import LoggingConfig
from norn.observability.logger import get_logger, init_logging, new_session


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_file_sink_writes_jsonl(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    sid = new_session()
    log = get_logger("test")
    log.info("agent.run", phase="start")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = tmp_path / f"{today}.jsonl"
    assert file.exists()
    events = _read_jsonl(file)
    assert len(events) == 1
    assert events[0]["event"] == "agent.run"
    assert events[0]["phase"] == "start"
    assert events[0]["session_id"] == sid


def test_file_sink_multiple_events(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    log = get_logger("test")
    log.info("e1", a=1)
    log.info("e2", b=2)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events = _read_jsonl(tmp_path / f"{today}.jsonl")
    assert [e["event"] for e in events] == ["e1", "e2"]


def test_file_sink_creates_dir(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(deep))
    init_logging(cfg)
    new_session()
    get_logger("test").info("hello")
    assert deep.exists()


def test_file_sink_expanduser(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = LoggingConfig(enabled=True, output="file", file_dir="~/logs")
    init_logging(cfg)
    new_session()
    get_logger("test").info("hello")
    assert (tmp_path / "logs").exists()


def test_console_sink_captures_event(capfd, tmp_path):
    cfg = LoggingConfig(enabled=True, output="console", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    get_logger("test").info("tool.call", tool_name="read_file")
    out = capfd.readouterr()
    combined = out.out + out.err
    assert "tool.call" in combined
    assert "read_file" in combined


def test_both_output_writes_to_file_and_console(capfd, tmp_path):
    cfg = LoggingConfig(enabled=True, output="both", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    get_logger("test").info("agent.run", phase="end")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert (tmp_path / f"{today}.jsonl").exists()
    out = capfd.readouterr()
    assert "agent.run" in (out.out + out.err)
```

### Step 3.2: Run — expect failures

```bash
uv run pytest tests/test_observability/test_sinks.py -v
```

Expected: all FAILED (stub impl doesn't write).

### Step 3.3: Implement real sinks.py

Replace `src/norn/observability/sinks.py`:

```python
"""Output sinks — console (rich) and file (JSONL rotating daily)."""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog


def build_console_handler(shared_processors: list) -> logging.Handler:
    """rich-backed colored console renderer for dev."""
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)
    return handler


class _DailyRotatingFileHandler(logging.Handler):
    """Writes JSONL, rotates by UTC date into <dir>/YYYY-MM-DD.jsonl."""

    def __init__(self, dir_path: Path):
        super().__init__()
        self.dir_path = dir_path
        self._current_date: str | None = None
        self._fh = None

    def _current_file(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._fh is not None:
                self._fh.close()
            self.dir_path.mkdir(parents=True, exist_ok=True)
            path = self.dir_path / f"{today}.jsonl"
            self._fh = path.open("a", encoding="utf-8")
            self._current_date = today
        return self.dir_path / f"{today}.jsonl"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._current_file()
            msg = self.format(record)
            self._fh.write(msg + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def build_file_handler(file_dir: str, shared_processors: list) -> logging.Handler:
    """Daily-rotating JSONL file handler."""
    dir_path = Path(file_dir).expanduser()
    dir_path.mkdir(parents=True, exist_ok=True)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = _DailyRotatingFileHandler(dir_path)
    handler.setFormatter(formatter)
    return handler
```

### Step 3.4: Run Task 3 tests — expect pass

```bash
uv run pytest tests/test_observability/test_sinks.py -v
```

Expected: 6 PASSED.

### Step 3.5: Run full suite

```bash
uv run pytest
```

Expected: all PASSED.

### Step 3.6: Commit

```bash
git add src/norn/observability/sinks.py tests/test_observability/test_sinks.py
git commit -m "feat(observability): implement console (rich) and file (JSONL rotating) sinks"
```

---

## Task 4 — CLI integration: init_logging + --verbose + env var

**Files:**
- Modify: `src/norn/cli/main.py` (call `init_logging` + `new_session` at each command entry; add `--verbose` flag)
- Create: `tests/test_cli/test_logging_integration.py`

### Step 4.1: Inspect current CLI structure

```bash
uv run python -c "import typer; from norn.cli.main import app; print([c.name for c in app.registered_commands])"
```

Expected: list of commands (`run`, `dream`, `coordinator`, ...).

### Step 4.2: Write failing integration test

Create `tests/test_cli/test_logging_integration.py`:

```python
"""Tests for CLI ↔ logging integration."""
import json
from datetime import datetime, timezone
from pathlib import Path


def test_run_command_initializes_logging(tmp_path, monkeypatch):
    """Running a command writes a session_id-tagged event to JSONL."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Run a lightweight command — check the log file is created
    # This is a smoke test; full E2E is Task 7.
    from norn.core.config import LoggingConfig
    from norn.observability.logger import get_logger, init_logging, new_session

    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path / "logs"))
    init_logging(cfg)
    sid = new_session()
    get_logger(__name__).info("agent.run", phase="start")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = tmp_path / "logs" / f"{today}.jsonl"
    assert file.exists()
    events = [json.loads(line) for line in file.read_text().splitlines()]
    assert events[0]["session_id"] == sid


def test_verbose_flag_sets_debug(tmp_path):
    from norn.core.config import LoggingConfig
    from norn.observability.logger import get_logger, init_logging, new_session

    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg, cli_level_override="DEBUG")
    new_session()
    get_logger("test").debug("debug.event", x=1)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = tmp_path / f"{today}.jsonl"
    events = [json.loads(line) for line in file.read_text().splitlines()]
    assert any(e["event"] == "debug.event" for e in events)


def test_env_var_sets_level(tmp_path, monkeypatch):
    monkeypatch.setenv("NORN_LOG_LEVEL", "DEBUG")
    from norn.core.config import LoggingConfig
    from norn.observability.logger import get_logger, init_logging, new_session

    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path), level="INFO")
    init_logging(cfg)
    new_session()
    get_logger("test").debug("debug.event", x=1)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = tmp_path / f"{today}.jsonl"
    events = [json.loads(line) for line in file.read_text().splitlines()]
    assert any(e["event"] == "debug.event" for e in events)


def test_cli_override_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NORN_LOG_LEVEL", "ERROR")
    from norn.core.config import LoggingConfig
    from norn.observability.logger import get_logger, init_logging, new_session

    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path), level="ERROR")
    init_logging(cfg, cli_level_override="DEBUG")
    new_session()
    get_logger("test").debug("debug.event", x=1)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = tmp_path / f"{today}.jsonl"
    events = [json.loads(line) for line in file.read_text().splitlines()]
    assert any(e["event"] == "debug.event" for e in events)


def test_disabled_logging_no_file(tmp_path):
    from norn.core.config import LoggingConfig
    from norn.observability.logger import get_logger, init_logging, new_session

    cfg = LoggingConfig(enabled=False, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    get_logger("test").info("suppressed", x=1)
    assert list(tmp_path.iterdir()) == []
```

### Step 4.3: Run — expect most to pass (Task 2/3 already enable these)

```bash
uv run pytest tests/test_cli/test_logging_integration.py -v
```

Expected: all PASSED (infra from T2/T3 covers these).

### Step 4.4: Add --verbose flag and init in CLI commands

Read current `src/norn/cli/main.py` to identify each `@app.command()` entry.

For **each** command (`run`, `dream`, `coordinator`, others), add:
1. `verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging")` to signature
2. At the top of the function body:
   ```python
   config = NornConfig.load()
   config.apply_env_overrides()
   init_logging(config.logging, cli_level_override="DEBUG" if verbose else None)
   new_session()
   ```

Refactor to a shared helper if the commands already share `NornConfig.load()`:

```python
# At top of cli/main.py
from norn.observability.logger import init_logging, new_session

def _bootstrap(verbose: bool = False) -> NornConfig:
    config = NornConfig.load()
    config.apply_env_overrides()
    init_logging(config.logging, cli_level_override="DEBUG" if verbose else None)
    new_session()
    return config
```

Then in each command: `config = _bootstrap(verbose)`.

### Step 4.5: Run CLI smoke test

```bash
uv run norn --help
uv run norn run --help
```

Expected: `--verbose` / `-v` visible.

### Step 4.6: Run full test suite

```bash
uv run pytest
```

Expected: all PASSED (including pre-existing CLI tests).

### Step 4.7: Commit

```bash
git add src/norn/cli/main.py tests/test_cli/test_logging_integration.py
git commit -m "feat(cli): wire init_logging + --verbose flag into CLI commands"
```

---

## Task 5 — Instrument core: agent.run, llm.complete, routing.decision, fallback

**Files:**
- Modify: `src/norn/core/agent.py` (wrap `AgentLoop.run`)
- Modify: `src/norn/core/llm.py` (instrument `LiteLLMProvider.complete`)
- Modify: `src/norn/core/router.py` (emit routing.decision + fallback)
- Create: `tests/test_observability/test_instrumentation_core.py`

### Step 5.1: Read current agent/llm/router signatures

```bash
uv run python -c "from norn.core.agent import AgentLoop; import inspect; print(inspect.getsource(AgentLoop.run))"
uv run python -c "from norn.core.llm import LiteLLMProvider; import inspect; print(inspect.getsource(LiteLLMProvider.complete))"
uv run python -c "from norn.core.router import RouterProvider; import inspect; print(inspect.getsource(RouterProvider.complete))"
```

### Step 5.2: Write failing tests

Create `tests/test_observability/test_instrumentation_core.py`:

```python
"""Integration tests: core engines emit expected events."""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from norn.core.config import LoggingConfig
from norn.observability.logger import init_logging, new_session


def _today_file(dir_: Path) -> Path:
    return dir_ / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"


def _events(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@pytest.fixture
def log_dir(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path), include_cost=False)
    init_logging(cfg)
    new_session()
    return tmp_path


@pytest.mark.asyncio
async def test_llm_provider_emits_llm_complete(log_dir):
    from norn.core.llm import LiteLLMProvider

    with patch("norn.core.llm.acompletion", new=AsyncMock()) as mock:
        mock.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi", tool_calls=None), finish_reason="stop")],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        provider = LiteLLMProvider(model="gpt-4", provider="openai")
        await provider.complete(messages=[{"role": "user", "content": "hi"}])

    events = _events(_today_file(log_dir))
    llm_events = [e for e in events if e["event"] == "llm.complete"]
    assert len(llm_events) == 1
    assert llm_events[0]["prompt_tokens"] == 10
    assert llm_events[0]["completion_tokens"] == 5
    assert llm_events[0]["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_router_emits_routing_decision(log_dir):
    # TODO: construct RouterProvider with mock tiers, call complete, assert event
    # This test scaffolding assumes RouterProvider(...) API from Phase 7.
    pass  # placeholder — expand once reviewing router.py structure


@pytest.mark.asyncio
async def test_router_emits_fallback_on_error(log_dir):
    # TODO: mock first-tier failure, assert fallback event
    pass  # placeholder


@pytest.mark.asyncio
async def test_agent_run_emits_start_and_end(log_dir):
    # TODO: construct AgentLoop with mock provider, run, assert 2 agent.run events
    pass  # placeholder


@pytest.mark.asyncio
async def test_agent_run_records_duration(log_dir):
    # TODO: verify duration_ms >= 0 in end event
    pass
```

Note: the `pass` placeholders will be filled in once the implementer reads the current router/agent signatures. Mark them `@pytest.mark.skip(reason="fill in after reviewing signatures")` if needed during first pass.

### Step 5.3: Run — expect 1 failure (the concrete LLM test)

```bash
uv run pytest tests/test_observability/test_instrumentation_core.py -v
```

Expected: 1 FAILED (llm event not emitted), 4 skipped/placeholder.

### Step 5.4: Instrument LiteLLMProvider.complete

Edit `src/norn/core/llm.py`. Around the `complete()` method body:

```python
# At top of file
import time
from norn.observability import EventName, get_logger

_log = get_logger(__name__)

# In complete(), wrap the acompletion call:
async def complete(self, messages, tools=None, **kwargs):
    start = time.monotonic()
    try:
        response = await acompletion(
            model=self._full_model_name(),
            messages=messages,
            tools=tools,
            **self._merged_kwargs(kwargs),
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        usage = getattr(response, "usage", None)
        _log.info(
            EventName.LLM_COMPLETE,
            tier=None,  # set by RouterProvider when applicable
            provider=self.provider,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason,
        )
        return self._parse_response(response)
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        _log.error(
            EventName.LLM_COMPLETE,
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
```

Adapt exact field access to current `LiteLLMProvider` impl.

### Step 5.5: Instrument RouterProvider

Edit `src/norn/core/router.py`. In `_select_tier()`:

```python
_log.info(
    EventName.ROUTING_DECISION,
    chosen_tier=tier_name,
    score=score,
    signals=signals_dict,
)
```

In `complete()`'s fallback except block:

```python
_log.warning(
    EventName.FALLBACK,
    from_tier=current_tier,
    to_tier=next_tier,
    error_type=type(e).__name__,
    error_message=str(e),
)
```

Also: when the router's underlying LLM call emits `llm.complete`, add `tier=current_tier` via `bind()` or by passing through. Simplest: in router, wrap LLM call with `log.bind(tier=tier_name)` context using `structlog.contextvars`.

### Step 5.6: Instrument AgentLoop.run

Edit `src/norn/core/agent.py`:

```python
import time
from norn.observability import EventName, get_logger

_log = get_logger(__name__)

async def run(self, ...):
    start = time.monotonic()
    _log.info(EventName.AGENT_RUN, phase="start")
    success = False
    error = None
    try:
        result = await self._do_run(...)
        success = True
        return result
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _log.info(
            EventName.AGENT_RUN,
            phase="end",
            duration_ms=int((time.monotonic() - start) * 1000),
            success=success,
            error=error,
        )
```

### Step 5.7: Fill in placeholder tests with concrete impls

Once the structure is clear from steps 5.4-5.6, remove the `pass` placeholders and write the real assertions.

### Step 5.8: Run Task 5 tests

```bash
uv run pytest tests/test_observability/test_instrumentation_core.py -v
```

Expected: all PASSED.

### Step 5.9: Run full suite — check for regressions

```bash
uv run pytest
```

Expected: all 420 existing + new tests PASSED.

### Step 5.10: Commit

```bash
git add src/norn/core/ tests/test_observability/test_instrumentation_core.py
git commit -m "feat(observability): instrument AgentLoop, LiteLLMProvider, RouterProvider with structured events"
```

---

## Task 6 — Instrument periphery: tool.call, permission.decision

**Files:**
- Modify: `src/norn/tools/registry.py` (instrument `execute`)
- Modify: `src/norn/permissions/checker.py` (instrument `check`)
- Create: `tests/test_observability/test_instrumentation_periphery.py`

### Step 6.1: Write failing tests

Create `tests/test_observability/test_instrumentation_periphery.py`:

```python
"""Integration tests: tool registry and permission checker emit events."""
import json
from datetime import datetime, timezone

import pytest

from norn.core.config import LoggingConfig
from norn.observability.logger import init_logging, new_session


def _events(dir_):
    f = dir_ / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


@pytest.fixture
def log_dir(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    return tmp_path


@pytest.mark.asyncio
async def test_tool_registry_emits_tool_call_on_success(log_dir):
    # TODO: construct ToolRegistry with a simple tool, invoke, assert event
    pass


@pytest.mark.asyncio
async def test_tool_registry_emits_tool_call_on_failure(log_dir):
    # TODO: tool raises, assert success=False, error captured
    pass


def test_permission_checker_emits_decision(log_dir):
    # TODO: invoke checker, assert permission.decision event
    pass


def test_permission_checker_emits_on_deny(log_dir):
    pass


def test_session_id_shared_across_periphery(log_dir):
    # Combined tool + permission events share same session_id
    pass
```

Same as T5: fill in placeholders after reading actual signatures.

### Step 6.2: Instrument tools/registry.py

```python
# Around execute()
import time
from norn.observability import EventName, get_logger
_log = get_logger(__name__)

async def execute(self, name, args, ...):
    start = time.monotonic()
    success = False
    error = None
    try:
        result = await self._actual_execute(name, args, ...)
        success = True
        return result
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _log.info(
            EventName.TOOL_CALL,
            tool_name=name,
            duration_ms=int((time.monotonic() - start) * 1000),
            success=success,
            error=error,
        )
```

### Step 6.3: Instrument permissions/checker.py

```python
from norn.observability import EventName, get_logger
_log = get_logger(__name__)

def check(self, tool_name, ...) -> bool:
    decision = self._compute_decision(...)
    _log.info(
        EventName.PERMISSION_DECISION,
        tool_name=tool_name,
        risk_level=risk,
        mode=self.mode.value,
        granted=decision.granted,
        reason=decision.reason,
    )
    return decision.granted
```

### Step 6.4: Fill in test assertions, run

```bash
uv run pytest tests/test_observability/test_instrumentation_periphery.py -v
```

Expected: all PASSED.

### Step 6.5: Run full suite

```bash
uv run pytest
```

Expected: all PASSED.

### Step 6.6: Commit

```bash
git add src/norn/tools/registry.py src/norn/permissions/checker.py tests/test_observability/test_instrumentation_periphery.py
git commit -m "feat(observability): instrument tool registry and permission checker"
```

---

## Task 7 — E2E integration tests

**Files:**
- Create: `tests/test_observability/test_e2e.py`

### Step 7.1: Write E2E tests

Create `tests/test_observability/test_e2e.py`:

```python
"""End-to-end observability tests: full command → 6 event types with correlated session_id."""
import json
from datetime import datetime, timezone

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from norn.core.config import LoggingConfig
from norn.observability.logger import init_logging, new_session


def _events(dir_):
    f = dir_ / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


@pytest.mark.asyncio
async def test_full_run_emits_all_expected_events(tmp_path):
    """A complete agent run emits the 6 core event types with a single session_id."""
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    sid = new_session()

    # TODO: orchestrate a minimal AgentLoop run with mocked LLM producing 1 tool call
    # then assert events contain: agent.run(start), agent.run(end),
    # llm.complete (>=1), tool.call (>=1), permission.decision (>=1)
    # All sharing session_id == sid.

    events = _events(tmp_path)
    assert all(e["session_id"] == sid for e in events)
    event_names = {e["event"] for e in events}
    assert "agent.run" in event_names
    assert "llm.complete" in event_names
    # etc.


@pytest.mark.asyncio
async def test_cost_included_when_enabled(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path), include_cost=True)
    init_logging(cfg)
    new_session()
    # TODO: emit llm.complete with known model, assert cost_usd present


def test_cost_absent_when_disabled(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path), include_cost=False)
    init_logging(cfg)
    new_session()
    # TODO: emit llm.complete, assert "cost_usd" not in event


def test_redaction_e2e(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    from norn.observability.logger import get_logger
    get_logger("test").info("test.event", api_key="sk-secret", model="gpt-4")
    events = _events(tmp_path)
    assert events[0]["api_key"] == "[REDACTED]"
    assert events[0]["model"] == "gpt-4"


def test_disabled_logging_e2e(tmp_path):
    cfg = LoggingConfig(enabled=False, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    from norn.observability.logger import get_logger
    get_logger("test").info("should.not.appear")
    assert list(tmp_path.iterdir()) == []
```

### Step 7.2: Fill in TODO orchestrations

Implementer reviews `AgentLoop` + mocks `LiteLLMProvider` to return a deterministic sequence (1 tool call then final answer). Full E2E exercise.

### Step 7.3: Run Task 7 tests

```bash
uv run pytest tests/test_observability/test_e2e.py -v
```

Expected: all PASSED.

### Step 7.4: Final full suite + ruff check

```bash
uv run pytest
uv run ruff check src/ tests/
```

Expected:
- All tests pass (420 + ~47 = ~467)
- Ruff: **no new errors** (27×E402 + 4×N806 + 1×F401 pre-existing, unchanged)

### Step 7.5: Commit

```bash
git add tests/test_observability/test_e2e.py
git commit -m "test(observability): add end-to-end integration tests for full event pipeline"
```

### Step 7.6: Final Phase 8 summary commit

Update `docs/plans/2026-04-21-norn-phase8-observability-design.md` acceptance criteria — tick all boxes.

```bash
git add docs/plans/2026-04-21-norn-phase8-observability-design.md
git commit -m "docs: mark Phase 8 acceptance criteria as complete"
```

---

## Final verification

**REQUIRED SUB-SKILL:** Use superpowers:verification-before-completion before declaring Phase 8 done.

Run:
```bash
uv run pytest -v
uv run ruff check src/ tests/
git log --oneline main..HEAD
```

Expected:
- All tests pass
- No new ruff errors
- ~8-10 commits in Phase 8 (one per task + pre-flight + acceptance)

---

## Rollback plan

If any task introduces regressions:
```bash
git reset --hard <commit-before-that-task>
```
Each task is an atomic commit — safe to revert individually.

---

## Notes for the executing agent

1. **TYPE_CHECKING imports:** use `from __future__ import annotations` + `if TYPE_CHECKING:` for `LoggingConfig` in `logger.py` to avoid circular imports.
2. **StrEnum:** all new enums use `StrEnum` (Python 3.11+), per AGENTS.md convention.
3. **Field(default_factory=...):** for mutable defaults in Pydantic models, per AGENTS.md.
4. **Ruff pre-existing errors:** do NOT touch the 32 pre-existing errors (27×E402, 4×N806, 1×F401). New code must be ruff-clean.
5. **Async-safety:** `contextvars` propagates across `await`. If wrapping sync code in `asyncio.to_thread`, use `contextvars.copy_context()`.
6. **Test isolation:** each test resets `session_id_var` via autouse fixture to avoid leakage.
7. **Litellm cost:** some free models return `None` or raise — the `add_cost_processor` must fail-open with `cost_usd = None`.
8. **Session-driven development:** each task starts with a **fresh subagent** + **code reviewer** pass. See superpowers:subagent-driven-development.
