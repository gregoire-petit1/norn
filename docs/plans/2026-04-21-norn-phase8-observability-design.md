# Norn Phase 8 — Observability Design

**Date:** 2026-04-21
**Status:** Design (pre-implementation)
**Scope:** POC minimal — structured logs foundation
**Estimated size:** ~800 lines, ~45 tests, 7 tasks

---

## 1. Goal

Introduce a **structured-logs foundation** for Norn so every significant runtime event (LLM call, tool invocation, routing decision, permission check, fallback, agent run) emits a machine-readable event with consistent fields. This unlocks:

- Post-mortem debugging (`jq` queries over JSONL files)
- Cost awareness (per-run $ estimate from litellm)
- Fundation for Phase 9 (metrics, traces, dashboards)
- Zero coupling between engines and the logging layer (logger is a cross-cutting concern)

**Non-goals (deferred to Phase 9):**
- CLI reporting commands (`norn logs tail`, `norn logs stats`, `norn cost report`)
- Metrics aggregation (Prometheus-style counters/histograms)
- Distributed tracing (OpenTelemetry spans)
- Dashboards / Grafana integration

---

## 2. Architecture

### 2.1 High-level flow

```
┌──────────────────────────────────────────────────────────────┐
│  CLI command entrypoint (cli/main.py)                        │
│    1. Load config (incl. LoggingConfig)                      │
│    2. init_logging(config, cli_level_override)               │
│    3. session_id = uuid4() → contextvars.set()              │
│    4. Run engine (AgentLoop, DreamEngine, CoordinatorEngine) │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  src/norn/observability/                                     │
│                                                              │
│  logger.py                                                   │
│    • configure_structlog(config)                             │
│    • get_logger(name) → structlog.BoundLogger                │
│    • session_id_var: ContextVar[str | None]                  │
│                                                              │
│  events.py                                                   │
│    • EventName(StrEnum): AGENT_RUN, LLM_COMPLETE, ...        │
│    • helper factories for consistent payloads                │
│                                                              │
│  processors.py                                               │
│    • add_session_id       (reads contextvars)                │
│    • add_timestamp_iso    (UTC ISO-8601)                     │
│    • add_cost_estimate    (for llm.complete events only)     │
│    • redact_secrets       (strip api_keys, env values)       │
│                                                              │
│  sinks.py                                                    │
│    • build_console_renderer() → rich-backed structlog dev    │
│    • build_file_renderer(path) → JSONL writer w/ rotation    │
└──────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
   Console (rich)   ~/.norn/logs/       (pytest capfd / tmp_path)
                    2026-04-21.jsonl
```

### 2.2 Module responsibilities

| Module | Responsibility | LOC estimate |
|--------|----------------|--------------|
| `observability/__init__.py` | Public API re-exports (`get_logger`, `EventName`, `init_logging`) | ~15 |
| `observability/events.py` | `EventName` StrEnum + event schema constants | ~50 |
| `observability/logger.py` | structlog configuration, `init_logging()`, `get_logger()`, `session_id_var` | ~120 |
| `observability/processors.py` | Custom structlog processors (session_id, cost, redaction) | ~100 |
| `observability/sinks.py` | Console (rich) + File (JSONL rotating) renderers | ~120 |

Total new code: ~400 LOC. Tests: ~400 LOC (unit + integration).

### 2.3 Dependency additions

```toml
# pyproject.toml
dependencies = [
    # ... existing
    "structlog>=24.1",
]
```

No other deps needed. `litellm` (already present) provides `completion_cost()`. `rich` (already present) powers dev console output.

---

## 3. Event catalog

### 3.1 EventName enum

```python
# src/norn/observability/events.py
from enum import StrEnum

class EventName(StrEnum):
    AGENT_RUN = "agent.run"
    LLM_COMPLETE = "llm.complete"
    TOOL_CALL = "tool.call"
    PERMISSION_DECISION = "permission.decision"
    ROUTING_DECISION = "routing.decision"
    FALLBACK = "fallback"
```

### 3.2 Event schemas

All events carry automatic fields injected by processors:
- `timestamp`: ISO-8601 UTC
- `session_id`: UUID string (from contextvar)
- `level`: log level (`info`, `warning`, `error`)
- `event`: EventName value

**Per-event fields:**

#### `agent.run`
```json
{
  "event": "agent.run",
  "phase": "start" | "end",
  "duration_ms": 12345,     // end only
  "success": true,           // end only
  "error": "..."             // end only, if failed
}
```

#### `llm.complete`
```json
{
  "event": "llm.complete",
  "tier": "fast" | "standard" | "powerful" | null,
  "provider": "openrouter",
  "model": "stepfun/step-3.5-flash:free",
  "prompt_tokens": 1234,
  "completion_tokens": 567,
  "total_tokens": 1801,
  "latency_ms": 2345,
  "cost_usd": 0.00123,      // nullable, computed via litellm.completion_cost()
  "finish_reason": "stop"
}
```

#### `tool.call`

Success events omit error fields:
```json
{
  "event": "tool.call",
  "tool_name": "read_file",
  "duration_ms": 42,
  "success": true
}
```

Failure events emit the harmonised `error_type` / `error_message` pair
(see Phase 9 follow-up A1):
```json
{
  "event": "tool.call",
  "tool_name": "bash",
  "duration_ms": 17,
  "success": false,
  "error_type": "PermissionDenied",
  "error_message": "Destructive command denied in strict mode"
}
```

#### `permission.decision`
```json
{
  "event": "permission.decision",
  "tool_name": "bash",
  "risk_level": "high",
  "mode": "interactive",
  "granted": true,
  "reason": "user_approved"
}
```

##### Reason taxonomy

The `reason` field uses a closed taxonomy (canonical source: `PermissionDecisionReason`
StrEnum in `src/norn/permissions/models.py`). Updated in Phase 9 follow-up A2 to be
non-null for every decision:

| Value | When emitted |
|---|---|
| `yolo` | yolo mode auto-approval |
| `auto_approved` | auto/interactive mode below the prompt threshold |
| `user_approved` | user approved at the interactive prompt |
| `user_denied` | user rejected at the interactive prompt |
| `destructive_denied` | strict mode rejected a destructive command |
| `prompt_required_no_handler` | prompt would be required but no `prompt_fn` was wired |

The structured event also carries `mode` and `risk_level`, which together with `reason`
form the (mode, risk, decision) triple useful for downstream filtering.

#### `routing.decision`
```json
{
  "event": "routing.decision",
  "chosen_tier": "standard",
  "score": 0.73,
  "signals": {"complexity": 0.6, "context_tokens": 1200}
}
```

#### `fallback`
```json
{
  "event": "fallback",
  "from_tier": "fast",
  "to_tier": "standard",
  "error_type": "TimeoutError",
  "error_message": "..."
}
```

---

## 4. Configuration schema

### 4.1 `LoggingConfig` Pydantic model

```python
# src/norn/core/config.py — ADD
from typing import Literal

class LoggingConfig(BaseModel):
    enabled: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    output: Literal["console", "file", "both"] = "both"
    file_dir: str = "~/.norn/logs"   # expanded via Path.expanduser()
    include_cost: bool = True
    redact_keys: list[str] = Field(
        default_factory=lambda: ["api_key", "authorization", "token", "password"]
    )

class NornConfig(BaseModel):
    # ... existing fields
    logging: LoggingConfig = LoggingConfig()
```

### 4.2 `configs/default.yaml` addition

```yaml
logging:
  enabled: true
  level: "INFO"          # DEBUG | INFO | WARNING | ERROR
  output: "both"         # console | file | both
  file_dir: "~/.norn/logs"
  include_cost: true
```

### 4.3 Level resolution priority (highest → lowest)

1. CLI flag `--verbose` / `-v` → forces `DEBUG`
2. Environment variable `NORN_LOG_LEVEL`
3. YAML `logging.level`
4. Default: `INFO`

Implementation: `init_logging(config, cli_override=None)` receives the CLI override and merges.

---

## 5. Session correlation

### 5.1 contextvar-based design

```python
# src/norn/observability/logger.py
from contextvars import ContextVar

session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)

def new_session() -> str:
    """Generate a new session UUID and bind to contextvar."""
    sid = str(uuid.uuid4())
    session_id_var.set(sid)
    return sid
```

### 5.2 structlog processor injection

```python
# src/norn/observability/processors.py
def add_session_id(logger, method_name, event_dict):
    sid = session_id_var.get()
    if sid is not None:
        event_dict["session_id"] = sid
    return event_dict
```

### 5.3 Async compatibility

`contextvars` is the asyncio-safe successor to `threading.local`. Norn engines are async (`AgentLoop.run()`, `CoordinatorEngine.run()`, etc.), and contextvars propagate correctly across `await` boundaries and `asyncio.TaskGroup`. No explicit passing required.

### 5.4 CLI integration point

```python
# src/norn/cli/main.py (conceptual)
@app.command()
def run(..., verbose: bool = typer.Option(False, "--verbose", "-v")):
    config = NornConfig.load()
    init_logging(config.logging, cli_override="DEBUG" if verbose else None)
    new_session()  # bind session_id
    log = get_logger(__name__)
    log.info(EventName.AGENT_RUN, phase="start")
    # ... run engine
```

---

## 6. Instrumentation points

### 6.1 Core (Task 5)

| Location | Event | Notes |
|----------|-------|-------|
| `core/agent.py` → `AgentLoop.run()` | `agent.run` (start + end) | Wrap in try/finally for duration + success |
| `core/llm.py` → `LiteLLMProvider.complete()` | `llm.complete` | Capture tokens from response, call `litellm.completion_cost()` if enabled |
| `core/router.py` → `RouterProvider._select_tier()` | `routing.decision` | Log chosen tier + signals |
| `core/router.py` → `RouterProvider.complete()` except block | `fallback` | On tier failure before retry |

### 6.2 Periphery (Task 6)

| Location | Event | Notes |
|----------|-------|-------|
| `tools/registry.py` → `ToolRegistry.execute()` | `tool.call` | Wrap execution, capture error |
| `permissions/checker.py` → `PermissionChecker.check()` | `permission.decision` | Log every decision (grant/deny) |

### 6.3 Decoupling principle

**Engines do NOT import observability directly.** Instead:
- `get_logger(__name__)` is module-level
- If `LoggingConfig.enabled = False`, `init_logging` configures structlog with a no-op processor chain
- Tests can inject a `TestingLogger` capturing events in a list

No `LLMProvider` protocol changes. No signature changes to public methods.

---

## 7. Output sinks

### 7.1 Console sink (rich-backed)

Used when `output in {"console", "both"}`. Uses `structlog.dev.ConsoleRenderer` with rich colors:

```
2026-04-21T14:23:01Z [info] llm.complete        session_id=abc123 tier=standard model=stepfun/... total_tokens=1801 cost_usd=0.00123 latency_ms=2345
```

### 7.2 File sink (JSONL rotating)

Used when `output in {"file", "both"}`. Writes one JSON object per line to:
```
~/.norn/logs/YYYY-MM-DD.jsonl
```

Rotation: **daily by filename** (new file per UTC date). No size-based rotation for POC — `logrotate` or Phase 9 can handle retention.

Implementation: `logging.FileHandler` with a custom formatter that calls `json.dumps(event_dict)`. Structlog's `JSONRenderer` emits the string; we wrap it with a stdlib `FileHandler`.

### 7.3 Dual-sink composition

structlog supports multiple final renderers via the stdlib logging bridge:

```python
structlog.configure(
    processors=[
        add_timestamp_iso,
        add_session_id,
        add_cost_estimate,       # conditional
        redact_secrets,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)

# Then use stdlib logging handlers for fan-out:
root = logging.getLogger()
if output in {"console", "both"}:
    root.addHandler(_make_console_handler())
if output in {"file", "both"}:
    root.addHandler(_make_file_handler(file_dir))
```

---

## 8. Cost estimation

### 8.1 litellm integration

```python
# src/norn/observability/processors.py
def add_cost_estimate(logger, method_name, event_dict):
    if event_dict.get("event") != EventName.LLM_COMPLETE:
        return event_dict
    if not _cost_enabled:
        return event_dict
    try:
        from litellm import completion_cost
        cost = completion_cost(
            model=event_dict["model"],
            prompt_tokens=event_dict["prompt_tokens"],
            completion_tokens=event_dict["completion_tokens"],
        )
        event_dict["cost_usd"] = round(float(cost), 6)
    except Exception:
        event_dict["cost_usd"] = None
    return event_dict
```

Fail-open: any exception in cost computation → `cost_usd: null`, never break the log pipeline.

---

## 9. Secret redaction

### 9.1 Processor

```python
def redact_secrets(logger, method_name, event_dict):
    for key in list(event_dict.keys()):
        if any(pattern in key.lower() for pattern in _REDACT_KEYS):
            event_dict[key] = "[REDACTED]"
    return event_dict
```

Applied **after** all enrichment, **before** rendering. Protects against accidental logging of `api_key`, `authorization`, `token`, `password` fields (configurable via `LoggingConfig.redact_keys`).

---

## 10. Testing strategy

### 10.1 Unit tests (Tasks 1-3, 5-6)

- `test_events.py` → EventName enum values stable
- `test_logger.py` → init_logging idempotent, session_id propagation across async
- `test_processors.py` → each processor in isolation (session_id, timestamp, cost, redact)
- `test_sinks.py` → console output contains event name, file sink writes valid JSONL

### 10.2 Integration tests (Task 7)

- Full CLI command → parse resulting JSONL → assert 6 event types present with same `session_id`
- Cost present on `llm.complete` when enabled
- No secrets in output (inject fake api_key, verify redacted)
- `LoggingConfig.enabled=false` → zero writes to disk
- `--verbose` override produces DEBUG events

### 10.3 Test infrastructure

```python
# tests/test_observability/conftest.py
@pytest.fixture
def capture_events(tmp_path, monkeypatch):
    """Route logs to tmp_path, return JSONL parser."""
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    yield lambda: _parse_jsonl(tmp_path)
```

---

## 11. Task breakdown (7 tasks)

| # | Task | Files | LOC | Tests |
|---|------|-------|-----|-------|
| T1 | Foundation: `LoggingConfig` + `events.py` | `core/config.py`, `observability/events.py`, `configs/default.yaml` | ~70 | 6 |
| T2 | Logger setup: structlog config, session_id contextvar | `observability/logger.py`, `observability/processors.py` | ~180 | 10 |
| T3 | Sinks: console (rich) + file (JSONL rotating) | `observability/sinks.py` | ~120 | 6 |
| T4 | CLI integration: `init_logging` + `--verbose` flag + env var | `cli/main.py` (all commands) | ~80 | 5 |
| T5 | Instrumentation core: `agent.run`, `llm.complete` (+cost), `routing.decision`, `fallback` | `core/agent.py`, `core/llm.py`, `core/router.py` | ~120 | 10 |
| T6 | Instrumentation periphery: `tool.call`, `permission.decision` | `tools/registry.py`, `permissions/checker.py` | ~60 | 5 |
| T7 | E2E integration tests | `tests/test_observability/test_e2e.py` | ~150 | 5 |

**Total: ~780 LOC, ~47 tests.**

---

## 12. Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| structlog + stdlib logging bridge complexity | Medium | Document setup carefully, keep processor chain linear |
| Perf overhead per event (serialization) | Low | JSONL write is async-friendly via stdlib `FileHandler`; negligible vs LLM latency |
| Contextvar leak across tests | Medium | `conftest.py` fixture resets `session_id_var` per test |
| Cost computation fails for unknown model | Low | Fail-open: `cost_usd: null`, logged silently |
| Disk growth (JSONL unbounded) | Low (POC) | Daily rotation by filename; Phase 9 adds retention policy |
| Coupling engines to observability module | Medium | Module-level `get_logger(__name__)`; no protocol changes; `enabled=false` = no-op |

---

## 13. Follow-ups (Phase 9 backlog)

1. CLI reporting: `norn logs tail`, `norn logs stats`, `norn cost report`
2. Metrics aggregation (counters, histograms) → Prometheus exposition
3. OpenTelemetry trace spans (multi-step agent runs)
4. Retention policy + size-based rotation
5. Dashboard templates (Grafana JSON)
6. Integration with the 2 Phase 7 backlog items already aligned:
   - INFO log du tier choisi → **covered by `routing.decision`**
   - Aggregation des exceptions tier fallback → **covered by `fallback` event + error enrichment**

---

## 14. Open questions (none blocking)

All design questions resolved during brainstorming. Ready for plan writing.

---

## 15. Acceptance criteria

- [x] All 6 event types emitted from a single `norn run "..."` command
      (validated end-to-end by
      `tests/test_observability/test_e2e.py::test_full_run_emits_all_core_events_with_correlated_session_id`)
- [x] JSONL file at `~/.norn/logs/YYYY-MM-DD.jsonl` contains valid JSON per line
      (`DailyRotatingJsonlHandler` + sink tests; E2E tests parse lines as JSON)
- [x] All events in a run share the same `session_id`
      (asserted in E2E test #1; concurrent isolation covered by E2E test #7)
- [x] `llm.complete` events include `cost_usd` when model supported by litellm
      (`test_cost_included_when_enabled`; fail-open when computation fails)
- [x] `--verbose` flag emits DEBUG events
      (level resolution: CLI override > env > config, `init_logging`)
- [x] `LoggingConfig.enabled=false` results in zero log output
      (`test_disabled_logging_writes_nothing`)
- [x] Zero new ruff errors (baseline 37, unchanged)
- [x] All 420 existing tests still pass + ~45 new tests pass
      (final count: 485 tests passing)
- [x] No changes to `LLMProvider` protocol
- [x] No engine directly depends on `observability.*` (only `get_logger()`)

**Phase 8 complete** — all 6 events wired (`agent.run`, `llm.complete`,
`routing.decision`, `fallback`, `tool.call`, `permission.decision`), 485 tests
passing, ruff delta 0.
