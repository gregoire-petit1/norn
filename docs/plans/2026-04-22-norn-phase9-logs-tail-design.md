# Phase 9 — F1: `norn logs tail` Design

**Date:** 2026-04-22
**Status:** Approved (brainstorming complete)
**Scope:** Phase 9 v1, item F1
**Effort estimate:** ~2h (logs.py ~80 LOC + logger.py +20 LOC + 13 tests)

---

## 1. Scope & Behavior

### 1.1 Command

```
norn logs tail [OPTIONS]
```

### 1.2 Default behavior (no args)

- Reads the entire current-day JSONL file at `~/.norn/logs/YYYY-MM-DD.jsonl` (UTC date)
- Formats and prints each event to stdout
- Exits 0

### 1.3 Edge cases

| Condition | Behavior |
|---|---|
| Log file absent | Print info to stderr (`No logs for today (<path>).`), exit 0 |
| Malformed JSON line | Skip, count, emit aggregated warning to stderr at end (`Skipped N malformed line(s).`) |
| Empty file | No output to stdout, exit 0 |

### 1.4 Out of scope (deferred to Phase 9 v2)

- `--follow` (live tail like `tail -f`)
- `--since 5m` (relative duration filter)
- `--level error` (level filter)
- `--lines N` (last N lines)
- Multi-day reading (only today)
- Cross-day rotation handling during read
- Pager / pagination

---

## 2. CLI Surface (Typer)

```python
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
) -> None: ...
```

### 2.1 Filter semantics

- `--event` / `-e`: repeatable, **OR** logic (matches if `event` ∈ provided set)
- `--session` / `-s`: exact UUID match, OR literal `"last"` / `"current"` (alias) which resolves via `~/.norn/state/last_session`
- Combining `--event` and `--session`: **AND** between filters, OR within each
- `--json`: bypass rich formatting, emit raw JSONL line (post-filter), scriptable for `jq`/`grep`

---

## 3. Output Format (rich, default)

One line per event, compact:

```
HH:MM:SS [LEVEL] event.name key1=val1 key2=val2 ...
```

### 3.1 Styling

| Level | Rich style |
|---|---|
| `debug` | `dim` |
| `info` | `cyan` |
| `warning` | `yellow` |
| `error` | `red` |

### 3.2 Field rendering rules

- **Hidden from payload** (already in prefix or redundant): `event`, `level`, `timestamp`, `logger`
- **Displayed**: all other top-level keys, sorted alphabetically (reproducible test output)
- **Non-scalar values** (dict/list): inline JSON via `json.dumps(separators=(",", ":"))`, **no truncation**
- **`session_id`**: truncated to first 8 chars in rich mode (`sess=ab12cd34`); full UUID preserved in `--json` mode

### 3.3 `--json` mode

Emits the original JSONL line verbatim (after filtering). Skipped lines never appear. The malformed-lines warning still goes to stderr.

---

## 4. Plumbing `--session last` / `--session current`

### 4.1 New state file

```
~/.norn/state/last_session
```

Contents: a single line with the current session UUID. Atomic writes (tmp + rename).

### 4.2 Modification to `src/norn/observability/logger.py`

```python
_STATE_DIR = Path.home() / ".norn" / "state"
_LAST_SESSION_FILE = _STATE_DIR / "last_session"


def new_session() -> str:
    sid = str(uuid.uuid4())
    session_id_var.set(sid)
    _persist_last_session(sid)  # best-effort, fail-open
    return sid


def _persist_last_session(sid: str) -> None:
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _LAST_SESSION_FILE.with_suffix(".tmp")
        tmp.write_text(sid, encoding="utf-8")
        tmp.replace(_LAST_SESSION_FILE)
    except OSError as exc:
        logger.warning("failed to persist last_session", error=str(exc))
```

### 4.3 Resolution in CLI

```python
def _resolve_session(value: str) -> str:
    if value in {"last", "current"}:
        try:
            return _LAST_SESSION_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise typer.BadParameter("No previous session found.") from exc
    return value
```

`typer.BadParameter` triggers exit code 2 by default.

---

## 5. Module Structure

```
src/norn/cli/
├── main.py     # adds: from norn.cli.logs import logs_app
│               #       app.add_typer(logs_app, name="logs")
└── logs.py     # NEW
```

### 5.1 `src/norn/cli/logs.py` contents

| Symbol | Purpose |
|---|---|
| `logs_app = typer.Typer(name="logs", help="Inspect Norn structured logs.")` | Sub-app |
| `tail(...)` | Command per §2 |
| `_resolve_session(value: str) -> str` | Resolve `last`/`current` alias |
| `_today_log_path() -> Path` | `Path(load_config().logging.file_dir).expanduser() / f"{datetime.now(UTC).date().isoformat()}.jsonl"` |
| `_iter_filtered_events(path, events_filter, session_filter) -> Iterator[tuple[str, dict]]` | Yields `(raw_line, parsed_dict)`, applies filters, tracks malformed count |
| `_format_rich_line(event_dict: dict) -> Text` | Returns Rich `Text` per §3 |

### 5.2 `main.py` integration

After existing imports:
```python
from norn.cli.logs import logs_app
```
After `app = typer.Typer(...)`:
```python
app.add_typer(logs_app, name="logs")
```

No new helpers, no changes to existing commands.

---

## 6. Tests

### 6.1 New file: `tests/test_cli/test_logs.py`

Pattern: `CliRunner` + seeded JSONL files in `tmp_path`, `monkeypatch` config to point `file_dir` at `tmp_path`.

| # | Test | Asserts |
|---|---|---|
| 1 | `test_tail_no_logs_today_prints_info` | Missing file → exit 0 + stderr message |
| 2 | `test_tail_default_emits_all_events` | 3 events seeded → 3 formatted lines |
| 3 | `test_tail_filter_event_single` | `--event tool.call` → only matching events |
| 4 | `test_tail_filter_event_multiple_or` | `--event tool.call --event llm.complete` → OR semantics |
| 5 | `test_tail_filter_session_uuid_exact` | `--session <full-uuid>` → exact match |
| 6 | `test_tail_filter_session_last_resolves_state_file` | Seed state file → resolution OK |
| 7 | `test_tail_filter_session_last_missing_state_errors` | No state file → exit 2 + BadParameter |
| 8 | `test_tail_filter_session_current_alias` | `--session current` ≡ `--session last` |
| 9 | `test_tail_json_passthrough` | `--json` emits raw JSONL filtered |
| 10 | `test_tail_skips_malformed_lines` | Invalid JSON line skipped + aggregated stderr warning |
| 11 | `test_tail_combined_filters_and_logic` | `--event ... --session ...` → AND between filters |

### 6.2 Extension: `tests/test_observability/test_logger.py`

| # | Test | Asserts |
|---|---|---|
| 12 | `test_new_session_persists_last_session_file` | `new_session()` writes `~/.norn/state/last_session` |
| 13 | `test_new_session_fails_open_on_oserror` | Monkeypatch `mkdir` to raise → no crash, sid returned |

### 6.3 Conftest impact

The autouse fixture in `conftest.py` already redirects `LoggingConfig.file_dir` → `tmp_path`. For tests touching `_STATE_DIR`, add a **local** `monkeypatch.setattr("norn.observability.logger._STATE_DIR", tmp_path / "state")`. No changes needed to the global autouse fixture.

---

## 7. Conventions Compliance

- `from __future__ import annotations` at top of `logs.py`
- `TYPE_CHECKING` for `pathlib.Path` if only used in annotations (likely needed for runtime → import normally)
- `Annotated[...]` for all Typer options (per existing pattern in `main.py`)
- No magic numbers; session truncation length `8` extracted as `_SESSION_DISPLAY_LEN = 8`
- Type hints throughout
- Ruff-clean (no new errors vs baseline 37)

---

## 8. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| `new_session()` write side-effect violates logger purity | Fail-open with warning; state dir creation lazy |
| State file race (two `norn` processes concurrent) | Last writer wins, atomic rename guarantees file integrity |
| Large JSONL file (>100MB) blocks rich render | Generator-based, line-by-line; deferred pagination noted |
| Conftest autouse leaks state file across tests | Local monkeypatch of `_STATE_DIR` per test |

---

## 9. Acceptance Criteria

- [ ] `norn logs tail` command registered under `norn logs` Typer sub-app
- [ ] Default invocation reads today's JSONL, formats per §3
- [ ] All filter combinations (§2.1) work with documented semantics
- [ ] `--session last` / `--session current` resolve via state file
- [ ] `new_session()` persists to `~/.norn/state/last_session` (fail-open)
- [ ] All 13 tests pass
- [ ] No ruff regression vs baseline 37
- [ ] Total test count: 485 + 13 = **498 passed**
