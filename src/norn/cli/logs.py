"""``norn logs tail`` and future log-inspection commands (Phase 9 F1).

This sub-Typer is mounted under ``norn logs`` from ``norn.cli.main``.

Design reference: ``docs/plans/2026-04-22-norn-phase9-logs-tail-design.md``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.text import Text

from norn.core.config import NornConfig
from norn.observability.logger import _LAST_SESSION_FILE

if TYPE_CHECKING:
    from collections.abc import Iterator

logs_app = typer.Typer(name="logs", help="Inspect Norn structured logs.")

# ---------------------------------------------------------------------------
# Module-level constants (no magic numbers in helpers)
# ---------------------------------------------------------------------------

_SESSION_DISPLAY_LEN = 8
"""Number of leading chars of a session UUID shown in rich mode."""

_HIDDEN_KEYS = frozenset({"event", "level", "timestamp", "logger"})
"""Keys already represented in the line prefix; suppressed from key=val tail."""

_LEVEL_STYLE = {
    "debug": "dim",
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
}
"""Rich style per level (used for the bracketed [LEVEL] token)."""

_TIMESTAMP_HHMMSS_END = 19
"""ISO-8601 ``HH:MM:SS`` ends at index 19 of a ``YYYY-MM-DDTHH:MM:SS...`` string."""

_console = Console()
_err_console = Console(stderr=True, soft_wrap=True)


# ---------------------------------------------------------------------------
# Internal helpers (named with leading underscore but referenced by tests via
# ``monkeypatch.setattr`` on the module path; keeping them module-private is
# fine because tests live in the same project).
# ---------------------------------------------------------------------------


def _today_log_path() -> Path:
    """Resolve today's JSONL log file path from the active config."""
    cfg = NornConfig.load()
    today = datetime.now(UTC).date().isoformat()
    return Path(cfg.logging.file_dir).expanduser() / f"{today}.jsonl"


def _resolve_session(value: str) -> str:
    """Resolve ``--session`` value, accepting ``last`` / ``current`` aliases.

    Raises ``typer.BadParameter`` (exit code 2) when the alias is requested
    but no state file exists yet — this happens when no ``norn`` command has
    been run yet, which is the only honest answer we can give.
    """
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
    """Yield ``(raw_line, parsed_dict)`` pairs that pass the active filters.

    Malformed JSON lines are silently skipped; an aggregated count is emitted
    to stderr at the end of iteration so users notice corruption without one
    warning per line spamming the output.
    """
    malformed = 0
    with path.open(encoding="utf-8") as fh:
        for raw_with_nl in fh:
            raw = raw_with_nl.rstrip("\n")
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if events_filter is not None and parsed.get("event") not in events_filter:
                continue
            if session_filter is not None and parsed.get("session_id") != session_filter:
                continue
            yield raw, parsed
    if malformed:
        _err_console.print(f"Skipped {malformed} malformed line(s).", style="yellow")


def _format_rich_line(ev: dict) -> Text:
    """Compact one-line rich rendering of an event.

    Format: ``HH:MM:SS [LEVEL] event.name k1=v1 k2=v2 ...`` with keys sorted
    for reproducible test output. Non-scalar values are inlined as compact
    JSON. ``session_id`` is truncated to ``_SESSION_DISPLAY_LEN`` chars.
    """
    ts = ev.get("timestamp", "")
    hhmmss = ts[11:_TIMESTAMP_HHMMSS_END] if len(ts) >= _TIMESTAMP_HHMMSS_END else ts
    level = (ev.get("level") or "info").lower()
    name = ev.get("event", "<no-event>")
    style = _LEVEL_STYLE.get(level, "white")

    text = Text()
    text.append(f"{hhmmss} ", style="dim")
    text.append(f"[{level.upper()}] ", style=style)
    text.append(name, style="bold")

    for key, value in sorted((k, v) for k, v in ev.items() if k not in _HIDDEN_KEYS):
        rendered: str
        if key == "session_id" and isinstance(value, str):
            rendered = value[:_SESSION_DISPLAY_LEN]
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(value, separators=(",", ":"))
        else:
            rendered = str(value)
        text.append(f" {key}={rendered}", style="dim")

    return text


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


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
    """Tail today's Norn log file with optional filters.

    Reads ``~/.norn/logs/YYYY-MM-DD.jsonl`` (per ``logging.file_dir`` config).
    Missing file → exit 0 with an info message on stderr (not an error: many
    users will run ``norn logs tail`` before any agent has emitted events).
    """
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
