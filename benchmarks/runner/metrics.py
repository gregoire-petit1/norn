"""Metrics extraction helpers for benchmark event traces."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pathlib import Path


def load_session_events(log_path: Path, session_id: str) -> list[dict[str, Any]]:
    """Parse a JSONL log file, returning events whose session_id matches.

    Malformed lines are skipped; a missing file returns []. Best-effort by
    design — bench metrics must never fail a run.
    """
    if not log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(event, dict) and event.get("session_id") == session_id:
            events.append(event)
    return events


def count_events(events: list[dict[str, Any]], event_name: str) -> int:
    """Count events matching a given event name."""
    return sum(1 for e in events if e.get("event") == event_name)


def sum_field(events: list[dict[str, Any]], field: str) -> int | float:
    """Sum a numeric field across all events (skipping events where field is absent)."""
    return sum(e.get(field, 0) for e in events)


def count_by_tool(events: list[dict[str, Any]]) -> dict[str, int]:
    """Count tool.call events grouped by tool_name."""
    counter: Counter[str] = Counter()
    for e in events:
        if e.get("event") == "tool.call" and "tool_name" in e:
            counter[e["tool_name"]] += 1
    return dict(counter)


def count_events_by(
    events: list[dict[str, Any]],
    event_name: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> int:
    """Count events matching event_name AND a custom predicate."""
    return sum(1 for e in events if e.get("event") == event_name and predicate(e))
