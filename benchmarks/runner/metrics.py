"""Metrics extraction helpers for benchmark event traces."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable


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
