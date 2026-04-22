"""Tests for benchmark metrics extraction helpers."""

from __future__ import annotations

from benchmarks.runner.metrics import (
    count_events,
    sum_field,
    count_by_tool,
    count_events_by,
)

EVENTS = [
    {"event": "llm.complete", "total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20},
    {"event": "llm.complete", "total_tokens": 200, "prompt_tokens": 150, "completion_tokens": 50},
    {"event": "tool.call", "tool_name": "bash", "success": True},
    {"event": "tool.call", "tool_name": "file_read", "success": True},
    {"event": "tool.call", "tool_name": "bash", "success": True},
    {"event": "tool.call", "tool_name": "bash", "error_type": "PermissionDenied", "success": False},
]


def test_count_events():
    assert count_events(EVENTS, "llm.complete") == 2
    assert count_events(EVENTS, "tool.call") == 4


def test_count_events_zero_for_missing():
    assert count_events(EVENTS, "nonexistent") == 0


def test_sum_field():
    assert sum_field(EVENTS, "total_tokens") == 300
    assert sum_field(EVENTS, "prompt_tokens") == 230


def test_sum_field_missing_key():
    assert sum_field(EVENTS, "nonexistent_field") == 0


def test_count_by_tool():
    dist = count_by_tool(EVENTS)
    assert dist == {"bash": 3, "file_read": 1}


def test_count_by_tool_empty():
    assert count_by_tool([]) == {}


def test_count_events_by():
    n = count_events_by(
        EVENTS,
        "tool.call",
        lambda e: e.get("error_type") == "PermissionDenied",
    )
    assert n == 1


def test_count_events_by_no_match():
    n = count_events_by(
        EVENTS,
        "tool.call",
        lambda e: e.get("error_type") == "NonExistent",
    )
    assert n == 0


def test_count_events_empty_list():
    assert count_events([], "llm.complete") == 0
    assert sum_field([], "total_tokens") == 0
