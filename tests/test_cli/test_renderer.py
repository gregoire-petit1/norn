"""Tests for the streaming CLI renderer."""

from __future__ import annotations

import pytest
from rich.console import Console

from norn.cli.renderer import StreamRenderer
from norn.core.models import AgentEvent, EventType, TokenUsage


async def _events(*items: AgentEvent):
    """Helper: async generator yielding AgentEvent items."""
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_renderer_text_only():
    """Text deltas should be printed and DONE should flush markdown."""
    console = Console(file=None, force_terminal=False, no_color=True, width=120)
    renderer = StreamRenderer(console)

    events = _events(
        AgentEvent(type=EventType.TEXT_DELTA, content="Hello"),
        AgentEvent(type=EventType.TEXT_DELTA, content=" world"),
        AgentEvent(type=EventType.DONE),
    )

    await renderer.render(events)
    assert renderer.full_text == "Hello world"


@pytest.mark.asyncio
async def test_renderer_tool_events():
    """TOOL_START and TOOL_END should be rendered."""
    buf = []
    console = Console(file=None, force_terminal=False, no_color=True, width=120)
    renderer = StreamRenderer(console)

    events = _events(
        AgentEvent(
            type=EventType.TOOL_START,
            tool_name="bash",
            tool_args="ls -la",
        ),
        AgentEvent(
            type=EventType.TOOL_END,
            tool_name="bash",
            tool_args="ls -la",
            duration_ms=250,
            success=True,
        ),
        AgentEvent(type=EventType.DONE),
    )

    await renderer.render(events)
    # Renderer should track tool events
    assert renderer.tool_count == 1


@pytest.mark.asyncio
async def test_renderer_tool_failure():
    """Failed tools should be tracked."""
    console = Console(file=None, force_terminal=False, no_color=True, width=120)
    renderer = StreamRenderer(console)

    events = _events(
        AgentEvent(
            type=EventType.TOOL_START,
            tool_name="bash",
            tool_args="rm -rf /",
        ),
        AgentEvent(
            type=EventType.TOOL_END,
            tool_name="bash",
            tool_args="rm -rf /",
            duration_ms=100,
            success=False,
        ),
        AgentEvent(type=EventType.DONE),
    )

    await renderer.render(events)
    assert renderer.tool_count == 1
    assert renderer.tool_failures == 1


@pytest.mark.asyncio
async def test_renderer_done_with_usage():
    """DONE event with usage should be recorded."""
    console = Console(file=None, force_terminal=False, no_color=True, width=120)
    renderer = StreamRenderer(console)

    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    events = _events(
        AgentEvent(type=EventType.TEXT_DELTA, content="Done."),
        AgentEvent(type=EventType.DONE, usage=usage, latency_ms=2000, model="test/model"),
    )

    await renderer.render(events)
    assert renderer.final_usage is not None
    assert renderer.final_usage.total_tokens == 150


@pytest.mark.asyncio
async def test_renderer_mixed_text_tool_text():
    """Text -> tool -> text flow should accumulate all text."""
    console = Console(file=None, force_terminal=False, no_color=True, width=120)
    renderer = StreamRenderer(console)

    events = _events(
        AgentEvent(type=EventType.TEXT_DELTA, content="Let me check"),
        AgentEvent(type=EventType.TOOL_START, tool_name="file_read", tool_args="main.py"),
        AgentEvent(
            type=EventType.TOOL_END,
            tool_name="file_read",
            tool_args="main.py",
            duration_ms=50,
            success=True,
        ),
        AgentEvent(type=EventType.TEXT_DELTA, content=". Found it!"),
        AgentEvent(type=EventType.DONE),
    )

    await renderer.render(events)
    assert renderer.full_text == "Let me check. Found it!"
    assert renderer.tool_count == 1
