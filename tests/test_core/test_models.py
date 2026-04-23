"""Tests for core message and response models."""

from norn.core.models import (
    AgentEvent,
    EventType,
    LLMResponse,
    Message,
    Role,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from norn.tools.base import ToolResult


def test_message_creation():
    msg = Message(role=Role.USER, content="hello")
    assert msg.role == Role.USER
    assert msg.content == "hello"
    assert msg.tool_call_id is None


def test_tool_call_creation():
    call = ToolCall(id="call_123", name="bash", arguments={"command": "ls"})
    assert call.id == "call_123"
    assert call.name == "bash"
    assert call.arguments == {"command": "ls"}


def test_tool_result_success():
    result = ToolResult(output="file1.py\nfile2.py")
    assert result.output == "file1.py\nfile2.py"
    assert result.error is None
    assert result.is_error is False


def test_tool_result_error():
    result = ToolResult(error="Permission denied")
    assert result.output is None
    assert result.error == "Permission denied"
    assert result.is_error is True


def test_llm_response_text_only():
    resp = LLMResponse(content="Hello!", tool_calls=[])
    assert resp.content == "Hello!"
    assert resp.tool_calls == []
    assert resp.has_tool_calls is False


def test_llm_response_with_tool_calls():
    calls = [ToolCall(id="c1", name="bash", arguments={"command": "ls"})]
    resp = LLMResponse(content=None, tool_calls=calls)
    assert resp.has_tool_calls is True


def test_stream_chunk():
    chunk = StreamChunk(content="partial", done=False)
    assert chunk.content == "partial"
    assert chunk.done is False


# ── AgentEvent / EventType / StreamChunk enhancements ─────────────────


def test_stream_chunk_has_usage_field():
    chunk = StreamChunk(content="hi", done=True, usage=TokenUsage(prompt_tokens=10))
    assert chunk.usage.prompt_tokens == 10


def test_agent_event_text_delta():
    event = AgentEvent(type=EventType.TEXT_DELTA, content="hello")
    assert event.type == EventType.TEXT_DELTA
    assert event.content == "hello"


def test_agent_event_tool_start():
    event = AgentEvent(type=EventType.TOOL_START, tool_name="bash", tool_args="ls")
    assert event.tool_name == "bash"


def test_agent_event_tool_end():
    event = AgentEvent(
        type=EventType.TOOL_END,
        tool_name="bash",
        tool_args="ls",
        duration_ms=150,
        success=True,
    )
    assert event.duration_ms == 150
    assert event.success is True


def test_agent_event_done():
    event = AgentEvent(
        type=EventType.DONE,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        latency_ms=2000,
    )
    assert event.usage.total_tokens == 150
    assert event.latency_ms == 2000


def test_event_type_values():
    assert EventType.TEXT_DELTA == "text_delta"
    assert EventType.TOOL_START == "tool_start"
    assert EventType.TOOL_END == "tool_end"
    assert EventType.DONE == "done"
