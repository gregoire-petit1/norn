"""Tests for core message and response models."""

from norn.core.models import (
    LLMResponse,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    ToolResult,
)


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
