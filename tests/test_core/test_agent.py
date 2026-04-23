"""Tests for the core agent loop."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from norn.core.agent import AgentLoop
from norn.core.config import PermissionMode
from norn.core.models import LLMResponse, TokenUsage, ToolCall
from norn.memory.models import MemoryConfig
from norn.memory.session_logger import SessionLogger
from norn.memory.store import MemoryStore
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier
from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.registry import ToolRegistry


class EchoInput(BaseModel):
    text: str


class EchoTool:
    name = "echo"
    description = "Echo text back"
    risk_level = RiskLevel.LOW
    input_model = EchoInput

    async def execute(self, input: EchoInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output=f"echo: {input.text}")


class WriteInput(BaseModel):
    path: str
    content: str


class WriteTool:
    name = "file_write"
    description = "Write a file"
    risk_level = RiskLevel.MEDIUM
    input_model = WriteInput

    async def execute(self, input: WriteInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output=f"wrote to {input.path}")


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


@pytest.fixture
def registry_with_write():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(WriteTool())
    return reg


@pytest.fixture
def mock_llm_text_only():
    """LLM that returns text without tool calls."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content="Hello, I'm Norn!",
            tool_calls=[],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    return llm


@pytest.fixture
def mock_llm_with_tool():
    """LLM that first calls a tool, then responds with text."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            # First call: tool invocation
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "test"})],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            # Second call: final text response
            LLMResponse(
                content="The echo said: test",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            ),
        ]
    )
    return llm


@pytest.mark.asyncio
async def test_agent_text_response(mock_llm_text_only, registry):
    agent = AgentLoop(llm=mock_llm_text_only, registry=registry)
    result = await agent.run("hello")
    assert result.content == "Hello, I'm Norn!"
    assert mock_llm_text_only.complete.call_count == 1


@pytest.mark.asyncio
async def test_agent_tool_call_then_response(mock_llm_with_tool, registry):
    agent = AgentLoop(llm=mock_llm_with_tool, registry=registry)
    result = await agent.run("echo something")
    assert result.content == "The echo said: test"
    assert mock_llm_with_tool.complete.call_count == 2


@pytest.mark.asyncio
async def test_agent_unknown_tool(mock_llm_text_only, registry):
    """LLM calls a tool that doesn't exist."""
    mock_llm_text_only.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="nonexistent", arguments={})],
            ),
            LLMResponse(content="Sorry, tool not found.", tool_calls=[]),
        ]
    )
    agent = AgentLoop(llm=mock_llm_text_only, registry=registry)
    result = await agent.run("do something")
    assert result.content == "Sorry, tool not found."


@pytest.mark.asyncio
async def test_agent_history_grows(mock_llm_text_only, registry):
    agent = AgentLoop(llm=mock_llm_text_only, registry=registry)
    await agent.run("first")
    await agent.run("second")
    # History should contain: user1, assistant1, user2, assistant2
    assert len(agent.history) == 4


@pytest.mark.asyncio
async def test_agent_permission_denied(registry_with_write):
    """When permission is denied, tool should not execute."""
    checker = PermissionChecker(
        mode=PermissionMode.INTERACTIVE,
        classifier=RiskClassifier(),
        prompt_fn=None,  # No prompt = auto-deny for MEDIUM+
    )

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="file_write",
                        arguments={"path": "test.py", "content": "hello"},
                    )
                ],
            ),
            LLMResponse(content="Permission was denied.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(llm=llm, registry=registry_with_write, permission_checker=checker)
    result = await agent.run("write a file")
    assert result.content == "Permission was denied."
    assert llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_agent_permission_approved_yolo(registry_with_write):
    """In yolo mode, everything is approved."""
    checker = PermissionChecker(
        mode=PermissionMode.YOLO,
        classifier=RiskClassifier(),
    )

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="file_write",
                        arguments={"path": "test.py", "content": "hello"},
                    )
                ],
            ),
            LLMResponse(content="File written.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(llm=llm, registry=registry_with_write, permission_checker=checker)
    result = await agent.run("write a file")
    assert result.content == "File written."


@pytest.mark.asyncio
async def test_agent_no_permission_checker_allows_all(registry):
    """Without a permission checker, all tools execute (backward-compatible)."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    agent = AgentLoop(llm=llm, registry=registry)
    result = await agent.run("echo")
    assert result.content == "done"


# --- Memory integration fixtures ---


@pytest.fixture
def memory_store(tmp_path):
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = MemoryStore(config)
    store.ensure_dirs()
    store.write_memory("# Norn Memory\n\n- User likes Python\n- Project: Norn agent\n")
    return store


@pytest.fixture
def session_logger(memory_store):
    return SessionLogger(memory_store)


# --- Memory integration tests ---


@pytest.mark.asyncio
async def test_agent_injects_memory_into_system_prompt(registry, memory_store):
    """When memory_store is provided, MEMORY.md is added to system prompt."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(content="I know you like Python!", tool_calls=[])
    )

    agent = AgentLoop(llm=llm, registry=registry, memory_store=memory_store)
    await agent.run("What do you know about me?")

    # Check that the system prompt sent to LLM includes memory
    call_args = llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    system_msg = messages[0]
    assert "User likes Python" in system_msg.content


@pytest.mark.asyncio
async def test_agent_tracks_session_stats(registry, memory_store, session_logger):
    """Agent tracks message and tool call counts for session logging."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            ),
            LLMResponse(content="Done.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(
        llm=llm,
        registry=registry,
        memory_store=memory_store,
        session_logger=session_logger,
    )
    await agent.run("echo hi")
    assert agent.tool_call_count == 1
    assert agent.user_message_count == 1


@pytest.mark.asyncio
async def test_agent_without_memory_still_works(registry):
    """Backward compatibility: agent works without memory."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(content="hello", tool_calls=[]))
    agent = AgentLoop(llm=llm, registry=registry)
    result = await agent.run("hi")
    assert result.content == "hello"


# --------------------------------------------------------------------------- #
# Streaming agent loop tests
# --------------------------------------------------------------------------- #


from norn.core.models import AgentEvent, EventType, StreamChunk


async def _mock_stream_text(text_chunks: list[str]):
    """Helper: mock LLM.stream() yielding text chunks then done."""
    for text in text_chunks:
        yield StreamChunk(content=text, done=False)
    yield StreamChunk(content=None, done=True)


async def _mock_stream_tool_call(tool_calls: list[ToolCall]):
    """Helper: mock LLM.stream() yielding a done chunk with tool calls."""
    yield StreamChunk(content=None, tool_calls=tool_calls, done=True)


@pytest.mark.asyncio
async def test_run_stream_text_only(registry):
    """run_stream should yield TEXT_DELTA events then DONE."""
    llm = AsyncMock()
    llm.stream = lambda **kwargs: _mock_stream_text(["Hello", " world"])

    agent = AgentLoop(llm=llm, registry=registry)
    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    text_events = [e for e in events if e.type == EventType.TEXT_DELTA]
    assert len(text_events) >= 1
    assert any(e.content == "Hello" for e in text_events)

    done_events = [e for e in events if e.type == EventType.DONE]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_run_stream_tool_call_then_text(registry):
    """run_stream should yield TOOL_START, TOOL_END, then TEXT_DELTA, DONE."""
    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First round: tool call
            async for chunk in _mock_stream_tool_call(
                [ToolCall(id="c1", name="echo", arguments={"text": "hi"})]
            ):
                yield chunk
        else:
            # Second round: text response
            async for chunk in _mock_stream_text(["Done"]):
                yield chunk

    llm = AsyncMock()
    llm.stream = mock_stream

    agent = AgentLoop(llm=llm, registry=registry)
    events = []
    async for event in agent.run_stream("echo hi"):
        events.append(event)

    types = [e.type for e in events]
    assert EventType.TOOL_START in types
    assert EventType.TOOL_END in types
    assert EventType.TEXT_DELTA in types
    assert EventType.DONE in types

    # TOOL_START should come before TOOL_END
    start_idx = types.index(EventType.TOOL_START)
    end_idx = types.index(EventType.TOOL_END)
    assert start_idx < end_idx

    # Tool end should have success=True
    tool_end = [e for e in events if e.type == EventType.TOOL_END][0]
    assert tool_end.success is True
    assert tool_end.tool_name == "echo"


@pytest.mark.asyncio
async def test_run_stream_updates_history(registry):
    """run_stream should update history like run()."""
    llm = AsyncMock()
    llm.stream = lambda **kwargs: _mock_stream_text(["Hello"])

    agent = AgentLoop(llm=llm, registry=registry)
    async for _ in agent.run_stream("hi"):
        pass

    assert len(agent.history) == 2  # user + assistant
    assert agent.user_message_count == 1
