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
