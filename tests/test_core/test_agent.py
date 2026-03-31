"""Tests for the core agent loop."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from norn.core.agent import AgentLoop
from norn.core.config import PermissionMode
from norn.core.models import LLMResponse, TokenUsage, ToolCall
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
