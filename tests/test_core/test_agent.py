"""Tests for the core agent loop."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from norn.core.agent import AgentLoop
from norn.core.models import LLMResponse, TokenUsage, ToolCall
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


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
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
