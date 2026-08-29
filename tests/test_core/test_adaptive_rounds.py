"""Tests for progress-gated round-budget extension (wave 2 — A3)."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from norn.core.agent import AgentLoop
from norn.core.models import LLMResponse, Message, Role, ToolCall
from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.registry import ToolRegistry


class WriteInput(BaseModel):
    path: str
    content: str = ""


class WriteTool:
    name = "file_write"
    description = "write"
    risk_level = RiskLevel.MEDIUM
    input_model = WriteInput

    async def execute(self, input: WriteInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output=f"wrote {input.path}")


def _registry():
    reg = ToolRegistry()
    reg.register(WriteTool())
    return reg


# --------------------------------------------------------------------------- #
# _recent_progress / _should_extend_rounds units
# --------------------------------------------------------------------------- #


def test_recent_progress_detects_write():
    msgs = [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="1", name="file_write", arguments={})],
        ),
        Message(role=Role.TOOL, content="wrote x", tool_call_id="1"),
    ]
    assert AgentLoop._recent_progress(msgs) is True


def test_recent_progress_false_on_errors():
    msgs = [Message(role=Role.TOOL, content="Exit code 1\nboom", tool_call_id="1")]
    assert AgentLoop._recent_progress(msgs) is False


def test_should_extend_disabled_by_default():
    agent = AgentLoop(llm=AsyncMock(), registry=_registry(), env_bootstrap=False, repo_map=False)
    assert agent._should_extend_rounds([], 0) is False


def test_should_extend_respects_ceiling():
    agent = AgentLoop(
        llm=AsyncMock(), registry=_registry(), env_bootstrap=False, repo_map=False,
        adaptive_rounds=True, max_round_extensions=1,
    )
    good = [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="1", name="file_write", arguments={})],
        ),
        Message(role=Role.TOOL, content="wrote x", tool_call_id="1"),
    ]
    assert agent._should_extend_rounds(good, 0) is True
    assert agent._should_extend_rounds(good, 1) is False  # ceiling reached


def test_should_not_extend_when_stuck():
    agent = AgentLoop(
        llm=AsyncMock(), registry=_registry(), env_bootstrap=False, repo_map=False,
        adaptive_rounds=True,
    )
    # error_flood: 3+ consecutive tool errors
    stuck = [Message(role=Role.TOOL, content="Error: nope", tool_call_id=str(i)) for i in range(3)]
    assert agent._should_extend_rounds(stuck, 0) is False


# --------------------------------------------------------------------------- #
# Loop integration
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_extension_grants_extra_rounds_on_progress():
    """A progressing run past the cap gets one extension, then stops."""
    calls = {"n": 0}

    async def complete(messages, tools=None, **kw):
        calls["n"] += 1
        # Always call file_write (progress) and never finish → drive to the cap.
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id=str(calls["n"]), name="file_write", arguments={"path": "f"})],
        )

    llm = AsyncMock()
    llm.complete = complete
    agent = AgentLoop(
        llm=llm, registry=_registry(), env_bootstrap=False, repo_map=False,
        max_tool_rounds=2, adaptive_rounds=True, max_round_extensions=1,
        round_extension_factor=0.5,
    )
    await agent.run("go")
    # base budget 2 + one extension of max(1, int(2*0.5))=1 → 3 rounds
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_no_extension_when_disabled():
    calls = {"n": 0}

    async def complete(messages, tools=None, **kw):
        calls["n"] += 1
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id=str(calls["n"]), name="file_write", arguments={"path": "f"})],
        )

    llm = AsyncMock()
    llm.complete = complete
    agent = AgentLoop(
        llm=llm, registry=_registry(), env_bootstrap=False, repo_map=False,
        max_tool_rounds=2, adaptive_rounds=False,
    )
    await agent.run("go")
    assert calls["n"] == 2  # no extension


@pytest.mark.asyncio
async def test_max_rounds_message_still_emitted():
    async def complete(messages, tools=None, **kw):
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="file_write", arguments={"path": "f"})],
        )

    llm = AsyncMock()
    llm.complete = complete
    agent = AgentLoop(
        llm=llm, registry=_registry(), env_bootstrap=False, repo_map=False,
        max_tool_rounds=1, adaptive_rounds=True, max_round_extensions=1,
    )
    result = await agent.run("go")
    assert result.content == "[Max tool rounds reached]"
