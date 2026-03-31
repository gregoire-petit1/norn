"""Tests for coordinator workers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from norn.coordinator.models import CoordinatorPhase, WorkerAssignment, WorkerResult
from norn.coordinator.scratchpad import Scratchpad
from norn.coordinator.worker import Worker
from norn.core.models import LLMResponse
from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.registry import ToolRegistry


class MockReadInput(BaseModel):
    path: str = ""


class MockReadTool:
    name = "file_read"
    description = "Read a file"
    risk_level = RiskLevel.LOW
    input_model = MockReadInput

    async def execute(self, input: MockReadInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output="file content")


class MockGrepInput(BaseModel):
    pattern: str = ""


class MockGrepTool:
    name = "grep"
    description = "Search files"
    risk_level = RiskLevel.LOW
    input_model = MockGrepInput

    async def execute(self, input: MockGrepInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output="search results")


@pytest.fixture
def full_registry():
    """Registry with multiple tools."""
    reg = ToolRegistry()
    reg.register(MockReadTool())
    reg.register(MockGrepTool())
    return reg


@pytest.fixture
def scratchpad(tmp_path):
    sp = Scratchpad(base_dir=tmp_path / "scratchpad")
    sp.ensure_dirs()
    return sp


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content="I found 3 auth-related modules.",
            tool_calls=[],
        )
    )
    return llm


class TestWorkerCreation:
    def test_worker_scopes_tools(self, full_registry, scratchpad, mock_llm):
        """Worker gets only the tools specified in the assignment."""
        assignment = WorkerAssignment(
            worker_id="r1",
            phase=CoordinatorPhase.RESEARCH,
            task="Find auth modules",
            tools=["file_read"],
            scratchpad_section="r1-findings.md",
        )
        worker = Worker(
            assignment=assignment,
            llm=mock_llm,
            registry=full_registry,
            scratchpad=scratchpad,
        )
        scoped_tools = worker.agent.registry.list_tools()
        tool_names = {t.name for t in scoped_tools}
        assert tool_names == {"file_read"}

    def test_worker_empty_tools_gets_none(self, full_registry, scratchpad, mock_llm):
        """Worker with no tools specified gets an empty registry."""
        assignment = WorkerAssignment(
            worker_id="r1",
            phase=CoordinatorPhase.RESEARCH,
            task="Just think",
            tools=[],
            scratchpad_section="r1.md",
        )
        worker = Worker(
            assignment=assignment,
            llm=mock_llm,
            registry=full_registry,
            scratchpad=scratchpad,
        )
        assert worker.agent.registry.list_tools() == []


class TestWorkerExecution:
    @pytest.mark.asyncio
    async def test_worker_runs_and_returns_result(self, full_registry, scratchpad, mock_llm):
        """Worker executes assignment and returns WorkerResult."""
        assignment = WorkerAssignment(
            worker_id="r1",
            phase=CoordinatorPhase.RESEARCH,
            task="Find auth modules",
            tools=["file_read"],
            scratchpad_section="r1-findings.md",
        )
        worker = Worker(
            assignment=assignment,
            llm=mock_llm,
            registry=full_registry,
            scratchpad=scratchpad,
        )
        result = await worker.run()
        assert isinstance(result, WorkerResult)
        assert result.success is True
        assert result.worker_id == "r1"
        assert "auth" in result.output.lower()

    @pytest.mark.asyncio
    async def test_worker_writes_to_scratchpad(self, full_registry, scratchpad, mock_llm):
        """Worker writes its output to the scratchpad."""
        assignment = WorkerAssignment(
            worker_id="r1",
            phase=CoordinatorPhase.RESEARCH,
            task="Find auth modules",
            tools=["file_read"],
            scratchpad_section="r1-findings.md",
        )
        worker = Worker(
            assignment=assignment,
            llm=mock_llm,
            registry=full_registry,
            scratchpad=scratchpad,
        )
        await worker.run()
        content = scratchpad.read("research", "r1-findings.md")
        assert content is not None
        assert "auth" in content.lower()

    @pytest.mark.asyncio
    async def test_worker_handles_llm_error(self, full_registry, scratchpad):
        """Worker handles LLM errors gracefully."""
        bad_llm = AsyncMock()
        bad_llm.complete = AsyncMock(side_effect=Exception("API down"))
        assignment = WorkerAssignment(
            worker_id="w1",
            phase=CoordinatorPhase.RESEARCH,
            task="Do research",
            tools=[],
            scratchpad_section="w1.md",
        )
        worker = Worker(
            assignment=assignment,
            llm=bad_llm,
            registry=full_registry,
            scratchpad=scratchpad,
        )
        result = await worker.run()
        assert result.success is False
        assert "API down" in result.error

    @pytest.mark.asyncio
    async def test_worker_system_prompt_includes_task(self, full_registry, scratchpad, mock_llm):
        """Worker's system prompt includes the assigned task."""
        assignment = WorkerAssignment(
            worker_id="r1",
            phase=CoordinatorPhase.RESEARCH,
            task="Investigate the auth module",
            tools=["file_read"],
            scratchpad_section="r1.md",
        )
        worker = Worker(
            assignment=assignment,
            llm=mock_llm,
            registry=full_registry,
            scratchpad=scratchpad,
        )
        await worker.run()
        call_args = mock_llm.complete.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_content = messages[0].content
        assert "Investigate the auth module" in system_content
