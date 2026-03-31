"""Tests for the coordinator engine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from norn.coordinator.engine import CoordinatorEngine
from norn.coordinator.models import CoordinatorPhase, CoordinatorResult
from norn.coordinator.scratchpad import Scratchpad
from norn.core.models import LLMResponse
from norn.tools.registry import ToolRegistry


@pytest.fixture
def scratchpad(tmp_path):
    sp = Scratchpad(base_dir=tmp_path / "scratchpad")
    sp.ensure_dirs()
    return sp


@pytest.fixture
def registry():
    return ToolRegistry()


def _make_assignment_response(assignments: list[dict]) -> LLMResponse:
    """Create an LLM response with worker assignments JSON."""
    return LLMResponse(content=json.dumps(assignments))


def _make_worker_llm(content: str = "Task completed.") -> AsyncMock:
    """Create a mock LLM that returns simple text."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(content=content, tool_calls=[]))
    return llm


class TestPhaseExecution:
    @pytest.mark.asyncio
    async def test_research_phase_dispatches_workers(self, scratchpad, registry):
        """Research phase creates and runs workers."""
        coordinator_llm = AsyncMock()
        coordinator_llm.complete = AsyncMock(
            return_value=_make_assignment_response(
                [
                    {
                        "worker_id": "r1",
                        "task": "Find auth modules",
                        "tools": [],
                        "scratchpad_section": "r1-findings.md",
                    },
                ]
            )
        )
        worker_llm = _make_worker_llm("Found auth in src/auth/.")

        engine = CoordinatorEngine(
            coordinator_llm=coordinator_llm,
            worker_llm=worker_llm,
            registry=registry,
            scratchpad=scratchpad,
        )
        results = await engine._run_phase(
            phase=CoordinatorPhase.RESEARCH,
            task="Refactor auth",
            context="",
        )
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].worker_id == "r1"

    @pytest.mark.asyncio
    async def test_phase_writes_to_scratchpad(self, scratchpad, registry):
        """Workers write their output to the scratchpad."""
        coordinator_llm = AsyncMock()
        coordinator_llm.complete = AsyncMock(
            return_value=_make_assignment_response(
                [
                    {
                        "worker_id": "r1",
                        "task": "Find modules",
                        "tools": [],
                        "scratchpad_section": "r1.md",
                    },
                ]
            )
        )
        worker_llm = _make_worker_llm("Found 3 modules.")

        engine = CoordinatorEngine(
            coordinator_llm=coordinator_llm,
            worker_llm=worker_llm,
            registry=registry,
            scratchpad=scratchpad,
        )
        await engine._run_phase(
            phase=CoordinatorPhase.RESEARCH,
            task="Find modules",
            context="",
        )
        content = scratchpad.read("research", "r1.md")
        assert content is not None
        assert "3 modules" in content

    @pytest.mark.asyncio
    async def test_phase_handles_empty_assignments(self, scratchpad, registry):
        """Phase with no assignments returns empty results."""
        coordinator_llm = AsyncMock()
        coordinator_llm.complete = AsyncMock(return_value=_make_assignment_response([]))
        worker_llm = _make_worker_llm()

        engine = CoordinatorEngine(
            coordinator_llm=coordinator_llm,
            worker_llm=worker_llm,
            registry=registry,
            scratchpad=scratchpad,
        )
        results = await engine._run_phase(
            phase=CoordinatorPhase.SYNTHESIS,
            task="Synthesize",
            context="",
        )
        assert results == []


class TestContextBuilding:
    @pytest.mark.asyncio
    async def test_context_includes_previous_phase_output(self, scratchpad, registry):
        """Context for next phase includes scratchpad content from previous phase."""
        # Pre-populate research findings
        scratchpad.write("research", "r1.md", "Found module A.")

        engine = CoordinatorEngine(
            coordinator_llm=AsyncMock(),
            worker_llm=AsyncMock(),
            registry=registry,
            scratchpad=scratchpad,
        )
        context = engine._build_phase_context()
        assert "Found module A" in context


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, scratchpad, registry):
        """Full R->S->I->V pipeline completes successfully."""
        # Coordinator LLM returns assignments for each phase
        coordinator_llm = AsyncMock()
        coordinator_llm.complete = AsyncMock(
            side_effect=[
                # Research phase assignments
                _make_assignment_response(
                    [
                        {
                            "worker_id": "r1",
                            "task": "Research",
                            "tools": [],
                            "scratchpad_section": "r1.md",
                        },
                    ]
                ),
                # Synthesis phase assignments
                _make_assignment_response(
                    [
                        {
                            "worker_id": "s1",
                            "task": "Write spec",
                            "tools": [],
                            "scratchpad_section": "s1.md",
                        },
                    ]
                ),
                # Implementation phase assignments
                _make_assignment_response(
                    [
                        {
                            "worker_id": "i1",
                            "task": "Implement",
                            "tools": [],
                            "scratchpad_section": "i1.md",
                        },
                    ]
                ),
                # Verification phase assignments
                _make_assignment_response(
                    [
                        {
                            "worker_id": "v1",
                            "task": "Verify",
                            "tools": [],
                            "scratchpad_section": "v1.md",
                        },
                    ]
                ),
            ]
        )
        worker_llm = _make_worker_llm("Done.")

        engine = CoordinatorEngine(
            coordinator_llm=coordinator_llm,
            worker_llm=worker_llm,
            registry=registry,
            scratchpad=scratchpad,
        )
        result = await engine.coordinate("Refactor auth module")
        assert isinstance(result, CoordinatorResult)
        assert result.success is True
        assert result.total_workers == 4
        assert len(result.phase_results) == 4

    @pytest.mark.asyncio
    async def test_pipeline_handles_coordinator_error(self, scratchpad, registry):
        """Pipeline handles coordinator LLM errors."""
        coordinator_llm = AsyncMock()
        coordinator_llm.complete = AsyncMock(side_effect=Exception("LLM down"))
        worker_llm = _make_worker_llm()

        engine = CoordinatorEngine(
            coordinator_llm=coordinator_llm,
            worker_llm=worker_llm,
            registry=registry,
            scratchpad=scratchpad,
        )
        result = await engine.coordinate("Do something")
        assert result.success is False
        assert "LLM down" in result.error

    @pytest.mark.asyncio
    async def test_pipeline_handles_invalid_json(self, scratchpad, registry):
        """Pipeline handles invalid JSON from coordinator."""
        coordinator_llm = AsyncMock()
        coordinator_llm.complete = AsyncMock(return_value=LLMResponse(content="not valid json"))
        worker_llm = _make_worker_llm()

        engine = CoordinatorEngine(
            coordinator_llm=coordinator_llm,
            worker_llm=worker_llm,
            registry=registry,
            scratchpad=scratchpad,
        )
        result = await engine.coordinate("Do something")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_pipeline_worker_failure_does_not_crash(self, scratchpad, registry):
        """A failing worker doesn't crash the pipeline."""
        coordinator_llm = AsyncMock()
        coordinator_llm.complete = AsyncMock(
            side_effect=[
                _make_assignment_response(
                    [
                        {
                            "worker_id": "r1",
                            "task": "Research",
                            "tools": [],
                            "scratchpad_section": "r1.md",
                        },
                    ]
                ),
                _make_assignment_response([]),
                _make_assignment_response([]),
                _make_assignment_response([]),
            ]
        )
        bad_worker_llm = AsyncMock()
        bad_worker_llm.complete = AsyncMock(side_effect=Exception("Worker crashed"))

        engine = CoordinatorEngine(
            coordinator_llm=coordinator_llm,
            worker_llm=bad_worker_llm,
            registry=registry,
            scratchpad=scratchpad,
        )
        result = await engine.coordinate("Do something")
        # Pipeline should still complete (worker failure is recorded, not fatal)
        assert isinstance(result, CoordinatorResult)
        research_results = result.phase_results.get(CoordinatorPhase.RESEARCH, [])
        assert len(research_results) == 1
        assert research_results[0].success is False
