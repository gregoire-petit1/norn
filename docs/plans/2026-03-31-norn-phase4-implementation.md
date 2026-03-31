# Norn Phase 4: Coordinator Multi-Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multi-agent coordinator that decomposes complex tasks into a 4-phase pipeline (Research → Synthesis → Implementation → Verification) with parallel workers and a shared filesystem scratchpad. Inspired by Claude Code's Coordinator mode.

**Architecture:** A `Coordinator` orchestrates 4 sequential phases. Each phase can dispatch parallel `Worker` instances — isolated `AgentLoop`s with scoped tool subsets and write access limited to their scratchpad section. Workers communicate via a `Scratchpad` (temp directory with structured sections). An `ActivationHeuristic` decides when coordinator mode is justified (vs. single-agent). The coordinator uses the existing `AgentLoop` and `ToolRegistry.scoped()` infrastructure.

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2, pytest, ruff. No new dependencies.

---

## Task 0: Coordinator Models and Directory Structure

**Files:**
- Create: `src/norn/coordinator/__init__.py`
- Create: `src/norn/coordinator/models.py`
- Create: `tests/test_coordinator/__init__.py`
- Create: `tests/test_coordinator/test_models.py`

**Step 1: Create directories**

```bash
mkdir -p src/norn/coordinator tests/test_coordinator
touch src/norn/coordinator/__init__.py tests/test_coordinator/__init__.py
```

**Step 2: Write the failing tests**

```python
# tests/test_coordinator/test_models.py
"""Tests for coordinator models."""

from norn.coordinator.models import (
    CoordinatorPhase,
    CoordinatorPlan,
    CoordinatorResult,
    WorkerAssignment,
    WorkerResult,
)


def test_coordinator_phase_ordering():
    """Phases follow R->S->I->V order."""
    phases = list(CoordinatorPhase)
    assert phases == [
        CoordinatorPhase.RESEARCH,
        CoordinatorPhase.SYNTHESIS,
        CoordinatorPhase.IMPLEMENTATION,
        CoordinatorPhase.VERIFICATION,
    ]


def test_worker_assignment_creation():
    assignment = WorkerAssignment(
        worker_id="research-1",
        phase=CoordinatorPhase.RESEARCH,
        task="Investigate the auth module structure",
        tools=["file_read", "grep", "glob"],
    )
    assert assignment.worker_id == "research-1"
    assert assignment.phase == CoordinatorPhase.RESEARCH
    assert len(assignment.tools) == 3


def test_worker_assignment_defaults():
    assignment = WorkerAssignment(
        worker_id="w1",
        phase=CoordinatorPhase.RESEARCH,
        task="do something",
    )
    assert assignment.tools == []
    assert assignment.scratchpad_section == ""


def test_worker_result_success():
    result = WorkerResult(
        worker_id="w1",
        phase=CoordinatorPhase.RESEARCH,
        success=True,
        output="Found 3 modules to refactor.",
    )
    assert result.success is True
    assert result.error is None


def test_worker_result_failure():
    result = WorkerResult(
        worker_id="w1",
        phase=CoordinatorPhase.IMPLEMENTATION,
        success=False,
        error="Tool execution failed: file not found",
    )
    assert result.success is False
    assert "file not found" in result.error


def test_coordinator_plan_creation():
    plan = CoordinatorPlan(
        task_description="Refactor the auth module across 5 files",
        phases={
            CoordinatorPhase.RESEARCH: [
                WorkerAssignment(
                    worker_id="r1",
                    phase=CoordinatorPhase.RESEARCH,
                    task="Analyze auth module dependencies",
                    tools=["file_read", "grep"],
                ),
            ],
            CoordinatorPhase.SYNTHESIS: [],
            CoordinatorPhase.IMPLEMENTATION: [],
            CoordinatorPhase.VERIFICATION: [],
        },
    )
    assert len(plan.phases[CoordinatorPhase.RESEARCH]) == 1
    assert plan.phases[CoordinatorPhase.SYNTHESIS] == []


def test_coordinator_result_success():
    result = CoordinatorResult(
        success=True,
        summary="Refactored auth module: 5 files updated, all tests pass.",
        phase_results={
            CoordinatorPhase.RESEARCH: [
                WorkerResult(
                    worker_id="r1",
                    phase=CoordinatorPhase.RESEARCH,
                    success=True,
                    output="Found deps.",
                )
            ],
        },
    )
    assert result.success is True
    assert result.total_workers == 1


def test_coordinator_result_failure():
    result = CoordinatorResult(
        success=False,
        summary="Failed during implementation.",
        error="Worker impl-1 failed.",
    )
    assert result.success is False
    assert result.total_workers == 0
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_coordinator/test_models.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'norn.coordinator.models'`

**Step 4: Write minimal implementation**

```python
# src/norn/coordinator/models.py
"""Coordinator models for multi-agent orchestration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, computed_field


class CoordinatorPhase(StrEnum):
    """The four phases of the coordinator pipeline."""

    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"


class WorkerAssignment(BaseModel):
    """A task assigned to a worker for a specific phase."""

    worker_id: str
    phase: CoordinatorPhase
    task: str
    tools: list[str] = []
    scratchpad_section: str = ""


class WorkerResult(BaseModel):
    """Result from a worker's execution."""

    worker_id: str
    phase: CoordinatorPhase
    success: bool
    output: str | None = None
    error: str | None = None


class CoordinatorPlan(BaseModel):
    """A plan decomposing a task into phased worker assignments."""

    task_description: str
    phases: dict[CoordinatorPhase, list[WorkerAssignment]] = {}


class CoordinatorResult(BaseModel):
    """Result of a complete coordinator run."""

    success: bool
    summary: str
    error: str | None = None
    phase_results: dict[CoordinatorPhase, list[WorkerResult]] = {}

    @computed_field
    @property
    def total_workers(self) -> int:
        """Total number of workers that ran across all phases."""
        return sum(len(results) for results in self.phase_results.values())
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_coordinator/test_models.py -v`
Expected: 8 PASSED

**Step 6: Run ruff**

Run: `uv run ruff check src/norn/coordinator/models.py`

**Step 7: Commit**

```bash
git add src/norn/coordinator/ tests/test_coordinator/
git commit -m "feat(coordinator): add coordinator models with phases, assignments, and results"
```

---

## Task 1: Scratchpad -- Filesystem-Based Worker Communication

**Files:**
- Create: `src/norn/coordinator/scratchpad.py`
- Create: `tests/test_coordinator/test_scratchpad.py`

The Scratchpad manages a temp directory with sections for each phase. Workers write to their assigned section. The coordinator reads all sections.

**Step 1: Write the failing tests**

```python
# tests/test_coordinator/test_scratchpad.py
"""Tests for the coordinator scratchpad."""

import pytest

from norn.coordinator.scratchpad import Scratchpad


@pytest.fixture
def scratchpad(tmp_path):
    """Create a scratchpad in a temp directory."""
    return Scratchpad(base_dir=tmp_path / "scratchpad")


class TestInit:
    def test_creates_directory_structure(self, scratchpad):
        """Scratchpad creates phase directories on init."""
        scratchpad.ensure_dirs()
        assert (scratchpad.base_dir / "research").is_dir()
        assert (scratchpad.base_dir / "synthesis").is_dir()
        assert (scratchpad.base_dir / "implementation").is_dir()
        assert (scratchpad.base_dir / "verification").is_dir()

    def test_creates_base_dir(self, scratchpad):
        """Base directory is created if it doesn't exist."""
        scratchpad.ensure_dirs()
        assert scratchpad.base_dir.is_dir()


class TestReadWrite:
    def test_write_and_read_section(self, scratchpad):
        """Write to a section and read it back."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "worker-1-findings.md", "# Findings\n\nFound 3 issues.")
        content = scratchpad.read("research", "worker-1-findings.md")
        assert "Found 3 issues" in content

    def test_read_nonexistent_returns_none(self, scratchpad):
        """Reading a file that doesn't exist returns None."""
        scratchpad.ensure_dirs()
        assert scratchpad.read("research", "nonexistent.md") is None

    def test_write_multiple_files(self, scratchpad):
        """Multiple workers can write to the same section."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "worker-a.md", "Findings A")
        scratchpad.write("research", "worker-b.md", "Findings B")
        assert scratchpad.read("research", "worker-a.md") == "Findings A"
        assert scratchpad.read("research", "worker-b.md") == "Findings B"

    def test_overwrite_existing(self, scratchpad):
        """Writing to an existing file overwrites it."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "notes.md", "v1")
        scratchpad.write("research", "notes.md", "v2")
        assert scratchpad.read("research", "notes.md") == "v2"


class TestListAndCollect:
    def test_list_section_files(self, scratchpad):
        """List all files in a section."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "a.md", "x")
        scratchpad.write("research", "b.md", "y")
        files = scratchpad.list_files("research")
        assert set(files) == {"a.md", "b.md"}

    def test_list_empty_section(self, scratchpad):
        """Empty section returns empty list."""
        scratchpad.ensure_dirs()
        assert scratchpad.list_files("research") == []

    def test_collect_section(self, scratchpad):
        """Collect all content from a section into a dict."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "w1.md", "Result 1")
        scratchpad.write("research", "w2.md", "Result 2")
        collected = scratchpad.collect("research")
        assert collected == {"w1.md": "Result 1", "w2.md": "Result 2"}

    def test_collect_empty_section(self, scratchpad):
        """Collect from empty section returns empty dict."""
        scratchpad.ensure_dirs()
        assert scratchpad.collect("research") == {}

    def test_collect_all(self, scratchpad):
        """Collect everything across all sections."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "r1.md", "Research 1")
        scratchpad.write("synthesis", "spec.md", "Spec content")
        all_content = scratchpad.collect_all()
        assert "research" in all_content
        assert "synthesis" in all_content
        assert all_content["research"]["r1.md"] == "Research 1"
        assert all_content["synthesis"]["spec.md"] == "Spec content"


class TestCleanup:
    def test_clear_section(self, scratchpad):
        """Clear removes all files from a section."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "a.md", "x")
        scratchpad.write("research", "b.md", "y")
        scratchpad.clear("research")
        assert scratchpad.list_files("research") == []

    def test_clear_all(self, scratchpad):
        """Clear all sections."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "a.md", "x")
        scratchpad.write("synthesis", "b.md", "y")
        scratchpad.clear_all()
        assert scratchpad.list_files("research") == []
        assert scratchpad.list_files("synthesis") == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coordinator/test_scratchpad.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/coordinator/scratchpad.py
"""Scratchpad: filesystem-based communication between coordinator and workers."""

from __future__ import annotations

from pathlib import Path

from norn.coordinator.models import CoordinatorPhase

_SECTIONS = [phase.value for phase in CoordinatorPhase]


class Scratchpad:
    """Filesystem-backed scratchpad for inter-worker communication."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def ensure_dirs(self) -> None:
        """Create the scratchpad directory structure."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for section in _SECTIONS:
            (self.base_dir / section).mkdir(exist_ok=True)

    def write(self, section: str, filename: str, content: str) -> None:
        """Write a file to a scratchpad section."""
        path = self.base_dir / section / filename
        path.write_text(content)

    def read(self, section: str, filename: str) -> str | None:
        """Read a file from a scratchpad section. Returns None if missing."""
        path = self.base_dir / section / filename
        if not path.exists():
            return None
        return path.read_text()

    def list_files(self, section: str) -> list[str]:
        """List all files in a section."""
        section_dir = self.base_dir / section
        if not section_dir.exists():
            return []
        return sorted(p.name for p in section_dir.iterdir() if p.is_file())

    def collect(self, section: str) -> dict[str, str]:
        """Collect all file contents from a section."""
        result: dict[str, str] = {}
        for filename in self.list_files(section):
            content = self.read(section, filename)
            if content is not None:
                result[filename] = content
        return result

    def collect_all(self) -> dict[str, dict[str, str]]:
        """Collect all content across all sections."""
        return {section: self.collect(section) for section in _SECTIONS if self.list_files(section)}

    def clear(self, section: str) -> None:
        """Remove all files from a section."""
        section_dir = self.base_dir / section
        if section_dir.exists():
            for p in section_dir.iterdir():
                if p.is_file():
                    p.unlink()

    def clear_all(self) -> None:
        """Remove all files from all sections."""
        for section in _SECTIONS:
            self.clear(section)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coordinator/test_scratchpad.py -v`
Expected: 13 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/coordinator/scratchpad.py`

**Step 6: Commit**

```bash
git add src/norn/coordinator/scratchpad.py tests/test_coordinator/test_scratchpad.py
git commit -m "feat(coordinator): add filesystem-backed scratchpad for worker communication"
```

---

## Task 2: Worker -- Isolated Agent with Scoped Tools

**Files:**
- Create: `src/norn/coordinator/worker.py`
- Create: `tests/test_coordinator/test_worker.py`

A Worker wraps an `AgentLoop` with a scoped `ToolRegistry` and writes its output to its assigned scratchpad section. Workers are ephemeral -- created per-assignment, run once, discarded.

**Step 1: Write the failing tests**

```python
# tests/test_coordinator/test_worker.py
"""Tests for coordinator workers."""

from unittest.mock import AsyncMock

import pytest

from norn.coordinator.models import CoordinatorPhase, WorkerAssignment, WorkerResult
from norn.coordinator.scratchpad import Scratchpad
from norn.coordinator.worker import Worker
from norn.core.models import LLMResponse
from norn.tools.registry import ToolRegistry


class MockReadTool:
    name = "file_read"
    description = "Read a file"
    risk_level = "low"

    class Input:
        pass

    input_model = Input

    async def execute(self, input, ctx):
        pass


class MockGrepTool:
    name = "grep"
    description = "Search files"
    risk_level = "low"

    class Input:
        pass

    input_model = Input

    async def execute(self, input, ctx):
        pass


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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coordinator/test_worker.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/coordinator/worker.py
"""Worker: isolated agent with scoped tools for coordinator tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from norn.coordinator.models import WorkerResult
from norn.core.agent import AgentLoop

if TYPE_CHECKING:
    from norn.coordinator.models import WorkerAssignment
    from norn.coordinator.scratchpad import Scratchpad
    from norn.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _build_worker_prompt(assignment: WorkerAssignment) -> str:
    """Build the system prompt for a worker."""
    return (
        f"You are a Norn worker (ID: {assignment.worker_id}) "
        f"in the {assignment.phase.value.upper()} phase.\n\n"
        f"## Your Task\n\n{assignment.task}\n\n"
        "## Rules\n\n"
        "- Focus exclusively on your assigned task.\n"
        "- Be concise and factual in your output.\n"
        "- Report your findings clearly.\n"
    )


class Worker:
    """An isolated agent that executes a single coordinator assignment."""

    def __init__(
        self,
        assignment: WorkerAssignment,
        llm: object,
        registry: ToolRegistry,
        scratchpad: Scratchpad,
        cwd: str = ".",
    ) -> None:
        self.assignment = assignment
        self.scratchpad = scratchpad

        # Scope the registry to only the assigned tools
        scoped_registry = registry.scoped(assignment.tools)

        self.agent = AgentLoop(
            llm=llm,
            registry=scoped_registry,
            system_prompt=_build_worker_prompt(assignment),
            cwd=cwd,
        )

    async def run(self) -> WorkerResult:
        """Execute the worker's assignment and write results to scratchpad."""
        try:
            response = await self.agent.run(self.assignment.task)
            output = response.content or ""

            # Write output to scratchpad
            if self.assignment.scratchpad_section:
                self.scratchpad.write(
                    self.assignment.phase.value,
                    self.assignment.scratchpad_section,
                    output,
                )

            return WorkerResult(
                worker_id=self.assignment.worker_id,
                phase=self.assignment.phase,
                success=True,
                output=output,
            )

        except Exception as e:
            logger.error("Worker %s failed: %s", self.assignment.worker_id, e)
            return WorkerResult(
                worker_id=self.assignment.worker_id,
                phase=self.assignment.phase,
                success=False,
                error=str(e),
            )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coordinator/test_worker.py -v`
Expected: 6 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/coordinator/worker.py`

**Step 6: Commit**

```bash
git add src/norn/coordinator/worker.py tests/test_coordinator/test_worker.py
git commit -m "feat(coordinator): add isolated worker with scoped tools and scratchpad output"
```

---

## Task 3: Activation Heuristic -- When to Use Coordinator

**Files:**
- Create: `src/norn/coordinator/heuristic.py`
- Create: `tests/test_coordinator/test_heuristic.py`

The heuristic decides whether a task is complex enough to warrant coordinator mode. It evaluates multiple indicators and requires >= 2 to activate.

**Step 1: Write the failing tests**

```python
# tests/test_coordinator/test_heuristic.py
"""Tests for the coordinator activation heuristic."""

from norn.coordinator.heuristic import ActivationHeuristic, ActivationResult


class TestIndicators:
    def test_long_text_detected(self):
        """Task text > 500 chars is an indicator."""
        h = ActivationHeuristic()
        text = "x" * 501
        result = h.evaluate(text)
        assert "long_text" in result.indicators

    def test_short_text_not_detected(self):
        h = ActivationHeuristic()
        result = h.evaluate("Fix the typo in README")
        assert "long_text" not in result.indicators

    def test_keywords_detected(self):
        """Keywords like 'refactor', 'across', 'multiple files' trigger."""
        h = ActivationHeuristic()
        result = h.evaluate("Refactor the auth module across multiple files")
        assert "keywords" in result.indicators

    def test_no_keywords(self):
        h = ActivationHeuristic()
        result = h.evaluate("Fix the typo")
        assert "keywords" not in result.indicators

    def test_file_count_detected(self):
        """Mentioning many files triggers the indicator."""
        h = ActivationHeuristic()
        text = "Update these files: " + ", ".join(f"src/mod{i}.py" for i in range(12))
        result = h.evaluate(text)
        assert "many_files" in result.indicators

    def test_few_files_not_detected(self):
        h = ActivationHeuristic()
        result = h.evaluate("Update src/main.py and src/utils.py")
        assert "many_files" not in result.indicators

    def test_explicit_parallel_keyword(self):
        """Words like 'parallel', 'concurrent', 'simultaneously' trigger."""
        h = ActivationHeuristic()
        result = h.evaluate("Run these checks in parallel across the codebase")
        assert "parallel_request" in result.indicators


class TestActivation:
    def test_activates_with_two_indicators(self):
        """Coordinator activates when >= 2 indicators are present."""
        h = ActivationHeuristic(threshold=2)
        # Long text + keywords
        text = "x" * 501 + " refactor across multiple modules"
        result = h.evaluate(text)
        assert result.should_activate is True

    def test_does_not_activate_with_one(self):
        """Single indicator is not enough."""
        h = ActivationHeuristic(threshold=2)
        result = h.evaluate("x" * 501)  # Only long text
        assert result.should_activate is False

    def test_does_not_activate_with_zero(self):
        h = ActivationHeuristic(threshold=2)
        result = h.evaluate("Fix typo")
        assert result.should_activate is False

    def test_custom_threshold(self):
        """Custom threshold works."""
        h = ActivationHeuristic(threshold=1)
        result = h.evaluate("Refactor the module")
        assert result.should_activate is True  # keywords alone is enough

    def test_result_includes_reason(self):
        h = ActivationHeuristic(threshold=2)
        text = "x" * 501 + " refactor across multiple files"
        result = h.evaluate(text)
        assert result.reason is not None
        assert len(result.reason) > 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coordinator/test_heuristic.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/coordinator/heuristic.py
"""Activation heuristic: decides when coordinator mode is justified."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_COORDINATOR_KEYWORDS = [
    r"\brefactor\b",
    r"\bacross\b",
    r"\bmultiple files\b",
    r"\bmulti-file\b",
    r"\bcodebase\b",
    r"\bmodules?\b.*\b(update|change|modify|rewrite)\b",
    r"\b(update|change|modify|rewrite)\b.*\bmodules?\b",
    r"\bsystem-wide\b",
    r"\bproject-wide\b",
]

_PARALLEL_KEYWORDS = [
    r"\bparallel\b",
    r"\bconcurrent(ly)?\b",
    r"\bsimultaneous(ly)?\b",
    r"\bin parallel\b",
]

# Matches paths like src/foo.py, tests/bar.ts, etc.
_FILE_PATTERN = re.compile(r"\b[\w/\\.-]+\.\w{1,5}\b")


@dataclass
class ActivationResult:
    """Result of the activation heuristic evaluation."""

    should_activate: bool
    indicators: list[str] = field(default_factory=list)
    reason: str = ""


class ActivationHeuristic:
    """Evaluates whether a task warrants coordinator mode."""

    def __init__(
        self,
        threshold: int = 2,
        min_text_length: int = 500,
        min_file_mentions: int = 10,
    ) -> None:
        self._threshold = threshold
        self._min_text_length = min_text_length
        self._min_file_mentions = min_file_mentions

    def evaluate(self, task_text: str) -> ActivationResult:
        """Evaluate whether coordinator mode should activate."""
        indicators: list[str] = []

        # Indicator 1: Long text
        if len(task_text) > self._min_text_length:
            indicators.append("long_text")

        # Indicator 2: Coordinator keywords
        text_lower = task_text.lower()
        if any(re.search(kw, text_lower) for kw in _COORDINATOR_KEYWORDS):
            indicators.append("keywords")

        # Indicator 3: Many file mentions
        file_matches = _FILE_PATTERN.findall(task_text)
        if len(file_matches) >= self._min_file_mentions:
            indicators.append("many_files")

        # Indicator 4: Parallel request
        if any(re.search(kw, text_lower) for kw in _PARALLEL_KEYWORDS):
            indicators.append("parallel_request")

        should_activate = len(indicators) >= self._threshold
        reason = (
            f"Activated: {len(indicators)}/{self._threshold} indicators "
            f"({', '.join(indicators)})"
            if should_activate
            else f"Not activated: {len(indicators)}/{self._threshold} indicators"
        )

        return ActivationResult(
            should_activate=should_activate,
            indicators=indicators,
            reason=reason,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coordinator/test_heuristic.py -v`
Expected: 10 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/coordinator/heuristic.py`

**Step 6: Commit**

```bash
git add src/norn/coordinator/heuristic.py tests/test_coordinator/test_heuristic.py
git commit -m "feat(coordinator): add activation heuristic with multi-indicator threshold"
```

---

## Task 4: Coordinator Prompts -- Phase-Specific LLM Instructions

**Files:**
- Create: `src/norn/coordinator/prompts.py`
- Create: `tests/test_coordinator/test_prompts.py`

Prompt templates for the coordinator at each phase transition, and for worker system prompts.

**Step 1: Write the failing tests**

```python
# tests/test_coordinator/test_prompts.py
"""Tests for coordinator prompt templates."""

from norn.coordinator.prompts import (
    build_coordinator_system_prompt,
    build_phase_prompt,
    build_worker_system_prompt,
)
from norn.coordinator.models import CoordinatorPhase


def test_coordinator_system_prompt_is_string():
    prompt = build_coordinator_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 100
    assert "coordinator" in prompt.lower()


def test_coordinator_system_prompt_has_rules():
    prompt = build_coordinator_system_prompt()
    assert "parallel" in prompt.lower()
    assert "worker" in prompt.lower()


def test_phase_prompt_research():
    prompt = build_phase_prompt(
        phase=CoordinatorPhase.RESEARCH,
        task="Refactor auth module",
        context="",
    )
    assert "research" in prompt.lower()
    assert "Refactor auth module" in prompt


def test_phase_prompt_synthesis_includes_findings():
    prompt = build_phase_prompt(
        phase=CoordinatorPhase.SYNTHESIS,
        task="Refactor auth module",
        context="## Research Findings\n\nWorker r1 found 3 modules.",
    )
    assert "synthesis" in prompt.lower()
    assert "Worker r1 found 3 modules" in prompt


def test_phase_prompt_implementation():
    prompt = build_phase_prompt(
        phase=CoordinatorPhase.IMPLEMENTATION,
        task="Refactor auth",
        context="## Specs\n\nSpec 1: Extract token validation.",
    )
    assert "implementation" in prompt.lower()
    assert "Extract token validation" in prompt


def test_phase_prompt_verification():
    prompt = build_phase_prompt(
        phase=CoordinatorPhase.VERIFICATION,
        task="Refactor auth",
        context="## Implementation\n\nWorker i1 updated 3 files.",
    )
    assert "verification" in prompt.lower()
    assert "updated 3 files" in prompt


def test_worker_system_prompt_includes_assignment():
    prompt = build_worker_system_prompt(
        worker_id="r1",
        phase=CoordinatorPhase.RESEARCH,
        task="Find all uses of the old auth API",
    )
    assert "r1" in prompt
    assert "RESEARCH" in prompt
    assert "Find all uses of the old auth API" in prompt


def test_worker_system_prompt_has_rules():
    prompt = build_worker_system_prompt(
        worker_id="w1",
        phase=CoordinatorPhase.IMPLEMENTATION,
        task="Update module X",
    )
    assert "focus" in prompt.lower() or "task" in prompt.lower()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coordinator/test_prompts.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/coordinator/prompts.py
"""Coordinator prompt templates for multi-agent orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from norn.coordinator.models import CoordinatorPhase


def build_coordinator_system_prompt() -> str:
    """Build the system prompt for the coordinator agent."""
    return """\
You are Norn's Coordinator -- a multi-agent orchestrator.

Your job is to decompose complex tasks into phases and assign work to parallel workers.

## Pipeline

1. RESEARCH: Workers investigate the codebase, gather context, identify scope
2. SYNTHESIS: You read research findings and create specific implementation specs
3. IMPLEMENTATION: Workers execute the specs (one per spec)
4. VERIFICATION: Workers test and validate the implementation

## Rules

1. **Parallelism is your superpower** -- always look for work that can run in parallel.
2. **Do NOT say "based on your findings"** -- read the actual findings from the scratchpad.
3. **You synthesize, workers execute** -- never implement directly.
4. **Each worker gets a specific, scoped task** -- no vague assignments.
5. **Scope tools per worker** -- research workers get read-only tools, impl workers get write tools.
6. **Fail fast** -- if a phase fails, stop and report rather than continuing blindly.

## Output Format

For each phase, return a JSON array of worker assignments:

```json
[
  {
    "worker_id": "r1",
    "task": "Find all usages of AuthService in src/",
    "tools": ["file_read", "grep", "glob"],
    "scratchpad_section": "r1-findings.md"
  }
]
```
"""


def build_phase_prompt(
    phase: CoordinatorPhase,
    task: str,
    context: str,
) -> str:
    """Build the user prompt for a specific phase transition."""
    phase_instructions = {
        "research": (
            "## Phase: RESEARCH\n\n"
            "You are starting the research phase. Analyze the task and create "
            "worker assignments to investigate the codebase.\n\n"
            "Workers should gather context, find relevant files, identify scope, "
            "and report findings. Use read-only tools (file_read, grep, glob).\n\n"
        ),
        "synthesis": (
            "## Phase: SYNTHESIS\n\n"
            "Research is complete. Read the findings below and create specific, "
            "actionable implementation specs. Each spec should be a self-contained "
            "unit of work that one worker can execute.\n\n"
        ),
        "implementation": (
            "## Phase: IMPLEMENTATION\n\n"
            "Specs are ready. Create worker assignments to implement each spec. "
            "Workers get write tools (file_write, file_edit) plus read tools. "
            "Each worker implements one spec.\n\n"
        ),
        "verification": (
            "## Phase: VERIFICATION\n\n"
            "Implementation is complete. Create worker assignments to verify the work. "
            "Workers should run tests, check for regressions, and validate the changes.\n\n"
        ),
    }

    parts = [phase_instructions.get(phase, "")]
    parts.append(f"## Task\n\n{task}\n\n")

    if context.strip():
        parts.append(f"## Context from Previous Phases\n\n{context}\n\n")

    parts.append("Create worker assignments as JSON.")
    return "\n".join(parts)


def build_worker_system_prompt(
    worker_id: str,
    phase: CoordinatorPhase,
    task: str,
) -> str:
    """Build the system prompt for an individual worker."""
    return (
        f"You are a Norn worker (ID: {worker_id}) "
        f"in the {phase.value.upper()} phase.\n\n"
        f"## Your Task\n\n{task}\n\n"
        "## Rules\n\n"
        "- Focus exclusively on your assigned task.\n"
        "- Be concise and factual in your output.\n"
        "- Report your findings clearly.\n"
        "- Do not take actions outside your assigned scope.\n"
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coordinator/test_prompts.py -v`
Expected: 8 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/coordinator/prompts.py`

**Step 6: Commit**

```bash
git add src/norn/coordinator/prompts.py tests/test_coordinator/test_prompts.py
git commit -m "feat(coordinator): add prompt templates for coordinator and worker phases"
```

---

## Task 5: Coordinator Engine -- R→S→I→V Pipeline

**Files:**
- Create: `src/norn/coordinator/engine.py`
- Create: `tests/test_coordinator/test_engine.py`

The main orchestrator that runs the 4-phase pipeline: creates scratchpad, asks LLM for worker assignments at each phase, dispatches workers, collects results, feeds context to next phase.

**Step 1: Write the failing tests**

```python
# tests/test_coordinator/test_engine.py
"""Tests for the coordinator engine."""

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
    llm.complete = AsyncMock(
        return_value=LLMResponse(content=content, tool_calls=[])
    )
    return llm


class TestPhaseExecution:
    @pytest.mark.asyncio
    async def test_research_phase_dispatches_workers(self, scratchpad, registry):
        """Research phase creates and runs workers."""
        coordinator_llm = AsyncMock()
        coordinator_llm.complete = AsyncMock(
            return_value=_make_assignment_response([
                {
                    "worker_id": "r1",
                    "task": "Find auth modules",
                    "tools": [],
                    "scratchpad_section": "r1-findings.md",
                },
            ])
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
            return_value=_make_assignment_response([
                {
                    "worker_id": "r1",
                    "task": "Find modules",
                    "tools": [],
                    "scratchpad_section": "r1.md",
                },
            ])
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
        coordinator_llm.complete = AsyncMock(
            return_value=_make_assignment_response([])
        )
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
                _make_assignment_response([
                    {"worker_id": "r1", "task": "Research", "tools": [], "scratchpad_section": "r1.md"},
                ]),
                # Synthesis phase assignments
                _make_assignment_response([
                    {"worker_id": "s1", "task": "Write spec", "tools": [], "scratchpad_section": "s1.md"},
                ]),
                # Implementation phase assignments
                _make_assignment_response([
                    {"worker_id": "i1", "task": "Implement", "tools": [], "scratchpad_section": "i1.md"},
                ]),
                # Verification phase assignments
                _make_assignment_response([
                    {"worker_id": "v1", "task": "Verify", "tools": [], "scratchpad_section": "v1.md"},
                ]),
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
        coordinator_llm.complete = AsyncMock(
            return_value=LLMResponse(content="not valid json")
        )
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
                _make_assignment_response([
                    {"worker_id": "r1", "task": "Research", "tools": [], "scratchpad_section": "r1.md"},
                ]),
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coordinator/test_engine.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/coordinator/engine.py
"""Coordinator engine: R->S->I->V multi-agent pipeline."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from norn.coordinator.models import (
    CoordinatorPhase,
    CoordinatorResult,
    WorkerAssignment,
    WorkerResult,
)
from norn.coordinator.prompts import build_coordinator_system_prompt, build_phase_prompt
from norn.coordinator.worker import Worker
from norn.core.models import Message, Role

if TYPE_CHECKING:
    from norn.coordinator.scratchpad import Scratchpad
    from norn.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_PHASE_ORDER = [
    CoordinatorPhase.RESEARCH,
    CoordinatorPhase.SYNTHESIS,
    CoordinatorPhase.IMPLEMENTATION,
    CoordinatorPhase.VERIFICATION,
]


class CoordinatorEngine:
    """Orchestrates the four-phase R->S->I->V pipeline."""

    def __init__(
        self,
        coordinator_llm: object,
        worker_llm: object,
        registry: ToolRegistry,
        scratchpad: Scratchpad,
        cwd: str = ".",
    ) -> None:
        self._coordinator_llm = coordinator_llm
        self._worker_llm = worker_llm
        self._registry = registry
        self._scratchpad = scratchpad
        self._cwd = cwd

    def _build_phase_context(self) -> str:
        """Build context string from all scratchpad content so far."""
        all_content = self._scratchpad.collect_all()
        if not all_content:
            return ""

        parts: list[str] = []
        for section, files in all_content.items():
            parts.append(f"## {section.upper()} Phase Results\n")
            for filename, content in sorted(files.items()):
                parts.append(f"### {filename}\n\n{content}\n")
        return "\n".join(parts)

    async def _get_assignments(
        self,
        phase: CoordinatorPhase,
        task: str,
        context: str,
    ) -> list[WorkerAssignment]:
        """Ask the coordinator LLM for worker assignments for a phase."""
        system_prompt = build_coordinator_system_prompt()
        user_prompt = build_phase_prompt(phase=phase, task=task, context=context)

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_prompt),
        ]

        response = await self._coordinator_llm.complete(messages=messages, tools=None)
        content = response.content or "[]"

        # Parse JSON (handle markdown code blocks)
        json_str = content.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else "[]"

        raw_assignments: list[dict[str, Any]] = json.loads(json_str)

        return [
            WorkerAssignment(
                worker_id=a.get("worker_id", f"{phase.value[0]}{i}"),
                phase=phase,
                task=a.get("task", ""),
                tools=a.get("tools", []),
                scratchpad_section=a.get("scratchpad_section", f"{phase.value[0]}{i}.md"),
            )
            for i, a in enumerate(raw_assignments)
        ]

    async def _run_phase(
        self,
        phase: CoordinatorPhase,
        task: str,
        context: str,
    ) -> list[WorkerResult]:
        """Run a single phase: get assignments, dispatch workers, collect results."""
        assignments = await self._get_assignments(phase, task, context)

        if not assignments:
            return []

        # Run workers sequentially for now (parallel via asyncio.gather in future)
        results: list[WorkerResult] = []
        for assignment in assignments:
            worker = Worker(
                assignment=assignment,
                llm=self._worker_llm,
                registry=self._registry,
                scratchpad=self._scratchpad,
                cwd=self._cwd,
            )
            result = await worker.run()
            results.append(result)

        return results

    async def coordinate(self, task: str) -> CoordinatorResult:
        """Run the full R->S->I->V pipeline for a task."""
        phase_results: dict[CoordinatorPhase, list[WorkerResult]] = {}

        try:
            for phase in _PHASE_ORDER:
                context = self._build_phase_context()
                results = await self._run_phase(phase, task, context)
                phase_results[phase] = results

            # Build summary
            total = sum(len(r) for r in phase_results.values())
            failed = sum(
                1
                for results in phase_results.values()
                for r in results
                if not r.success
            )
            summary = (
                f"Coordinator completed: {total} workers across 4 phases"
                f"{f', {failed} failed' if failed else ', all succeeded'}."
            )

            return CoordinatorResult(
                success=True,
                summary=summary,
                phase_results=phase_results,
            )

        except json.JSONDecodeError as e:
            logger.error("Coordinator failed: invalid JSON from LLM: %s", e)
            return CoordinatorResult(
                success=False,
                summary="Coordinator failed during planning.",
                error=f"Invalid JSON from coordinator LLM: {e}",
                phase_results=phase_results,
            )
        except Exception as e:
            logger.error("Coordinator failed: %s", e)
            return CoordinatorResult(
                success=False,
                summary="Coordinator failed.",
                error=str(e),
                phase_results=phase_results,
            )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coordinator/test_engine.py -v`
Expected: 8 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/coordinator/engine.py`

**Step 6: Commit**

```bash
git add src/norn/coordinator/engine.py tests/test_coordinator/test_engine.py
git commit -m "feat(coordinator): add four-phase R->S->I->V pipeline engine"
```

---

## Task 6: Config Integration -- CoordinatorConfig

**Files:**
- Modify: `src/norn/core/config.py`
- Modify: `tests/test_core/test_config.py`

Add coordinator configuration to `NornConfig`.

**Step 1: Write the failing tests**

Add to `tests/test_core/test_config.py`:

```python
def test_coordinator_config_defaults():
    """NornConfig should have coordinator config with defaults."""
    config = NornConfig()
    assert config.coordinator.enabled is False
    assert config.coordinator.activation_threshold == 2
    assert config.coordinator.max_workers_per_phase == 5


def test_coordinator_config_custom():
    """Coordinator config should be overridable."""
    config = NornConfig(
        coordinator={"enabled": True, "activation_threshold": 1, "max_workers_per_phase": 3}
    )
    assert config.coordinator.enabled is True
    assert config.coordinator.activation_threshold == 1
    assert config.coordinator.max_workers_per_phase == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: FAIL -- `NornConfig` has no field `coordinator`

**Step 3: Modify config.py**

Add a `CoordinatorConfig` model and include it in `NornConfig`:

```python
class CoordinatorConfig(BaseModel):
    enabled: bool = False
    activation_threshold: int = 2
    max_workers_per_phase: int = 5
    worker_model: str | None = None  # None = use main LLM
```

Add to `NornConfig`:

```python
class NornConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    flags: FlagsConfig = FlagsConfig()
    memory: MemorySystemConfig = MemorySystemConfig()
    coordinator: CoordinatorConfig = CoordinatorConfig()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: ALL PASSED (8 tests: 6 existing + 2 new)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED

**Step 6: Commit**

```bash
git add src/norn/core/config.py tests/test_core/test_config.py
git commit -m "feat(config): add coordinator configuration with activation threshold and worker limits"
```

---

## Task 7: CLI Wiring -- Coordinate Command

**Files:**
- Modify: `src/norn/cli/main.py`

Add a `norn coordinate` command for manual coordinator mode. Update `config` command to show coordinator status.

**Step 1: Update CLI**

Key additions to `src/norn/cli/main.py`:

```python
@app.command()
def coordinate(prompt: str = typer.Argument(help="Task to coordinate")) -> None:
    """Run a task using multi-agent coordinator mode."""
    config = NornConfig.load()
    config.apply_env_overrides()

    if not config.coordinator.enabled:
        console.print("[yellow]Coordinator is disabled. Enable with coordinator.enabled=true in config.[/yellow]")
        raise typer.Exit(1)

    provider = _build_provider(config)
    flag_registry = _build_flag_registry(config)
    registry = _build_registry(flag_registry)

    from norn.coordinator.engine import CoordinatorEngine
    from norn.coordinator.scratchpad import Scratchpad

    import tempfile
    scratchpad_dir = Path(tempfile.mkdtemp(prefix="norn-coord-"))
    scratchpad = Scratchpad(base_dir=scratchpad_dir)
    scratchpad.ensure_dirs()

    engine = CoordinatorEngine(
        coordinator_llm=provider,
        worker_llm=provider,
        registry=registry,
        scratchpad=scratchpad,
        cwd=str(Path.cwd()),
    )

    async def _coordinate() -> None:
        with console.status("[dim]Coordinating...[/dim]"):
            result = await engine.coordinate(prompt)

        if result.success:
            console.print(f"[green]Coordinator complete:[/green] {result.summary}")
            for phase, results in result.phase_results.items():
                console.print(f"\n  [bold]{phase.value.upper()}[/bold]:")
                for wr in results:
                    status = "[green]OK[/green]" if wr.success else "[red]FAIL[/red]"
                    console.print(f"    {status} {wr.worker_id}: {(wr.output or wr.error or '')[:80]}")
        else:
            console.print(f"[red]Coordinator failed:[/red] {result.error}")

    asyncio.run(_coordinate())
```

Update `config` command to display coordinator status:

```python
console.print(f"  Coordinator: {'enabled' if cfg.coordinator.enabled else 'disabled'}")
if cfg.coordinator.enabled:
    console.print(f"  Coord threshold: {cfg.coordinator.activation_threshold}")
    console.print(f"  Max workers/phase: {cfg.coordinator.max_workers_per_phase}")
```

**Step 2: Verify CLI**

Run: `uv run norn --help` (should show `coordinate` command)
Run: `uv run norn config` (should show coordinator status)

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED

**Step 4: Commit**

```bash
git add src/norn/cli/main.py
git commit -m "feat(cli): add coordinate command and coordinator status to config display"
```

---

## Task 8: Integration Tests

**Files:**
- Modify: `tests/test_integration.py`

Add integration tests that exercise the full coordinator pipeline.

**Step 1: Write integration tests**

Add to `tests/test_integration.py`:

```python
from norn.coordinator.engine import CoordinatorEngine
from norn.coordinator.heuristic import ActivationHeuristic
from norn.coordinator.models import CoordinatorPhase
from norn.coordinator.scratchpad import Scratchpad


@pytest.mark.asyncio
async def test_coordinator_full_pipeline(tmp_path):
    """Full coordinator pipeline: R->S->I->V with mock LLMs."""
    scratchpad = Scratchpad(base_dir=tmp_path / "scratchpad")
    scratchpad.ensure_dirs()
    registry = ToolRegistry()

    coordinator_llm = AsyncMock()
    coordinator_llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(content=json.dumps([
                {"worker_id": "r1", "task": "Research", "tools": [], "scratchpad_section": "r1.md"},
            ])),
            LLMResponse(content=json.dumps([
                {"worker_id": "s1", "task": "Spec", "tools": [], "scratchpad_section": "s1.md"},
            ])),
            LLMResponse(content=json.dumps([
                {"worker_id": "i1", "task": "Implement", "tools": [], "scratchpad_section": "i1.md"},
            ])),
            LLMResponse(content=json.dumps([
                {"worker_id": "v1", "task": "Verify", "tools": [], "scratchpad_section": "v1.md"},
            ])),
        ]
    )
    worker_llm = AsyncMock()
    worker_llm.complete = AsyncMock(
        return_value=LLMResponse(content="Done.", tool_calls=[])
    )

    engine = CoordinatorEngine(
        coordinator_llm=coordinator_llm,
        worker_llm=worker_llm,
        registry=registry,
        scratchpad=scratchpad,
    )
    result = await engine.coordinate("Refactor auth module across 5 files")
    assert result.success is True
    assert result.total_workers == 4


def test_activation_heuristic_integration():
    """Heuristic correctly identifies complex tasks."""
    h = ActivationHeuristic(threshold=2)

    # Simple task: should NOT activate
    simple = h.evaluate("Fix typo in README")
    assert simple.should_activate is False

    # Complex task: should activate
    complex_task = (
        "Refactor the authentication module across multiple files. "
        "The auth service in src/auth/service.py needs to be split into "
        + "x" * 500
    )
    complex_result = h.evaluate(complex_task)
    assert complex_result.should_activate is True


def test_scratchpad_isolation(tmp_path):
    """Workers can only write to their assigned section."""
    sp = Scratchpad(base_dir=tmp_path / "scratchpad")
    sp.ensure_dirs()

    # Worker writes to research
    sp.write("research", "r1.md", "Research findings")
    # Worker writes to implementation
    sp.write("implementation", "i1.md", "Implementation output")

    # Sections are isolated
    assert sp.read("research", "r1.md") == "Research findings"
    assert sp.read("research", "i1.md") is None
    assert sp.read("implementation", "i1.md") == "Implementation output"
    assert sp.read("implementation", "r1.md") is None
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: ALL PASSED (11 existing + 3 new = 14)

**Step 3: Run full test suite + ruff**

Run: `uv run pytest -v && uv run ruff check src/ tests/`
Expected: ALL PASSED

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for coordinator pipeline and activation heuristic"
```

---

## Task 9: Final Verification and Cleanup

**Step 1: Run full test suite**

```bash
uv run pytest -v
```
Expected: ALL PASSED

**Step 2: Run ruff**

```bash
uv run ruff check src/ tests/
```
Expected: 0 errors (except pre-existing E402 in CLI)

**Step 3: Verify git log**

```bash
git log --oneline
```
Expected: Clean atomic commits from Phase 4

**Step 4: Manual smoke test**

```bash
uv run norn --help
uv run norn config
```
Expected: Shows `coordinate` command and coordinator status

**Step 5: Final commit (if any cleanup needed)**

```bash
# Only if cleanup was needed
git add -A && git commit -m "chore: phase 4 cleanup"
```

---

## Summary

| Task | Component | Tests | Files |
|------|-----------|-------|-------|
| 0 | Coordinator models | 8 | `coordinator/models.py` |
| 1 | Scratchpad | 13 | `coordinator/scratchpad.py` |
| 2 | Worker | 6 | `coordinator/worker.py` |
| 3 | Activation heuristic | 10 | `coordinator/heuristic.py` |
| 4 | Coordinator prompts | 8 | `coordinator/prompts.py` |
| 5 | Coordinator engine | 8 | `coordinator/engine.py` |
| 6 | Config integration | 2 | `core/config.py` (modify) |
| 7 | CLI wiring | manual | `cli/main.py` (modify) |
| 8 | Integration tests | 3 | `test_integration.py` (modify) |
| 9 | Final verification | manual | -- |

**Total: ~58 new tests, 9 commits, 6 new files, 3 modified files**

**No new dependencies** -- uses only asyncio, Pydantic, and existing infrastructure.
