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
