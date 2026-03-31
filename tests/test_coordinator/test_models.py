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
