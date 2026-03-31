"""Tests for coordinator prompt templates."""

from norn.coordinator.models import CoordinatorPhase
from norn.coordinator.prompts import (
    build_coordinator_system_prompt,
    build_phase_prompt,
    build_worker_system_prompt,
)


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
