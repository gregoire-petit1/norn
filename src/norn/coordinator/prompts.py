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
