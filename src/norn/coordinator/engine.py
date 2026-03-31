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
from norn.core.utils import extract_json

if TYPE_CHECKING:
    from norn.coordinator.scratchpad import Scratchpad
    from norn.core.llm import LLMProvider
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
        coordinator_llm: LLMProvider,
        worker_llm: LLMProvider,
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

        raw_assignments: list[dict[str, Any]] = json.loads(extract_json(content))

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
            failed = sum(1 for results in phase_results.values() for r in results if not r.success)
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
