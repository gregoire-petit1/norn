"""Worker: isolated agent with scoped tools for coordinator tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from norn.coordinator.models import WorkerResult
from norn.coordinator.prompts import build_worker_system_prompt
from norn.core.agent import AgentLoop

if TYPE_CHECKING:
    from norn.coordinator.models import WorkerAssignment
    from norn.coordinator.scratchpad import Scratchpad
    from norn.core.llm import LLMProvider
    from norn.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Worker:
    """An isolated agent that executes a single coordinator assignment."""

    def __init__(
        self,
        assignment: WorkerAssignment,
        llm: LLMProvider,
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
            system_prompt=build_worker_system_prompt(
                worker_id=assignment.worker_id,
                phase=assignment.phase,
                task=assignment.task,
            ),
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
