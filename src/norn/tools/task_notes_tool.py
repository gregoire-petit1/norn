"""Tool exposing the persistent task scratchpad to the agent (wave 2 — A1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from norn.core.task_notes import ACTION_SECTIONS
from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult

if TYPE_CHECKING:
    from norn.core.task_notes import TaskNotes


class TaskNotesInput(BaseModel):
    """Input for the task_notes tool."""

    action: Literal["read", "set_goal", "set_plan", "record_decision", "note_progress"]
    content: str = ""


class TaskNotesTool:
    """Maintain a durable plan/progress scratchpad that survives compaction.

    Use it on multi-step tasks: set the goal and plan up front, record key
    decisions, and note progress after each milestone. The notes are shown
    back to you every turn even after older messages are dropped from context.
    """

    name = "task_notes"
    description = (
        "Maintain a persistent task scratchpad that survives context compaction. "
        "Actions: read; set_goal; set_plan (overwrite the step list); "
        "record_decision (append); note_progress (append). "
        "Use on multi-step tasks to stay oriented across many turns."
    )
    risk_level = RiskLevel.LOW
    input_model = TaskNotesInput

    def __init__(self, notes: TaskNotes) -> None:
        self._notes = notes

    async def execute(self, input: TaskNotesInput, ctx: ToolContext) -> ToolResult:
        try:
            if input.action == "read":
                content = self._notes.read()
                return ToolResult(output=content or "Task notes are empty.")
            section = ACTION_SECTIONS[input.action]
            self._notes.write_section(section, input.content)
            return ToolResult(output=f"Updated {section}.")
        except Exception as e:
            return ToolResult(error=str(e), error_type=ToolErrorType.EXECUTION_ERROR.value)
