"""File write tool for the Norn agent."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


class FileWriteInput(BaseModel):
    """Input for file write operations."""

    path: str
    content: str


class FileWriteTool:
    """Create or overwrite a file."""

    name = "file_write"
    description = "Create a new file or overwrite an existing one with the provided content."
    risk_level = RiskLevel.MEDIUM
    input_model = FileWriteInput

    async def execute(self, input: FileWriteInput, ctx: ToolContext) -> ToolResult:
        target = Path(input.path)
        if not target.is_absolute():
            target = Path(ctx.cwd) / target

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(input.content, encoding="utf-8")
            return ToolResult(output=f"Wrote {len(input.content)} bytes to {target}")
        except Exception as e:
            return ToolResult(error=str(e), error_type=ToolErrorType.EXECUTION_ERROR.value)
