"""Glob tool for file pattern matching."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


class GlobInput(BaseModel):
    """Input for glob pattern matching."""

    pattern: str
    path: str | None = None


class GlobTool:
    """Find files matching a glob pattern."""

    name = "glob"
    description = "Find files matching a glob pattern in a directory."
    risk_level = RiskLevel.LOW
    input_model = GlobInput

    async def execute(self, input: GlobInput, ctx: ToolContext) -> ToolResult:
        base = Path(input.path) if input.path else Path(ctx.cwd)
        if not base.is_absolute():
            base = Path(ctx.cwd) / base

        if not base.exists():
            return ToolResult(
                error=f"Directory not found: {base}",
                error_type=ToolErrorType.FILE_NOT_FOUND.value,
            )

        matches = sorted(base.glob(input.pattern))
        if not matches:
            return ToolResult(output="No matches found.")

        output = "\n".join(str(m) for m in matches[:500])
        return ToolResult(output=output)
