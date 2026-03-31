"""File read tool for the Norn agent."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class FileReadInput(BaseModel):
    """Input for file read operations."""

    path: str
    offset: int = 0
    limit: int = 2000


class FileReadTool:
    """Read a file or list a directory."""

    name = "file_read"
    description = "Read a file's contents or list a directory's entries."
    risk_level = RiskLevel.LOW
    input_model = FileReadInput

    async def execute(self, input: FileReadInput, ctx: ToolContext) -> ToolResult:
        target = Path(input.path)
        if not target.is_absolute():
            target = Path(ctx.cwd) / target

        if not target.exists():
            return ToolResult(error=f"Path not found: {target}")

        if target.is_dir():
            entries = sorted(target.iterdir())
            listing = "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)
            return ToolResult(output=listing or "(empty directory)")

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            selected = lines[input.offset : input.offset + input.limit]
            numbered = [f"{i + input.offset + 1}: {line}" for i, line in enumerate(selected)]
            return ToolResult(output="\n".join(numbered))
        except Exception as e:
            return ToolResult(error=str(e))
