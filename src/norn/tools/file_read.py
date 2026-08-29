"""File read tool for the Norn agent."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


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

    def __init__(self, *, axi_output: bool = False) -> None:
        self._axi = axi_output

    async def execute(self, input: FileReadInput, ctx: ToolContext) -> ToolResult:
        target = Path(input.path)
        if not target.is_absolute():
            target = Path(ctx.cwd) / target

        if not target.exists():
            return ToolResult(
                error=f"Path not found: {target}",
                error_type=ToolErrorType.FILE_NOT_FOUND.value,
            )

        if target.is_dir():
            entries = sorted(target.iterdir())
            listing = "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)
            if self._axi:
                from norn.tools.axi_format import axi_dir_header

                dirs = sum(1 for e in entries if e.is_dir())
                header = axi_dir_header(dirs=dirs, files=len(entries) - dirs)
                return ToolResult(output=header + ("\n" + listing if listing else ""))
            return ToolResult(output=listing or "(empty directory)")

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            selected = lines[input.offset : input.offset + input.limit]
            numbered = [f"{i + input.offset + 1}: {line}" for i, line in enumerate(selected)]
            body = "\n".join(numbered)
            if self._axi:
                from norn.tools.axi_format import axi_file_header

                start = input.offset + 1 if selected else 0
                end = input.offset + len(selected)
                header = axi_file_header(
                    str(target), start=start, end=end, total_lines=len(lines)
                )
                return ToolResult(output=header + ("\n" + body if body else ""))
            return ToolResult(output=body)
        except Exception as e:
            return ToolResult(error=str(e), error_type=ToolErrorType.EXECUTION_ERROR.value)
