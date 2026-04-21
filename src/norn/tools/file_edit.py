"""File edit tool for targeted string replacement."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


class FileEditInput(BaseModel):
    """Input for file edit operations."""

    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


class FileEditTool:
    """Replace a specific string in a file."""

    name = "file_edit"
    description = "Replace an exact string occurrence in a file with new content."
    risk_level = RiskLevel.MEDIUM
    input_model = FileEditInput

    async def execute(self, input: FileEditInput, ctx: ToolContext) -> ToolResult:
        target = Path(input.path)
        if not target.is_absolute():
            target = Path(ctx.cwd) / target

        if not target.exists():
            return ToolResult(
                error=f"File not found: {target}",
                error_type=ToolErrorType.FILE_NOT_FOUND.value,
            )

        try:
            content = target.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(
                error=f"Cannot read file: {e}",
                error_type=ToolErrorType.EXECUTION_ERROR.value,
            )

        count = content.count(input.old_string)

        if count == 0:
            return ToolResult(
                error=f"Old string not found in {target}",
                error_type=ToolErrorType.INVALID_ARGUMENT.value,
            )

        if count > 1 and not input.replace_all:
            return ToolResult(
                error=f"Found {count} matches. Use replace_all=true or provide more context.",
                error_type=ToolErrorType.INVALID_ARGUMENT.value,
            )

        if input.replace_all:
            new_content = content.replace(input.old_string, input.new_string)
        else:
            new_content = content.replace(input.old_string, input.new_string, 1)

        target.write_text(new_content, encoding="utf-8")
        replaced = count if input.replace_all else 1
        return ToolResult(output=f"Replaced {replaced} occurrence(s) in {target}")
