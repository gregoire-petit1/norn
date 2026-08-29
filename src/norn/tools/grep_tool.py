"""Grep tool for content search."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


class GrepInput(BaseModel):
    """Input for content search."""

    pattern: str
    path: str | None = None
    include: str | None = None


class GrepTool:
    """Search file contents using regex patterns."""

    name = "grep"
    description = "Search file contents using regular expressions."
    risk_level = RiskLevel.LOW
    input_model = GrepInput

    def __init__(self, *, axi_output: bool = False) -> None:
        self._axi = axi_output

    async def execute(self, input: GrepInput, ctx: ToolContext) -> ToolResult:
        base = Path(input.path) if input.path else Path(ctx.cwd)
        if not base.is_absolute():
            base = Path(ctx.cwd) / base

        if not base.exists():
            return ToolResult(
                error=f"Path not found: {base}",
                error_type=ToolErrorType.FILE_NOT_FOUND.value,
            )

        try:
            regex = re.compile(input.pattern)
        except re.error as e:
            return ToolResult(
                error=f"Invalid regex: {e}",
                error_type=ToolErrorType.INVALID_ARGUMENT.value,
            )

        results: list[str] = []
        matched_files: set[str] = set()
        truncated = False
        files = base.rglob(input.include or "*") if base.is_dir() else [base]

        for filepath in files:
            if not filepath.is_file():
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{filepath}:{i}: {line.strip()}")
                        matched_files.add(str(filepath))
                        if len(results) >= 200:
                            truncated = True
                            break
            except Exception:
                continue

            if len(results) >= 200:
                truncated = True
                break

        if self._axi:
            from norn.tools.axi_format import axi_match_result

            return ToolResult(
                output=axi_match_result(
                    results,
                    total_matches=len(results),
                    file_count=len(matched_files),
                    truncated=truncated,
                    full_hint="a narrower pattern, include=, or path=",
                )
            )
        return ToolResult(output="\n".join(results) if results else "No matches found.")
