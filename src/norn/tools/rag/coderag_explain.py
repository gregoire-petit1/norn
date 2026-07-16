"""coderag_explain tool — file-level symbol summary."""

from __future__ import annotations

from pydantic import BaseModel, Field

from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.rag.coderag_search import _get_or_build_rag


class CodeRAGExplainInput(BaseModel):
    file_path: str = Field(description="Relative file path to explain (e.g. src/norn/core/agent.py)")


class CodeRAGExplainTool:
    name = "coderag_explain"
    description = (
        "List all functions and classes defined in a file with their signatures and line numbers. "
        "Useful for understanding a file's structure before reading it in full."
    )
    risk_level = RiskLevel.LOW
    input_model = CodeRAGExplainInput

    async def execute(self, input: CodeRAGExplainInput, ctx: ToolContext) -> ToolResult:
        rag = _get_or_build_rag(ctx)
        if rag is None:
            return ToolResult(error="CodeRAG index not available")

        result = rag.file_summary(input.file_path)
        return ToolResult(output=result)
