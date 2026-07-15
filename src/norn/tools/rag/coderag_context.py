"""coderag_context tool — symbol lookup with call-graph expansion."""

from __future__ import annotations

from pydantic import BaseModel, Field

from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.rag.coderag_search import _get_or_build_rag


class CodeRAGContextInput(BaseModel):
    symbol: str = Field(description="Function or class name to look up")


class CodeRAGContextTool:
    name = "coderag_context"
    description = (
        "Look up a specific function or class by name and return its source code "
        "plus its callers and callees (one-hop call graph)."
    )
    risk_level = RiskLevel.LOW
    input_model = CodeRAGContextInput

    async def execute(self, input: CodeRAGContextInput, ctx: ToolContext) -> ToolResult:
        rag = _get_or_build_rag(ctx)
        if rag is None:
            return ToolResult(error="CodeRAG index not available")

        result = rag.symbol_context(input.symbol)
        return ToolResult(output=result)
