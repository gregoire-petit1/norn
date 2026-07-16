"""coderag_search tool — semantic + keyword search over the indexed codebase."""

from __future__ import annotations

from pydantic import BaseModel, Field

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class CodeRAGSearchInput(BaseModel):
    query: str = Field(description="Natural language or keyword query to search the codebase")
    k: int = Field(default=6, description="Number of top results to return (1-20)", ge=1, le=20)


class CodeRAGSearchTool:
    name = "coderag_search"
    description = (
        "Search the codebase for relevant functions/classes using hybrid BM25 + semantic retrieval. "
        "Returns source code of the most relevant symbols with file location."
    )
    risk_level = RiskLevel.LOW
    input_model = CodeRAGSearchInput

    async def execute(self, input: CodeRAGSearchInput, ctx: ToolContext) -> ToolResult:
        rag = _get_or_build_rag(ctx)
        if rag is None:
            return ToolResult(error="CodeRAG index not available (rank-bm25 not installed?)")

        result = rag.context(input.query, k=input.k)
        return ToolResult(output=result)


def _get_or_build_rag(ctx: ToolContext):
    """Lazy-init CodeRAG on the ToolContext cwd."""
    try:
        from norn.rag import CodeRAG
    except ImportError:
        return None

    # Cache on ctx to avoid rebuilding each call
    if not hasattr(ctx, "_coderag"):
        rag = CodeRAG(vector_enabled=True)
        rag.build(ctx.cwd)
        ctx._coderag = rag  # type: ignore[attr-defined]

    return ctx._coderag  # type: ignore[attr-defined]
