"""Token budget assembler for CodeRAG context.

Given ranked chunks (and optional call-graph expansion), assembles a
context string that fits within a token budget. Approximates tokens as
chars / 4 (rough but fast — no tokenizer dependency needed).
"""

from __future__ import annotations

from norn.rag.call_graph import CallGraph
from norn.rag.chunker import CodeChunk
from norn.rag.search import ScoredChunk

_CHARS_PER_TOKEN = 4  # ~GPT average; good enough for budget guarding


def assemble_context(
    ranked: list[ScoredChunk],
    call_graph: CallGraph | None,
    chunk_by_id: dict[str, CodeChunk],
    token_budget: int = 8000,
    expand_hops: int = 1,
) -> str:
    """Assemble retrieved chunks + call-graph neighbors into a context string.

    Priority order:
    1. Direct hits (ranked list)
    2. One-hop call-graph neighbors (callers + callees)
    3. Two-hop (if budget allows)
    """
    budget_chars = token_budget * _CHARS_PER_TOKEN
    used_ids: set[str] = set()
    sections: list[str] = []

    def _append(chunk: CodeChunk, label: str) -> bool:
        nonlocal budget_chars
        text = f"### {label}: `{chunk.display}`\n\n```python\n{chunk.source.rstrip()}\n```\n"
        cost = len(text)
        if budget_chars - cost < 0:
            return False
        budget_chars -= cost
        sections.append(text)
        used_ids.add(chunk.id)
        return True

    # Direct hits
    for sc in ranked:
        if sc.chunk.id not in used_ids:
            _append(sc.chunk, "MATCH")

    # Call-graph expansion
    if call_graph and expand_hops >= 1:
        neighbors: set[str] = set()
        for sc in ranked:
            neighbors.update(call_graph.neighbors(sc.chunk.id))
        for nid in sorted(neighbors):
            if nid not in used_ids:
                chunk = chunk_by_id.get(nid)
                if chunk:
                    _append(chunk, "CALLER/CALLEE")

    if not sections:
        return "(no relevant code found)"

    header = f"## CodeRAG Context ({len(sections)} chunks)\n\n"
    return header + "\n".join(sections)
