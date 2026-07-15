"""AST-based call graph — no external dependencies.

Builds caller/callee relationships between named symbols in the same codebase.
Only tracks intra-codebase calls (not stdlib/third-party).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field

from norn.rag.chunker import CodeChunk


@dataclass
class CallGraph:
    """Bidirectional call graph."""

    # symbol_id -> set of symbol_ids that this symbol calls
    callees: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # symbol_id -> set of symbol_ids that call this symbol
    callers: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def callers_of(self, symbol_id: str) -> list[str]:
        return sorted(self.callers.get(symbol_id, set()))

    def callees_of(self, symbol_id: str) -> list[str]:
        return sorted(self.callees.get(symbol_id, set()))

    def neighbors(self, symbol_id: str) -> list[str]:
        """Callers + callees (one hop)."""
        result = set(self.callers_of(symbol_id)) | set(self.callees_of(symbol_id))
        return sorted(result)


def build_call_graph(chunks: list[CodeChunk]) -> CallGraph:
    """Build intra-codebase call graph from parsed chunks."""
    graph = CallGraph()

    # Index all known symbol names -> ids
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        name_to_ids[chunk.name].append(chunk.id)

    for chunk in chunks:
        try:
            tree = ast.parse(chunk.source)
        except SyntaxError:
            continue

        called_names = _extract_called_names(tree)
        for called in called_names:
            targets = name_to_ids.get(called, [])
            for target_id in targets:
                if target_id != chunk.id:
                    graph.callees[chunk.id].add(target_id)
                    graph.callers[target_id].add(chunk.id)

    return graph


def _extract_called_names(tree: ast.AST) -> set[str]:
    """Collect all function/method names called within an AST."""
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    return called
