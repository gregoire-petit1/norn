"""CodeRAG — embedding-augmented code retrieval for Norn.

Extends the AST repo_map with BM25 + optional vector search,
call-graph expansion, and token-budget-aware context assembly.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from norn.rag.budget import assemble_context
from norn.rag.call_graph import CallGraph, build_call_graph
from norn.rag.chunker import CodeChunk, extract_chunks_from_file
from norn.rag.search import HybridSearchIndex, ScoredChunk

_DEFAULT_EXCLUDES = [
    ".venv/**", "venv/**", "node_modules/**", "__pycache__/**",
    "*.egg-info/**", "dist/**", "build/**", ".git/**",
    ".tox/**", ".mypy_cache/**", ".pytest_cache/**", ".ruff_cache/**",
]


class CodeRAG:
    """Main CodeRAG interface: index a directory, search, return context."""

    def __init__(
        self,
        vector_enabled: bool = True,
        embed_model: str = "nomic-embed-text",
        ollama_base: str = "http://localhost:11434",
        token_budget: int = 8000,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self._token_budget = token_budget
        self._excludes = exclude_patterns or _DEFAULT_EXCLUDES
        self._index = HybridSearchIndex(
            vector_enabled=vector_enabled,
            embed_model=embed_model,
            ollama_base=ollama_base,
        )
        self._chunks: list[CodeChunk] = []
        self._chunk_by_id: dict[str, CodeChunk] = {}
        self._call_graph: CallGraph | None = None
        self._indexed_root: str | None = None

    def build(self, cwd: str) -> int:
        """Index all Python files under cwd. Returns chunk count."""
        root = Path(cwd)
        files = _discover_python_files(root, self._excludes)
        chunks: list[CodeChunk] = []
        for f in files:
            chunks.extend(extract_chunks_from_file(f, root))

        self._chunks = chunks
        self._chunk_by_id = {c.id: c for c in chunks}
        self._call_graph = build_call_graph(chunks)
        self._index.index(chunks)
        self._indexed_root = cwd
        return len(chunks)

    def search(self, query: str, k: int = 8) -> list[ScoredChunk]:
        """Return top-k scored chunks for the query."""
        return self._index.search(query, k=k)

    def context(self, query: str, k: int = 8, expand_hops: int = 1) -> str:
        """Search + call-graph expand + assemble into token-budgeted context string."""
        ranked = self.search(query, k=k)
        return assemble_context(
            ranked,
            self._call_graph,
            self._chunk_by_id,
            token_budget=self._token_budget,
            expand_hops=expand_hops,
        )

    def symbol_context(self, symbol_name: str) -> str:
        """Return source + one-hop neighbors for a named symbol."""
        matches = [c for c in self._chunks if c.name == symbol_name or c.id.endswith(f":{symbol_name}")]
        if not matches:
            return f"(symbol `{symbol_name}` not found in index)"

        scored = [ScoredChunk(chunk=m, score=1.0, source="exact") for m in matches]
        return assemble_context(
            scored,
            self._call_graph,
            self._chunk_by_id,
            token_budget=self._token_budget,
            expand_hops=1,
        )

    def file_summary(self, rel_path: str) -> str:
        """Return all chunks for a file as a summary."""
        chunks = [c for c in self._chunks if c.path == rel_path or c.path.endswith(rel_path)]
        if not chunks:
            return f"(file `{rel_path}` not found in index)"

        lines = [f"## File: `{rel_path}` ({len(chunks)} symbols)\n"]
        for c in chunks:
            parent = f"{c.parent}." if c.parent else ""
            lines.append(f"- `{parent}{c.name}` ({c.kind}) L{c.line_start}-{c.line_end}: {c.signature}")
        return "\n".join(lines)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


def _discover_python_files(root: Path, excludes: list[str]) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if any(fnmatch.fnmatch(rel_dir + "/", p) or fnmatch.fnmatch(rel_dir, p) for p in excludes):
            dirnames.clear()
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = Path(dirpath) / fname
            rel = os.path.relpath(fpath, root)
            if any(fnmatch.fnmatch(rel, p) for p in excludes):
                continue
            files.append(fpath)
    files.sort(key=lambda p: (len(p.relative_to(root).parts), str(p)))
    return files