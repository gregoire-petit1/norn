"""Code chunk extractor — AST-based, no extra dependencies.

Extracts function/class bodies (not just signatures) for embedding + BM25 indexing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_PYTHON_EXTS = {".py"}
_MAX_FILE_SIZE = 1_048_576


@dataclass
class CodeChunk:
    """A named code chunk suitable for retrieval."""

    path: str
    name: str
    kind: str  # "function", "class", "method", "async_function", "async_method"
    line_start: int
    line_end: int
    source: str
    signature: str
    parent: str = ""  # class name for methods

    @property
    def id(self) -> str:
        if not self.parent:
            return f"{self.path}:{self.name}"
        return f"{self.path}:{self.parent}.{self.name}"

    @property
    def display(self) -> str:
        loc = f"{self.path}:{self.line_start}-{self.line_end}"
        parent_prefix = f"{self.parent}." if self.parent else ""
        return f"{parent_prefix}{self.name} ({self.kind}) @ {loc}"


def extract_chunks(path: str, source: str) -> list[CodeChunk]:
    """Extract named code chunks from Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    chunks: list[CodeChunk] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_src = _extract_source(lines, node)
            chunks.append(CodeChunk(
                path=path,
                name=node.name,
                kind="class",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                source=class_src,
                signature=f"class {node.name}",
            ))
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_async = isinstance(child, ast.AsyncFunctionDef)
                    kind = "async_method" if is_async else "method"
                    chunks.append(CodeChunk(
                        path=path,
                        name=child.name,
                        kind=kind,
                        line_start=child.lineno,
                        line_end=child.end_lineno or child.lineno,
                        source=_extract_source(lines, child),
                        signature=_format_sig(child, is_async),
                        parent=node.name,
                    ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_async = isinstance(node, ast.AsyncFunctionDef)
            kind = "async_function" if is_async else "function"
            chunks.append(CodeChunk(
                path=path,
                name=node.name,
                kind=kind,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                source=_extract_source(lines, node),
                signature=_format_sig(node, is_async),
            ))

    return chunks


def extract_chunks_from_file(file_path: Path, root: Path) -> list[CodeChunk]:
    """Read file and extract chunks; returns [] on error."""
    rel = str(file_path.relative_to(root))
    try:
        if file_path.stat().st_size > _MAX_FILE_SIZE:
            return []
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return extract_chunks(rel, source)


def _extract_source(lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "".join(lines[start:end])


def _format_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool) -> str:
    prefix = "async def" if is_async else "def"
    parts = [arg.arg for arg in node.args.args]
    return f"{prefix} {node.name}({', '.join(parts)})"
