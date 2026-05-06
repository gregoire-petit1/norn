"""Repo map — AST-based codebase context for system prompt injection (W3.1).

Generates a condensed map of the codebase showing file paths with
function/class signatures and line numbers. Reduces token waste from
full file reads by 70-90%.

Zero external dependencies: uses stdlib `ast` for Python, regex for TypeScript.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Language extension mapping
_PYTHON_EXTENSIONS = {".py"}
_TYPESCRIPT_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}

# Default exclude patterns (glob)
_DEFAULT_EXCLUDES = [
    ".venv/**",
    "venv/**",
    "node_modules/**",
    "__pycache__/**",
    "*.egg-info/**",
    "dist/**",
    "build/**",
    "*_pb2.py",
    "*_pb2_grpc.py",
    "migrations/**",
    ".git/**",
    ".tox/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
]

# Max file size to parse (1MB)
_MAX_FILE_SIZE = 1_048_576

# TypeScript/JS extraction patterns
_TS_PATTERNS = [
    re.compile(r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", re.MULTILINE),
    re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+)?\s*=", re.MULTILINE),
]

# Signature max length for rendering
_MAX_SIGNATURE_LEN = 50


@dataclass
class Symbol:
    """A code symbol (class, function, constant)."""

    name: str
    kind: Literal["class", "function", "async_function", "constant", "method", "async_method"]
    line: int
    signature: str = ""
    children: list[Symbol] = field(default_factory=list)


@dataclass
class FileSymbols:
    """Symbols extracted from a single file."""

    path: str  # relative to cwd
    symbols: list[Symbol]
    mtime: float


class RepoMap:
    """Generate a condensed map of a codebase for LLM context.

    Uses ast.parse() for Python and regex for TypeScript/JS.
    Caches results by file mtime within a session.
    """

    def __init__(
        self,
        max_chars: int = 2000,
        languages: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.languages = languages or ["python", "typescript"]
        self.exclude_patterns = exclude_patterns or _DEFAULT_EXCLUDES
        self._cache: dict[str, FileSymbols] = {}

    def scan(self, cwd: str) -> str:
        """Scan cwd and return rendered repo map string."""
        files = self._discover_files(cwd)
        file_symbols: list[FileSymbols] = []

        for file_path in files:
            fs = self._parse_file(file_path, cwd)
            if fs and fs.symbols:
                file_symbols.append(fs)

        if not file_symbols:
            return "## Repo Map\n\n(no symbols found)"

        return self._render(file_symbols)

    def _discover_files(self, cwd: str) -> list[Path]:
        """Discover source files, respecting exclude patterns."""
        root = Path(cwd)
        extensions: set[str] = set()
        if "python" in self.languages:
            extensions.update(_PYTHON_EXTENSIONS)
        if "typescript" in self.languages:
            extensions.update(_TYPESCRIPT_EXTENSIONS)

        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)

            # Check exclude patterns on directory
            skip_dir = False
            for pattern in self.exclude_patterns:
                if fnmatch.fnmatch(rel_dir + "/", pattern) or fnmatch.fnmatch(rel_dir, pattern):
                    skip_dir = True
                    break
                # Also check with trailing slash for subdirs
                if fnmatch.fnmatch(rel_dir + "/x", pattern):
                    skip_dir = True
                    break
            if skip_dir:
                dirnames.clear()
                continue

            # Filter hidden dirs
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            for filename in filenames:
                ext = Path(filename).suffix.lower()
                if ext not in extensions:
                    continue

                file_path = Path(dirpath) / filename
                rel_path = os.path.relpath(file_path, root)

                # Check exclude patterns on file
                excluded = False
                for pattern in self.exclude_patterns:
                    if fnmatch.fnmatch(rel_path, pattern):
                        excluded = True
                        break
                if excluded:
                    continue

                # Skip files > 1MB
                try:
                    if file_path.stat().st_size > _MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue

                files.append(file_path)

        # Sort: shallower files first, then alphabetical
        files.sort(key=lambda p: (len(p.relative_to(root).parts), str(p)))
        return files

    def _parse_file(self, path: Path, cwd: str) -> FileSymbols | None:
        """Parse a file, using cache if mtime unchanged."""
        rel_path = os.path.relpath(path, cwd)

        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None

        # Check cache
        if rel_path in self._cache and self._cache[rel_path].mtime == mtime:
            return self._cache[rel_path]

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        ext = path.suffix.lower()
        if ext in _PYTHON_EXTENSIONS:
            symbols = self._parse_python(path, source)
        elif ext in _TYPESCRIPT_EXTENSIONS:
            symbols = self._parse_typescript(path, source)
        else:
            return None

        fs = FileSymbols(path=rel_path, symbols=symbols, mtime=mtime)
        self._cache[rel_path] = fs
        return fs

    def _parse_python(self, path: Path, source: str) -> list[Symbol]:
        """Extract symbols from Python source using ast."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        symbols: list[Symbol] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                children = self._extract_class_methods(node)
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="class",
                        line=node.lineno,
                        signature=f"class {node.name}",
                        children=children,
                    )
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_async = isinstance(node, ast.AsyncFunctionDef)
                sig = self._format_function_signature(node, is_async)
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="async_function" if is_async else "function",
                        line=node.lineno,
                        signature=sig,
                    )
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id.isupper()
                        and not target.id.startswith("_")
                    ):
                        symbols.append(
                            Symbol(
                                name=target.id,
                                kind="constant",
                                line=node.lineno,
                                signature=f"{target.id} = ...",
                            )
                        )

        return symbols

    def _extract_class_methods(self, class_node: ast.ClassDef) -> list[Symbol]:
        """Extract methods from a class definition."""
        methods: list[Symbol] = []
        for node in ast.iter_child_nodes(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_async = isinstance(node, ast.AsyncFunctionDef)
                sig = self._format_function_signature(node, is_async)
                methods.append(
                    Symbol(
                        name=node.name,
                        kind="async_method" if is_async else "method",
                        line=node.lineno,
                        signature=sig,
                    )
                )
        return methods

    def _format_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> str:
        """Format a function/method signature from AST node."""
        prefix = "async def" if is_async else "def"
        args_parts: list[str] = []

        for arg in node.args.args:
            annotation = ""
            if arg.annotation:
                annotation = f": {ast.unparse(arg.annotation)}"
            args_parts.append(f"{arg.arg}{annotation}")

        args_str = ", ".join(args_parts)
        if len(args_str) > _MAX_SIGNATURE_LEN:
            args_str = args_str[:_MAX_SIGNATURE_LEN] + "..."

        return f"{prefix} {node.name}({args_str})"

    def _parse_typescript(self, path: Path, source: str) -> list[Symbol]:
        """Extract symbols from TypeScript/JS source using regex."""
        symbols: list[Symbol] = []
        lines = source.splitlines()

        for line_num, line in enumerate(lines, 1):
            for pattern in _TS_PATTERNS:
                match = pattern.match(line)
                if match:
                    name = match.group(1)
                    if "class " in line:
                        kind: Literal[
                            "class",
                            "function",
                            "async_function",
                            "constant",
                            "method",
                            "async_method",
                        ] = "class"
                    elif "function " in line:
                        kind = "async_function" if "async " in line else "function"
                    else:
                        kind = "constant"
                    symbols.append(
                        Symbol(
                            name=name,
                            kind=kind,
                            line=line_num,
                            signature=line.strip()[:60],
                        )
                    )
                    break

        return symbols

    def _render(self, files: list[FileSymbols], max_chars: int | None = None) -> str:
        """Render file symbols into a compact text map."""
        budget = max_chars or self.max_chars
        lines: list[str] = ["## Repo Map", ""]

        for fs in files:
            file_line = f"{fs.path} ({len(fs.symbols)} symbols)"
            entry_lines: list[str] = [file_line]

            for sym in fs.symbols:
                entry_lines.append(f"  {sym.signature:<50} L{sym.line}")
                for child in sym.children:
                    entry_lines.append(f"    {child.signature:<48} L{child.line}")

            entry_text = "\n".join(entry_lines)
            current = "\n".join(lines)
            if len(current) + len(entry_text) + 1 > budget:
                lines.append("  ...(truncated)")
                break
            lines.append(entry_text)

        return "\n".join(lines)
