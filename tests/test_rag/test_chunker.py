"""Tests for code chunker."""

from __future__ import annotations

import pytest

from norn.rag.chunker import CodeChunk, extract_chunks


SAMPLE_PYTHON = '''\
class MyClass:
    def __init__(self, x: int) -> None:
        self.x = x

    def compute(self) -> int:
        return self.x * 2


def standalone(a, b):
    return a + b


async def async_fn(ctx):
    pass
'''


class TestExtractChunks:
    def test_extracts_class(self):
        chunks = extract_chunks("foo.py", SAMPLE_PYTHON)
        names = [c.name for c in chunks]
        assert "MyClass" in names

    def test_extracts_methods(self):
        chunks = extract_chunks("foo.py", SAMPLE_PYTHON)
        names = [c.name for c in chunks]
        assert "__init__" in names
        assert "compute" in names

    def test_extracts_standalone_function(self):
        chunks = extract_chunks("foo.py", SAMPLE_PYTHON)
        names = [c.name for c in chunks]
        assert "standalone" in names

    def test_extracts_async_function(self):
        chunks = extract_chunks("foo.py", SAMPLE_PYTHON)
        kinds = {c.name: c.kind for c in chunks}
        assert kinds["async_fn"] == "async_function"

    def test_method_has_parent(self):
        chunks = extract_chunks("foo.py", SAMPLE_PYTHON)
        method = next(c for c in chunks if c.name == "compute")
        assert method.parent == "MyClass"

    def test_method_id_includes_class(self):
        chunks = extract_chunks("foo.py", SAMPLE_PYTHON)
        method = next(c for c in chunks if c.name == "compute")
        assert "MyClass.compute" in method.id

    def test_chunk_source_nonempty(self):
        chunks = extract_chunks("foo.py", SAMPLE_PYTHON)
        for c in chunks:
            assert c.source.strip(), f"{c.name} has empty source"

    def test_chunk_line_numbers(self):
        chunks = extract_chunks("foo.py", SAMPLE_PYTHON)
        standalone = next(c for c in chunks if c.name == "standalone")
        assert standalone.line_start >= 1
        assert standalone.line_end >= standalone.line_start

    def test_syntax_error_returns_empty(self):
        chunks = extract_chunks("bad.py", "def broken(:")
        assert chunks == []

    def test_path_stored_on_chunk(self):
        chunks = extract_chunks("src/module.py", SAMPLE_PYTHON)
        for c in chunks:
            assert c.path == "src/module.py"
