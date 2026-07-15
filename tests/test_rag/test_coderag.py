"""Integration tests for CodeRAG (BM25-only mode)."""

from __future__ import annotations

import pytest

from norn.rag import CodeRAG


SAMPLE_CODE = '''\
def tokenize(text: str) -> list[str]:
    """Split text into tokens."""
    return text.split()


def search_index(query: str, tokens: list[str]) -> list[str]:
    """Search an index using tokenized query."""
    q_tokens = tokenize(query)
    return [t for t in tokens if t in q_tokens]
'''


@pytest.fixture
def rag_dir(tmp_path):
    """Create a tmp dir with one Python file."""
    pytest.importorskip("rank_bm25")
    (tmp_path / "mymod.py").write_text(SAMPLE_CODE)
    return tmp_path


class TestCodeRAGBuild:
    def test_build_returns_chunk_count(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        count = rag.build(str(rag_dir))
        assert count >= 2  # tokenize + search_index + their methods

    def test_chunk_count_property(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        assert rag.chunk_count >= 2

    def test_build_empty_dir(self, tmp_path):
        pytest.importorskip("rank_bm25")
        rag = CodeRAG(vector_enabled=False)
        count = rag.build(str(tmp_path))
        assert count == 0


class TestCodeRAGSearch:
    def test_search_returns_scored_chunks(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        results = rag.search("tokenize text", k=3)
        assert len(results) >= 1

    def test_relevant_match_in_results(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        results = rag.search("tokenize text", k=5)
        names = [r.chunk.name for r in results]
        assert "tokenize" in names


class TestCodeRAGContext:
    def test_context_returns_string(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        ctx = rag.context("search query tokens")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_context_contains_match_label(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        ctx = rag.context("tokenize text")
        assert "MATCH" in ctx or "tokenize" in ctx


class TestCodeRAGSymbolContext:
    def test_symbol_found(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        ctx = rag.symbol_context("tokenize")
        assert "tokenize" in ctx

    def test_symbol_not_found(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        ctx = rag.symbol_context("nonexistent_xyz")
        assert "not found" in ctx

    def test_symbol_includes_callgraph_neighbor(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        # search_index calls tokenize → tokenize should appear in callers section
        ctx = rag.symbol_context("tokenize")
        assert "search_index" in ctx or "CALLER" in ctx


class TestCodeRAGFileSummary:
    def test_file_summary_known_file(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        summary = rag.file_summary("mymod.py")
        assert "tokenize" in summary
        assert "search_index" in summary

    def test_file_summary_unknown_file(self, rag_dir):
        rag = CodeRAG(vector_enabled=False)
        rag.build(str(rag_dir))
        summary = rag.file_summary("nonexistent.py")
        assert "not found" in summary
