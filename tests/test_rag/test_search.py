"""Tests for hybrid search index (BM25-only; vector requires Ollama)."""

from __future__ import annotations

import pytest

from norn.rag.chunker import CodeChunk
from norn.rag.search import HybridSearchIndex, _tokenize


def _make_chunk(name: str, source: str, path: str = "f.py") -> CodeChunk:
    return CodeChunk(
        path=path,
        name=name,
        kind="function",
        line_start=1,
        line_end=5,
        source=source,
        signature=f"def {name}()",
    )


class TestTokenize:
    def test_splits_on_non_alpha(self):
        tokens = _tokenize("hello_world foo")
        assert "hello" in tokens
        assert "world" in tokens

    def test_camel_case_split(self):
        tokens = _tokenize("CamelCaseName")
        assert "camel" in tokens
        assert "case" in tokens

    def test_filters_single_chars(self):
        tokens = _tokenize("a b c")
        assert tokens == []


class TestBM25Search:
    def setup_method(self):
        pytest.importorskip("rank_bm25")
        self.idx = HybridSearchIndex(vector_enabled=False)
        self.chunks = [
            _make_chunk("parse_python", "def parse_python(source): parse ast tree python"),
            _make_chunk("build_index", "def build_index(): build chromadb index"),
            _make_chunk("rrf_fusion", "def rrf_fusion(): reciprocal rank fusion combine scores"),
        ]
        self.idx.index(self.chunks)

    def test_search_returns_results(self):
        results = self.idx.search("parse python ast", k=3)
        assert len(results) >= 1

    def test_most_relevant_first(self):
        results = self.idx.search("parse python ast", k=3)
        names = [r.chunk.name for r in results]
        assert names[0] == "parse_python"

    def test_search_empty_index(self):
        idx = HybridSearchIndex(vector_enabled=False)
        idx.index([])
        results = idx.search("anything", k=5)
        assert results == []

    def test_k_limits_results(self):
        results = self.idx.search("build index", k=1)
        assert len(results) == 1

    def test_score_source_is_bm25(self):
        results = self.idx.search("parse python", k=2)
        assert all(r.source == "bm25" for r in results)


class TestRRFFusion:
    def test_rrf_combines_lists(self):
        from norn.rag.search import HybridSearchIndex, ScoredChunk

        c1 = _make_chunk("alpha", "alpha function")
        c2 = _make_chunk("beta", "beta function")

        bm25 = [ScoredChunk(chunk=c1, score=1.0, source="bm25"),
                ScoredChunk(chunk=c2, score=0.5, source="bm25")]
        vector = [ScoredChunk(chunk=c2, score=1.0, source="vector"),
                  ScoredChunk(chunk=c1, score=0.8, source="vector")]

        fused = HybridSearchIndex._rrf_fusion(bm25, vector, k=2)
        assert len(fused) == 2
        assert all(r.source == "rrf" for r in fused)
