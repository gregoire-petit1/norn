"""Hybrid search — BM25 (keyword) + optional vector (semantic) with RRF fusion.

BM25 always available (rank-bm25, pure Python).
Vector search requires: chromadb + Ollama running locally (for embeddings).

Graceful degradation: if ChromaDB/Ollama unavailable → BM25-only mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from norn.rag.chunker import CodeChunk

_log = logging.getLogger(__name__)

_RRF_K = 60  # RRF constant, standard value


@dataclass
class ScoredChunk:
    chunk: CodeChunk
    score: float
    source: str = "bm25"  # "bm25" | "vector" | "rrf"


class HybridSearchIndex:
    """BM25 + optional vector search with RRF fusion.

    Usage:
        idx = HybridSearchIndex(vector_enabled=True)
        idx.index(chunks)
        results = idx.search("parse python function", k=5)
    """

    def __init__(
        self,
        vector_enabled: bool = True,
        embed_model: str = "nomic-embed-text",
        ollama_base: str = "http://localhost:11434",
        collection_name: str = "norn_coderag",
    ) -> None:
        self._vector_enabled = vector_enabled
        self._embed_model = embed_model
        self._ollama_base = ollama_base
        self._collection_name = collection_name

        self._chunks: list[CodeChunk] = []
        self._bm25: Any = None
        self._tokenized: list[list[str]] = []
        self._chroma: Any = None  # chromadb Collection

    def index(self, chunks: list[CodeChunk]) -> None:
        """Index a list of code chunks."""
        self._chunks = chunks
        if not chunks:
            return

        self._build_bm25(chunks)
        if self._vector_enabled:
            self._build_vector(chunks)

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        """Hybrid search: RRF of BM25 + vector (or BM25-only if vector unavailable)."""
        if not self._chunks:
            return []

        bm25_results = self._search_bm25(query, k=k * 2)

        if self._chroma is None:
            return bm25_results[:k]

        vector_results = self._search_vector(query, k=k * 2)
        return self._rrf_fusion(bm25_results, vector_results, k=k)

    # --- BM25 ---

    def _build_bm25(self, chunks: list[CodeChunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]
        except ImportError:
            _log.warning("rank-bm25 not installed; BM25 search disabled")
            return

        self._tokenized = [_tokenize(c.source + " " + c.signature + " " + c.name) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def _search_bm25(self, query: str, k: int) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(s), source="bm25")
            for i, s in ranked[:k]
        ]

    # --- Vector (ChromaDB + Ollama) ---

    def _build_vector(self, chunks: list[CodeChunk]) -> None:
        try:
            import chromadb  # type: ignore[import]
        except ImportError:
            _log.info("chromadb not installed; vector search disabled")
            return

        embeddings = self._embed_batch([c.source[:2000] for c in chunks])
        if not embeddings:
            return

        try:
            client = chromadb.Client()
            try:
                client.delete_collection(self._collection_name)
            except Exception:
                pass
            coll = client.create_collection(self._collection_name)
            coll.add(
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=[c.source[:2000] for c in chunks],
                metadatas=[{"path": c.path, "name": c.name, "kind": c.kind} for c in chunks],
            )
            self._chroma = coll
        except Exception as exc:
            _log.warning("chromadb index failed: %s", exc)

    def _search_vector(self, query: str, k: int) -> list[ScoredChunk]:
        if self._chroma is None:
            return []
        q_embed = self._embed_batch([query])
        if not q_embed:
            return []
        try:
            res = self._chroma.query(query_embeddings=q_embed, n_results=min(k, len(self._chunks)))
        except Exception as exc:
            _log.warning("vector search failed: %s", exc)
            return []

        chunk_by_id = {c.id: c for c in self._chunks}
        results = []
        for ids, dists in zip(res["ids"], res["distances"]):
            for cid, dist in zip(ids, dists):
                chunk = chunk_by_id.get(cid)
                if chunk:
                    score = 1.0 / (1.0 + dist)
                    results.append(ScoredChunk(chunk=chunk, score=score, source="vector"))
        return results

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama HTTP API."""
        try:
            import httpx
        except ImportError:
            _log.warning("httpx not installed; vector embedding disabled")
            return []
        try:
            resp = httpx.post(
                f"{self._ollama_base}/api/embed",
                json={"model": self._embed_model, "input": texts},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("embeddings", [])
        except Exception as exc:
            _log.debug("Ollama embedding failed: %s", exc)
            return []

    # --- RRF fusion ---

    @staticmethod
    def _rrf_fusion(
        bm25: list[ScoredChunk],
        vector: list[ScoredChunk],
        k: int,
    ) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, CodeChunk] = {}

        for rank, sc in enumerate(bm25, 1):
            cid = sc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            chunks[cid] = sc.chunk

        for rank, sc in enumerate(vector, 1):
            cid = sc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            chunks[cid] = sc.chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(chunk=chunks[cid], score=s, source="rrf") for cid, s in ranked[:k]]


def _tokenize(text: str) -> list[str]:
    import re
    # Split on non-alphanumeric, also split CamelCase and snake_case
    tokens = re.findall(r"[a-z]+|[A-Z][a-z]*|[0-9]+", text)
    return [t.lower() for t in tokens if len(t) > 1]
