"""Tests for AST call graph."""

from __future__ import annotations

from norn.rag.call_graph import build_call_graph
from norn.rag.chunker import extract_chunks


CALLER_CODE = '''\
def helper(x):
    return x + 1

def main_func(y):
    result = helper(y)
    return result
'''


class TestCallGraph:
    def test_callee_detected(self):
        chunks = extract_chunks("m.py", CALLER_CODE)
        graph = build_call_graph(chunks)
        main_id = next(c.id for c in chunks if c.name == "main_func")
        callees = graph.callees_of(main_id)
        assert any("helper" in cid for cid in callees)

    def test_caller_detected(self):
        chunks = extract_chunks("m.py", CALLER_CODE)
        graph = build_call_graph(chunks)
        helper_id = next(c.id for c in chunks if c.name == "helper")
        callers = graph.callers_of(helper_id)
        assert any("main_func" in cid for cid in callers)

    def test_no_self_call(self):
        chunks = extract_chunks("m.py", CALLER_CODE)
        graph = build_call_graph(chunks)
        for chunk in chunks:
            assert chunk.id not in graph.callees_of(chunk.id)

    def test_neighbors_includes_both(self):
        chunks = extract_chunks("m.py", CALLER_CODE)
        graph = build_call_graph(chunks)
        helper_id = next(c.id for c in chunks if c.name == "helper")
        neighbors = graph.neighbors(helper_id)
        assert any("main_func" in n for n in neighbors)

    def test_empty_chunks(self):
        graph = build_call_graph([])
        assert graph.callers_of("foo") == []
        assert graph.callees_of("foo") == []
