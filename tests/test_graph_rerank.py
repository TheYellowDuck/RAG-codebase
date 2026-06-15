"""Graph-aware reranking (_graph_rerank): promote graph-central pool members.

Re-ranks the fused pool by rank-fusing the RRF order with a PPR-connectivity order
(seeded by the top hits). Verified with a stub graph so it's deterministic — no
models, no index build."""
from coderag.retrieve.retriever import Retriever


class _C:
    def __init__(self, i):
        self.id = i


class _Graph:
    def __init__(self, ppr):
        self._ppr = ppr

    def personalized_pagerank(self, seeds, **kw):
        return self._ppr


class _Index:
    def __init__(self, graph):
        self.graph = graph


def _retriever(ppr):
    r = Retriever.__new__(Retriever)        # bypass heavy __init__
    r.index = _Index(_Graph(ppr))
    return r


def test_falls_back_to_rrf_when_no_connectivity():
    cands = [_C(x) for x in "abcde"]         # RRF order a..e
    out = _retriever({})._graph_rerank(cands, k=3, seeds=2)
    assert [c.id for c in out] == ["a", "b", "c"]   # empty PPR -> plain RRF top-k


def test_connectivity_promotes_a_low_ranked_candidate():
    cands = [_C(x) for x in "abcde"]
    # 'e' is last by RRF but strongly connected to the seed -> should move up
    out = [c.id for c in _retriever({"a": 1.0, "e": 0.95})._graph_rerank(cands, k=5, seeds=1)]
    assert set(out) == set("abcde")          # same set, reordered
    assert out[0] == "a"                     # top of both signals stays #1
    assert out.index("e") < 4                # 'e' promoted above its RRF position (last)
    assert out != ["a", "b", "c", "d", "e"]  # connectivity changed the order
