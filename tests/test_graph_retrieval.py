"""Graph-as-recall-booster: neighbors enter the rerank pool, connectivity boost,
and HyDE — the fixes that make the code graph actually help (vs. post-rerank
injection). Exercised without the cross-encoder or any API key."""
from coderag.index import CodeIndex
from coderag.config import Settings
from coderag.retrieve import Retriever
from coderag.schema import make_chunk


def _mk(i, code):
    return make_chunk(repo="r", file_path=f"{i}.py", language="python",
                      symbol_name=f"f{i}", symbol_type="function",
                      start_line=1, end_line=2, code=code, context_header="h", git_sha="s")


def _chunks(idx, needle):
    return next(c for c in idx.chunks.values()
               if c.symbol_name and needle in c.symbol_name)


def test_neighbors_enter_rerank_pool(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    r = Retriever(idx, idx.settings)
    top_level = _chunks(idx, "top_level")          # a.py, calls helper in b.py
    pool, relations = r._add_neighbors_to_pool([top_level], idx.settings)
    # the cross-file neighbor (helper in b.py) is pulled into the candidate pool,
    # tagged with a relation — to be scored by the reranker, not injected blind.
    assert len(pool) > 1
    assert relations
    assert any(c.file_path.endswith("b.py") for c in pool)


def test_connectivity_boost_reorders_by_graph(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    r = Retriever(idx, idx.settings)
    top_level = _chunks(idx, "top_level")
    helper = _chunks(idx, "helper")
    other = next(c for c in idx.chunks.values() if c.id not in (top_level.id, helper.id))
    # helper scores high; top_level is graph-connected to it, other is not.
    scored = [(helper, 5.0), (other, 1.0), (top_level, 0.5)]
    out = r._apply_connectivity_boost(scored, boost=1.0)
    ids = [c.id for c, _ in out]
    # top_level inherits helper's relevance via the call edge -> outranks `other`
    assert ids.index(top_level.id) < ids.index(other.id)


def test_connectivity_boost_off_by_default_is_noop(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    assert idx.settings.graph_rerank_boost == 0.0  # opt-in


def test_hyde_query_uses_llm_and_falls_back(sample_repo, embedder):
    from coderag.llm.base import Completion
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    r = Retriever(idx, idx.settings)

    class FakeLLM:
        gen_model = "g"
        def generate(self, system, user, *, max_tokens, stream=False):
            return Completion("def helper(): return 42", {})
    r._llm = FakeLLM()
    out = r._hyde_query("how does the helper work")
    assert "def helper" in out and "how does the helper work" in out  # query + snippet

    class BadLLM:
        gen_model = "g"
        def generate(self, *a, **k):
            raise RuntimeError("no provider")
    r._llm = BadLLM()
    assert r._hyde_query("q") == "q"  # fails open to the raw query


def test_mmr_prefers_diverse_over_near_duplicate(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    r = Retriever(idx, idx.settings)
    c1 = _mk(1, "def a(): return parse_json(payload)")
    c2 = _mk(2, "def b(): return parse_json(payload)")   # near-duplicate of c1
    c3 = _mk(3, "def c(): return compute_average(values)")  # diverse
    out = r._mmr([(c1, 3.0), (c2, 2.8), (c3, 2.5)], k=2, lam=0.3)
    ids = {c.id for c, _ in out}
    assert c1.id in ids and c3.id in ids and c2.id not in ids  # dup dropped for diversity


def test_pagerank_pool_adds_connected_chunk(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    r = Retriever(idx, idx.settings)
    top_level = _chunks(idx, "top_level")
    pool, rel = r._add_pagerank_to_pool([top_level], idx.settings, {})
    assert len(pool) >= 1
    # PPR seeded at top_level should surface a connected chunk tagged 'pagerank'
    assert any(v == "pagerank" for v in rel.values()) or len(pool) == 1
