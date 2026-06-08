from coderag.index import CodeIndex
from coderag.config import Settings
from coderag.retrieve import Retriever


def test_build_indexes_consistent(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    assert len(idx.chunks) > 0
    # dense and lexical indexes are keyed by the same chunk ids
    assert len(idx.vector_store) == len(idx.chunks)
    assert len(idx.bm25) == len(idx.chunks)
    st = idx.stats()
    assert st["files"] == 2
    assert "function" in st["chunks_by_type"]


def test_retrieval_finds_relevant_file(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    r = Retriever(idx, idx.settings)
    res = r.retrieve("helper function returns 42", k=5, use_rerank=False)
    assert res
    assert any(x.chunk.file_path.endswith("b.py") for x in res)


def test_dense_only_and_bm25_only(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    r = Retriever(idx, idx.settings)
    assert r.retrieve("helper", k=3, use_rerank=False, use_bm25=False)   # dense only
    assert r.retrieve("helper", k=3, use_rerank=False, use_dense=False)  # bm25 only


def test_save_load_roundtrip(sample_repo, embedder, tmp_path):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    d = str(tmp_path / "idx")
    idx.save(d)
    idx2 = CodeIndex.load(d, embedder=embedder)
    assert len(idx2.chunks) == len(idx.chunks)
    assert idx2.graph.stats()["nodes"] == idx.graph.stats()["nodes"]
    assert idx2.graph.stats()["edges"] == idx.graph.stats()["edges"]
    res = Retriever(idx2, idx2.settings, embedder=embedder).retrieve(
        "widget scale", k=3, use_rerank=False)
    assert res


def test_graph_expansion_adds_neighbors(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    r = Retriever(idx, idx.settings)
    base = r.retrieve("top level helper", k=3, use_rerank=False, expand_graph=False)
    expanded = r.retrieve("top level helper", k=3, use_rerank=False, expand_graph=True)
    assert len(expanded) >= len(base)
    # graph_context is a compact, code-free structural map
    ctx = r.graph_context(base)
    assert isinstance(ctx, str)
