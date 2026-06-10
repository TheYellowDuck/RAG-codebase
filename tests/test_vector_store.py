"""Vector backends: exact (default) + optional HNSW (ANN), same interface.

The HNSW tests skip automatically when hnswlib isn't installed. We verify the
factory selects the right backend, HNSW closely matches exact's top-k on
clustered data (the realistic case), and save/load + remove behave.
"""
import numpy as np
import pytest

from coderag.config import Settings
from coderag.index.vector_store import VectorStore, make_vector_store, load_vector_store


def _clustered(n=2000, d=64, c=80, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((c, d)).astype("float32")
    X = centers[rng.integers(0, c, n)] + 0.12 * rng.standard_normal((n, d)).astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = [f"c{i}" for i in range(n)]
    return ids, X, centers, rng


def test_factory_defaults_to_exact():
    assert isinstance(make_vector_store(Settings()), VectorStore)
    assert isinstance(make_vector_store(Settings(vector_backend="exact")), VectorStore)


def test_exact_search_is_sorted_and_correct():
    ids, X, *_ = _clustered(n=300, d=32)
    vs = make_vector_store(Settings())
    vs.add(ids, X)
    hits = vs.search(X[7], top_n=5)
    assert hits[0][0] == "c7"                      # a vector is its own nearest neighbor
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)  # best first


# ---------------------------------------------------------------- HNSW ------ #
hnswlib = pytest.importorskip("hnswlib")


def test_factory_selects_hnsw():
    from coderag.index.hnsw_store import HNSWVectorStore
    store = make_vector_store(Settings(vector_backend="hnsw"))
    assert isinstance(store, HNSWVectorStore)


def test_hnsw_matches_exact_topk_on_clustered_data():
    ids, X, _, rng = _clustered(n=3000, d=64, c=120)
    exact = make_vector_store(Settings()); exact.add(ids, X)
    hnsw = make_vector_store(Settings(vector_backend="hnsw", hnsw_ef_search=128)); hnsw.add(ids, X)

    qs = X[rng.integers(0, len(ids), 60)]
    k = 10
    recalls = []
    for q in qs:
        e = {c for c, _ in exact.search(q, k)}
        h = {c for c, _ in hnsw.search(q, k)}
        recalls.append(len(e & h) / k)
    assert np.mean(recalls) >= 0.90      # approximate, but very close on real-ish data


def test_hnsw_remove_excludes_marked():
    ids, X, *_ = _clustered(n=400, d=32)
    hnsw = make_vector_store(Settings(vector_backend="hnsw"))
    hnsw.add(ids, X)
    hnsw.remove({"c1", "c2", "c3"})
    assert len(hnsw) == 397
    got = {c for c, _ in hnsw.search(X[1], top_n=400)}
    assert {"c1", "c2", "c3"}.isdisjoint(got)


def test_hnsw_save_load_roundtrip(tmp_path):
    ids, X, *_ = _clustered(n=500, d=32)
    hnsw = make_vector_store(Settings(vector_backend="hnsw", hnsw_ef_search=100))
    hnsw.add(ids, X)
    before = [c for c, _ in hnsw.search(X[5], top_n=5)]
    hnsw.save(str(tmp_path))

    loaded = load_vector_store(Settings(vector_backend="hnsw"), str(tmp_path))
    assert len(loaded) == 500
    after = [c for c, _ in loaded.search(X[5], top_n=5)]
    assert before == after
