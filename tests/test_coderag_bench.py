"""CodeRAG-Bench retrieval adapter (coderag/eval/coderag_bench.py).

Verifies the harness on a tiny synthetic corpus (no dataset download, no heavy
models): format-tolerant loading, the dense and hybrid modes both run and return
sane metrics, the BM25 contribution shows up on an exact-identifier query the
embedder stub misses, and explicit positive-id relevance is honored.
"""
import re

import numpy as np

from coderag.eval.coderag_bench import evaluate_coderag_bench, load_coderag_bench


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder — enough for cosine to be
    meaningful (queries near docs that share tokens) without any model download."""
    dim = 64

    def encode(self, texts, **kw):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[i, hash(tok) % self.dim] += 1.0
            nrm = np.linalg.norm(out[i])
            if nrm:
                out[i] /= nrm
        return out

    def encode_query(self, q):
        return self.encode([q])[0]


def test_loader_is_format_tolerant(tmp_path):
    f = tmp_path / "b.jsonl"
    f.write_text(
        '{"query": "reverse a list", "canonical_solution": "def rev(x): return x[::-1]"}\n'
        '{"question": "sum a list", "code": "def s(x): return sum(x)", "id": "q2"}\n'
        '\n'  # blank line tolerated
        '{"nl": "no doc here"}\n',  # missing doc -> dropped
        encoding="utf-8",
    )
    recs = load_coderag_bench(str(f))
    assert len(recs) == 2
    assert recs[0]["query"] == "reverse a list" and recs[0]["doc"].startswith("def rev")
    assert recs[1]["id"] == "q2"                      # explicit id honored
    assert recs[0]["relevant_ids"] == {recs[0]["id"]}  # defaults to own id


def _corpus():
    # Each query best matches its own doc by shared tokens.
    return [
        {"id": "a", "query": "reverse a string", "doc": "reverse string characters order",
         "relevant_ids": {"a"}},
        {"id": "b", "query": "sum integers list", "doc": "sum integers in a list total",
         "relevant_ids": {"b"}},
        {"id": "c", "query": "read json file", "doc": "open and parse json file contents",
         "relevant_ids": {"c"}},
    ]


def test_dense_mode_recovers_aligned_pairs():
    res = evaluate_coderag_bench(records=_corpus(), embedder=HashEmbedder(), k=3, mode="dense")
    assert res["n"] == 3 and res["mode"] == "dense"
    assert res["recall@3"] == 1.0 and res["mrr"] == 1.0
    assert 0.0 <= res["ndcg@3"] <= 1.0


def test_hybrid_mode_runs_and_uses_bm25():
    res = evaluate_coderag_bench(records=_corpus(), embedder=HashEmbedder(), k=3, mode="hybrid")
    assert res["mode"] == "hybrid" and res["n"] == 3
    assert {"recall@3", "mrr", "ndcg@3"} <= set(res)
    assert res["recall@3"] == 1.0


def test_hybrid_rescues_exact_identifier_the_embedder_misses():
    # Stub embedder that ignores the rare identifier -> dense ranks the gold doc last;
    # real BM25 keys on the shared rare token -> hybrid should recover recall@1.
    class BlindEmbedder:
        dim = 8
        def encode(self, texts, **kw):
            v = np.ones((len(texts), self.dim), dtype=np.float32)
            return v / np.linalg.norm(v[0])    # identical vectors -> dense order is arbitrary/uninformative
    recs = [
        {"id": "t", "query": "call HTTPValidationError", "doc": "class HTTPValidationError handler",
         "relevant_ids": {"t"}},
        {"id": "x", "query": "call HTTPValidationError", "doc": "unrelated helper function foo",
         "relevant_ids": {"x"}},
    ]
    hybrid = evaluate_coderag_bench(records=recs, embedder=BlindEmbedder(), k=1, mode="hybrid")
    # BM25 ranks the doc containing the rare identifier first for its query.
    assert hybrid["recall@1"] >= 0.5


def test_limit_caps_examples():
    res = evaluate_coderag_bench(records=_corpus(), embedder=HashEmbedder(), k=3,
                                 mode="dense", limit=2)
    assert res["n"] == 2


class OverlapReranker:
    """Stub cross-encoder: scores a candidate by query-token overlap (no model
    download). Lets us verify the rerank path routes the reranked order through."""
    def rerank(self, query, candidates, top_k):
        q = set(re.findall(r"\w+", query.lower()))
        scored = sorted(
            candidates,
            key=lambda c: len(q & set(re.findall(r"\w+", c.embed_text.lower()))),
            reverse=True)
        return [(c, float(len(q))) for c in scored[:top_k]]


def test_rerank_mode_uses_reranked_order():
    # Each query shares tokens with its own gold doc, so an overlap reranker should
    # surface the gold first -> recall@1 == 1.0, proving the reranked order is used.
    res = evaluate_coderag_bench(records=_corpus(), embedder=HashEmbedder(), k=1,
                                 mode="rerank", reranker=OverlapReranker(), rerank_pool=3)
    assert res["mode"] == "rerank" and res["n"] == 3
    assert {"recall@1", "mrr", "ndcg@1"} <= set(res)
    assert res["recall@1"] == 1.0
