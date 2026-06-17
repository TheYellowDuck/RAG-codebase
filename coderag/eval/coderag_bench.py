"""External-validity benchmark #2: CodeRAG-Bench-style retrieval.

CodeSearchNet ([codesearchnet.py]) validates the *embedder alone* — retrieve a
function from its own docstring. CodeRAG-Bench goes a step further: each query (a
programming problem / NL request) has one or more relevant documents (canonical
solutions, library docs, tutorials) inside a shared corpus. Running our **full
retriever** — dense + BM25 fused by RRF, not just the embedder — over that corpus
gives an external check on the retrieval *pipeline*, closing the gap CSN leaves.

Format-tolerant (like the CSN adapter): accepts a self-contained JSONL where each
record carries a query and its gold document, plus optional ids / extra positive
ids. The corpus is every document in the file; relevance defaults to each query's
own document, or the record's explicit positive ids when given. The official
BEIR-style (queries / corpus / qrels) split scores the same way once flattened to
this shape. The harness logic is unit-tested on a tiny synthetic sample, so it's
verified even without the (large, unbundled) dataset downloaded.

Generation / pass@k (executing model output) is intentionally out of scope — that
needs the official execution harness; this adapter measures retrieval quality, the
part our system owns.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from ..config import Settings
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank

_QUERY_FIELDS = ("query", "question", "nl", "prompt", "intent", "text")
_DOC_FIELDS = ("doc", "document", "canonical_solution", "answer", "code",
               "positive", "positive_passage", "passage", "context")
_ID_FIELDS = ("doc_id", "id", "_id", "docid")
_POS_FIELDS = ("relevant_ids", "positive_ids", "positives", "gold_ids")


def _first(d: dict, names) -> str:
    for n in names:
        v = d.get(n)
        if not v:
            continue
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return " ".join(map(str, v))
        return str(v)
    return ""


def load_coderag_bench(path: str) -> list[dict]:
    """Load (id, query, doc, relevant_ids) records from a CodeRAG-Bench-style JSONL."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rid = next((str(d[k]) for k in _ID_FIELDS if d.get(k) is not None), f"d{i}")
            pos = next((d[k] for k in _POS_FIELDS if d.get(k)), None)
            rec = {
                "id": rid,
                "query": _first(d, _QUERY_FIELDS),
                "doc": _first(d, _DOC_FIELDS),
                "relevant_ids": {str(x) for x in pos} if pos else {rid},
            }
            if rec["query"] and rec["doc"]:
                out.append(rec)
    return out


class _Doc:
    """Duck-typed stand-in for a Chunk so we can reuse the cross-encoder Reranker
    (it scores `candidate.embed_text`) on a generic document corpus."""
    __slots__ = ("id", "embed_text")

    def __init__(self, doc_id: str, text: str):
        self.id = doc_id
        self.embed_text = text


def evaluate_coderag_bench(path: Optional[str] = None, *, embedder=None, k: int = 10,
                           mode: str = "hybrid", limit: Optional[int] = None,
                           records: Optional[list[dict]] = None,
                           reranker=None, rerank_pool: int = 50) -> dict:
    """recall@k / MRR / nDCG@k for retrieving each query's relevant doc(s) from the
    shared corpus.

    mode: 'dense'   = embedder only (cosine), comparable to the CSN protocol;
          'hybrid'  = dense + BM25 fused by RRF — the project's real retriever core;
          'rerank'  = hybrid, then cross-encoder rerank of the top `rerank_pool`.
    `records` lets callers/tests pass pre-loaded data (skips file IO); `embedder`
    and `reranker` default to the configured models (pass stubs in tests)."""
    recs = records if records is not None else load_coderag_bench(path)
    if limit:
        recs = recs[:limit]
    if not recs:
        return {"n": 0}

    ids = [r["id"] for r in recs]
    docs = [r["doc"] for r in recs]
    queries = [r["query"] for r in recs]
    relevant = [r["relevant_ids"] for r in recs]

    # Dense ranking (always computed): cosine over L2-normalized vectors.
    if embedder is None:
        from ..embed import Embedder
        embedder = Embedder.from_settings(Settings.from_env())
    doc_vecs = embedder.encode(docs)
    q_vecs = embedder.encode(queries, is_query=True)
    sims = q_vecs @ doc_vecs.T
    dense_rank = [[ids[j] for j in np.argsort(-sims[i])] for i in range(len(recs))]

    if mode == "dense":
        ranked = dense_rank
    elif mode in ("hybrid", "rerank"):
        # The retriever's core: fuse dense with real BM25 via the same RRF.
        from ..index.bm25_index import BM25Index
        from ..retrieve.retriever import rrf
        bm = BM25Index()
        bm.add(ids, docs)
        fused = []
        for i in range(len(recs)):
            lex = [cid for cid, _ in bm.search(queries[i], top_n=len(ids))]
            fused.append(rrf(dense_rank[i], lex))

        if mode == "hybrid":
            ranked = fused
        else:  # rerank: cross-encoder over the fused top `rerank_pool`
            if reranker is None:
                from ..retrieve.rerank import Reranker
                reranker = Reranker(Settings().rerank_model)
            id_to_doc = dict(zip(ids, docs, strict=True))
            ranked = []
            for i in range(len(recs)):
                pool = fused[i][:rerank_pool]
                cands = [_Doc(cid, id_to_doc[cid]) for cid in pool]
                top = [c.id for c, _ in reranker.rerank(queries[i], cands, top_k=len(cands))]
                tail = [cid for cid in fused[i] if cid not in set(pool)]
                ranked.append(top + tail)   # reranked pool first, rest in fused order
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'dense', 'hybrid', or 'rerank'")

    n = len(recs)
    rec_sum = mrr_sum = ndcg_sum = 0.0
    for i in range(n):
        rec_sum += recall_at_k(ranked[i], relevant[i], k)
        mrr_sum += reciprocal_rank(ranked[i], relevant[i])
        ndcg_sum += ndcg_at_k(ranked[i], relevant[i], k)
    return {"n": n, "mode": mode,
            f"recall@{k}": rec_sum / n, "mrr": mrr_sum / n, f"ndcg@{k}": ndcg_sum / n}
