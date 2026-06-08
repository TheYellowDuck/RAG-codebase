"""Retrieval pipeline (outline §3): dense + lexical + RRF + rerank + graph.

  query
    ├─ dense (semantic) search   → ranked chunk_ids        (§3.1)
    ├─ lexical (BM25) search     → ranked chunk_ids        (§3.2)
    ├─ Reciprocal Rank Fusion    → fused candidates        (§3.3)
    ├─ cross-encoder rerank      → final top-k             (§3.4)
    └─ code-graph expansion      → + precise neighbors      (token-saver)

Every flag here is also a settings field, so the eval harness can ablate
(dense-only, +bm25, +rerank, +graph) on a single index.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..schema import Chunk
from ..index import CodeIndex
from ..embed import Embedder
from .rerank import Reranker


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    source: str = "retrieved"      # "retrieved" | "graph"
    relation: Optional[str] = None  # for graph neighbors: calls/called_by/...

    @property
    def citation(self) -> str:
        return self.chunk.citation


def rrf(*ranked_lists: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion (§3.3). Each input is an ordered list of chunk_ids,
    best first. Fuse by rank — cosine and BM25 scores aren't comparable, so don't
    normalize them; RRF is parameter-light and beats either retriever alone."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, index: CodeIndex, settings: Settings,
                 embedder: Optional[Embedder] = None,
                 reranker: Optional[Reranker] = None):
        self.index = index
        self.settings = settings
        # Reuse the index's embedder so we don't load the model twice.
        self.embedder = embedder or index.embedder
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.settings.rerank_model)
        return self._reranker

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, *, k: Optional[int] = None,
                 use_dense: Optional[bool] = None, use_bm25: Optional[bool] = None,
                 use_rerank: Optional[bool] = None, expand_graph: Optional[bool] = None
                 ) -> list[RetrievedChunk]:
        s = self.settings
        k = k or s.final_k
        use_dense = s.use_dense if use_dense is None else use_dense
        use_bm25 = s.use_bm25 if use_bm25 is None else use_bm25
        use_rerank = s.use_rerank if use_rerank is None else use_rerank
        expand_graph = s.expand_graph if expand_graph is None else expand_graph

        if not (use_dense or use_bm25):
            raise ValueError("Enable at least one of dense / bm25 retrieval.")

        ranked_lists: list[list[str]] = []
        if use_dense:
            qvec = self.embedder.encode_query(query)
            dense = self.index.vector_store.search(qvec, s.dense_top_n)
            ranked_lists.append([cid for cid, _ in dense])
        if use_bm25:
            lexical = self.index.bm25.search(query, s.bm25_top_n)
            ranked_lists.append([cid for cid, _ in lexical])

        # Single retriever → its own order; multiple → fuse by rank.
        fused = rrf(*ranked_lists, k=s.rrf_k) if len(ranked_lists) > 1 else ranked_lists[0]
        candidate_ids = fused[: s.fuse_top_n]
        candidates = [self.index.get_chunk(cid) for cid in candidate_ids]
        candidates = [c for c in candidates if c is not None]

        if use_rerank and candidates:
            scored = self.reranker.rerank(query, candidates, k)
            results = [RetrievedChunk(c, sc, i + 1) for i, (c, sc) in enumerate(scored)]
        else:
            # Fall back to fused rank; synthesize a descending score.
            top = candidates[:k]
            results = [RetrievedChunk(c, 1.0 / (i + 1), i + 1) for i, c in enumerate(top)]

        if expand_graph:
            results = self._expand_with_graph(results, s.graph_expand_budget)
        return results

    # ------------------------------------------------------------------ #
    def _expand_with_graph(self, results: list[RetrievedChunk],
                           budget: int) -> list[RetrievedChunk]:
        """Add a few code-graph neighbors of the top results — callees, callers,
        the enclosing class, imports. This hands the model connected context
        without it (or us) scanning whole files: the graph already knows what's
        relevant, so we spend tokens only on the chunks that matter."""
        if budget <= 0 or not self.index.graph.nodes:
            return results
        already = {r.chunk.id for r in results}
        added: list[RetrievedChunk] = []
        next_rank = len(results) + 1
        for r in results:
            if len(added) >= budget:
                break
            for nb in self.index.graph.neighbors(r.chunk.id, depth=1):
                if len(added) >= budget:
                    break
                if nb.chunk_id in already:
                    continue
                chunk = self.index.get_chunk(nb.chunk_id)
                if chunk is None:
                    continue
                already.add(nb.chunk_id)
                added.append(RetrievedChunk(
                    chunk=chunk, score=0.0, rank=next_rank,
                    source="graph", relation=nb.relation,
                ))
                next_rank += 1
        return results + added

    # ------------------------------------------------------------------ #
    def graph_context(self, results: list[RetrievedChunk], max_lines: int = 20) -> str:
        """A compact structural map of how the retrieved chunks connect — names +
        locations + relation, no code. Cheap tokens that let the model reason
        about the surrounding structure without us shipping whole files."""
        lines: list[str] = []
        seen: set[str] = set()
        for r in results:
            if r.source == "graph":
                continue
            neighbors = self.index.graph.neighbors(r.chunk.id, depth=1)
            if not neighbors:
                continue
            label = r.chunk.symbol_name or r.chunk.file_path
            for nb in neighbors:
                key = f"{label}->{nb.relation}->{nb.node.qualified_name}"
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"- {label} {nb.relation} {nb.node.qualified_name} ({nb.node.citation})"
                )
                if len(lines) >= max_lines:
                    return "\n".join(lines)
        return "\n".join(lines)
