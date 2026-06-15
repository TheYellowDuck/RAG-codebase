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

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..schema import Chunk
from ..index import CodeIndex
from ..embed import Embedder
from ..tokenization import code_tokens
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


def rrf(*ranked_lists: list[str], k: int = 60,
        weights: Optional[list[float]] = None) -> list[str]:
    """Reciprocal Rank Fusion (§3.3). Each input is an ordered list of chunk_ids,
    best first. Fuse by rank — cosine and BM25 scores aren't comparable, so don't
    normalize them; RRF is parameter-light and beats either retriever alone.
    `weights` (per list) lets you up/down-weight dense vs lexical (default equal)."""
    scores: dict[str, float] = defaultdict(float)
    for i, ranked in enumerate(ranked_lists):
        w = weights[i] if weights else 1.0
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += w / (k + rank)
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
        self._llm = None  # lazily created only if HyDE is used (keeps retrieval key-free)

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.settings.rerank_model)
        return self._reranker

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, *, k: Optional[int] = None,
                 use_dense: Optional[bool] = None, use_bm25: Optional[bool] = None,
                 use_rerank: Optional[bool] = None, expand_graph: Optional[bool] = None,
                 use_hyde: Optional[bool] = None,
                 graph_rerank: Optional[bool] = None,
                 llm_rerank: Optional[bool] = None) -> list[RetrievedChunk]:
        s = self.settings
        k = k or s.final_k
        use_dense = s.use_dense if use_dense is None else use_dense
        use_bm25 = s.use_bm25 if use_bm25 is None else use_bm25
        use_rerank = s.use_rerank if use_rerank is None else use_rerank
        expand_graph = s.expand_graph if expand_graph is None else expand_graph
        use_hyde = s.use_hyde if use_hyde is None else use_hyde

        if not (use_dense or use_bm25):
            raise ValueError("Enable at least one of dense / bm25 retrieval.")

        ranked_lists: list[list[str]] = []
        weights: list[float] = []
        if use_dense:
            dense_query = self._hyde_query(query) if use_hyde else query
            qvec = self.embedder.encode_query(dense_query)
            dense = self.index.vector_store.search(qvec, s.dense_top_n)
            ranked_lists.append([cid for cid, _ in dense])
            weights.append(s.dense_weight)
        if use_bm25:
            lexical = self.index.bm25.search(query, s.bm25_top_n)
            ranked_lists.append([cid for cid, _ in lexical])
            weights.append(s.bm25_weight)

        # Single retriever → its own order; multiple → fuse by (weighted) rank.
        fused = (rrf(*ranked_lists, k=s.rrf_k, weights=weights)
                 if len(ranked_lists) > 1 else ranked_lists[0])
        candidate_ids = fused[: s.fuse_top_n]
        candidates = [self.index.get_chunk(cid) for cid in candidate_ids]
        candidates = [c for c in candidates if c is not None]

        # Graph as a recall booster: pull connected context into the rerank POOL so
        # it competes on query relevance (no fake scores, no forced injection).
        graph_rel: dict[str, str] = {}
        if expand_graph and s.graph_expand_prererank:
            candidates, graph_rel = self._add_neighbors_to_pool(candidates, s)
        if s.graph_pagerank:  # Aider-style: PPR-rank the connected subgraph to select
            candidates, graph_rel = self._add_pagerank_to_pool(candidates, s, graph_rel)

        graph_rerank = s.graph_rerank if graph_rerank is None else graph_rerank
        llm_rerank = s.llm_rerank if llm_rerank is None else llm_rerank
        if llm_rerank and candidates:
            top = self._llm_rerank(query, candidates, k, s.llm_rerank_pool)
            results = [self._mk(c, 1.0 / (i + 1), i + 1, graph_rel)
                       for i, c in enumerate(top)]
        elif use_rerank and candidates:
            scored = self.reranker.rerank(query, candidates, len(candidates))
            if s.graph_rerank_boost > 0:
                scored = self._apply_connectivity_boost(scored, s.graph_rerank_boost)
            scored = self._mmr(scored, k, s.mmr_lambda) if s.use_mmr else scored[:k]
            results = [self._mk(c, sc, i + 1, graph_rel)
                       for i, (c, sc) in enumerate(scored)]
        elif graph_rerank and candidates:
            top = self._graph_rerank(candidates, k, s.graph_rerank_seeds)
            results = [self._mk(c, 1.0 / (i + 1), i + 1, graph_rel)
                       for i, c in enumerate(top)]
        else:
            # Fall back to fused rank; synthesize a descending score.
            top = candidates[:k]
            results = [self._mk(c, 1.0 / (i + 1), i + 1, graph_rel)
                       for i, c in enumerate(top)]

        # Legacy: append neighbors AFTER rerank with a fake score (kept for ablation;
        # this is the path the eval showed adds distractors — see RESULTS §4).
        if expand_graph and not s.graph_expand_prererank:
            results = self._expand_with_graph(results, s.graph_expand_budget)
        return results

    def _llm_rerank(self, query: str, candidates: list[Chunk], k: int,
                    pool: int) -> list[Chunk]:
        """Listwise LLM reranking: show the top-`pool` candidates to the LLM and let
        it reason about which match the query — disambiguates near-duplicate symbols
        where cross-encoders fail (the one lever that significantly lifts recall@5,
        +0.086 on FastAPI p<0.001). Costs one LLM call/query; fails open to fused
        order on any error (no key, bad output)."""
        cand = candidates[:pool]
        if not cand:
            return candidates[:k]
        try:
            if self._llm is None:
                from ..llm import get_llm_client
                self._llm = get_llm_client()
            items = "\n\n".join(
                f"{i + 1}. {c.file_path}:{c.start_line}\n{(c.code or '')[:240]}"
                for i, c in enumerate(cand))
            txt = self._llm.generate(
                "You rank code-search candidates by relevance to the query.",
                f"Query: {query}\n\nCandidates:\n{items}\n\nOutput ONLY the numbers of "
                f"the {k} most relevant candidates, best first, comma-separated.",
                max_tokens=80).text
            order: list[int] = []
            for x in re.findall(r"\d+", txt):
                o = int(x)
                if 1 <= o <= len(cand) and o not in order:
                    order.append(o)
            ranked = [cand[o - 1] for o in order]
            chosen = {id(c) for c in ranked}
            ranked += [c for c in cand if id(c) not in chosen]   # fill from fused order
            return ranked[:k]
        except Exception:
            return candidates[:k]        # fail open

    def _graph_rerank(self, candidates: list[Chunk], k: int, seeds: int) -> list[Chunk]:
        """Re-rank the fused pool by rank-fusing its RRF order with a PPR-connectivity
        order (seeded by the top hits). Promotes pool members graph-connected to the
        top candidates — breaks near-duplicate-symbol ties without a cross-encoder."""
        order = [c.id for c in candidates]
        ppr = self.index.graph.personalized_pagerank(order[:seeds])
        if not ppr:
            return candidates[:k]
        conn = sorted(order, key=lambda cid: ppr.get(cid, 0.0), reverse=True)
        by_id = {c.id: c for c in candidates}
        return [by_id[cid] for cid in rrf(order, conn)[:k]]

    def _mk(self, chunk: Chunk, score: float, rank: int,
            graph_rel: dict[str, str]) -> RetrievedChunk:
        rel = graph_rel.get(chunk.id)
        return RetrievedChunk(chunk, score, rank,
                              source="graph" if rel else "retrieved", relation=rel)

    # ------------------------------------------------------------------ #
    def _hyde_query(self, query: str) -> str:
        """HyDE (§ retrieval): draft a hypothetical code snippet for the question and
        search with query+snippet — helps when the question's words differ from the
        code's. Fails open: any LLM error falls back to the raw query."""
        try:
            from ..llm import get_llm_client
            if self._llm is None:
                self._llm = get_llm_client()
            doc = self._llm.generate(
                "Write a short, realistic code snippet or docstring that would plausibly "
                "answer the user's question about a codebase. Output only code, no prose.",
                f"Question: {query}", max_tokens=256).text
            return f"{query}\n{doc}"
        except Exception:
            return query

    def _add_neighbors_to_pool(self, candidates: list[Chunk],
                               s: Settings) -> tuple[list[Chunk], dict[str, str]]:
        """Add 1-hop graph neighbors of the top fused candidates to the pool, so the
        reranker (not a fake score) decides whether they're relevant."""
        existing = {c.id for c in candidates}
        relations: dict[str, str] = {}
        pool = list(candidates)
        for c in candidates[: s.graph_expand_seed]:
            if len(relations) >= s.graph_pool_budget:
                break
            for nb in self.index.graph.neighbors(c.id, depth=s.graph_expand_depth):
                if len(relations) >= s.graph_pool_budget:
                    break
                if nb.chunk_id in existing:
                    continue
                chunk = self.index.get_chunk(nb.chunk_id)
                if chunk is None:
                    continue
                existing.add(nb.chunk_id)
                relations[nb.chunk_id] = nb.relation
                pool.append(chunk)
        return pool, relations

    def _add_pagerank_to_pool(self, candidates: list[Chunk], s: Settings,
                              graph_rel: dict[str, str]
                              ) -> tuple[list[Chunk], dict[str, str]]:
        """Rank the connected subgraph by personalized PageRank (restarting at the
        top retrieval hits) and add the highest-scoring connected chunks to the
        pool — graph used to *select* context, then the reranker confirms it."""
        seeds = [c.id for c in candidates[: s.graph_pagerank_seeds]]
        ppr = self.index.graph.personalized_pagerank(seeds)
        if not ppr:
            return candidates, graph_rel
        existing = {c.id for c in candidates}
        ranked = sorted((nid for nid in ppr if nid not in existing),
                        key=lambda n: ppr[n], reverse=True)
        pool = list(candidates)
        for nid in ranked[: s.graph_pagerank_add]:
            chunk = self.index.get_chunk(nid)
            if chunk is None:
                continue
            pool.append(chunk)
            graph_rel.setdefault(nid, "pagerank")
        return pool, graph_rel

    def _mmr(self, scored: list[tuple[Chunk, float]], k: int,
             lam: float) -> list[tuple[Chunk, float]]:
        """Maximal Marginal Relevance: pick k items trading relevance against
        diversity (code-token Jaccard) so the final set isn't near-duplicates."""
        if not scored:
            return []
        toks = {c.id: set(code_tokens(c.code)) for c, _ in scored}
        vals = [sc for _, sc in scored]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        rel = {c.id: (sc - lo) / rng for c, sc in scored}

        def jac(a: set, b: set) -> float:
            return len(a & b) / len(a | b) if (a or b) else 0.0

        pool = list(scored)
        selected: list[tuple[Chunk, float]] = []
        while pool and len(selected) < k:
            best, best_val = None, -1e9
            for c, sc in pool:
                div = max((jac(toks[c.id], toks[t.id]) for t, _ in selected), default=0.0)
                val = lam * rel[c.id] - (1 - lam) * div
                if val > best_val:
                    best_val, best = val, (c, sc)
            selected.append(best)
            pool = [(c, sc) for c, sc in pool if c.id != best[0].id]
        return selected

    def _apply_connectivity_boost(self, scored: list[tuple[Chunk, float]],
                                  boost: float) -> list[tuple[Chunk, float]]:
        """Graph-aware reranking: raise a chunk's score by `boost` × (summed positive
        score of its graph-neighbors that are also candidates). Connected-to-relevant
        is itself a relevance signal; this reorders, never adds chunks."""
        by_id = {c.id: sc for c, sc in scored}
        adjusted = []
        for c, sc in scored:
            bump = sum(max(0.0, by_id[nb.chunk_id])
                       for nb in self.index.graph.neighbors(c.id, depth=1)
                       if nb.chunk_id in by_id)
            adjusted.append((c, sc + boost * bump))
        return sorted(adjusted, key=lambda x: x[1], reverse=True)

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
