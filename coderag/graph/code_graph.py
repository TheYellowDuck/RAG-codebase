"""The code graph — connections between files, symbols, and calls.

Why this exists: so the model doesn't have to scan the whole codebase. Retrieval
finds the *entry point* chunk; the graph then says which other chunks connect to it
(callees, callers, imports, the enclosing class) so we can hand the model a few
precise neighbors plus a compact structural map instead of dumping whole files.

Nodes are keyed by chunk id (so a graph node *is* a retrievable/citable chunk).

Edge types (stored on the *source* node; reverse links maintained for traversal):
  contains   class  -> its methods
  calls      symbol -> the symbol it calls
  imports    module -> a symbol/module it imports

Resolution is heuristic but high-precision (no full type inference). An edge is
added only when the target resolves confidently, in this priority order:
  1. a unique definition of the name;
  2. else a definition in the caller's OWN file (language scoping);
  3. else a definition in a file the caller IMPORTS the name from (import-aware);
  4. else nothing — ambiguous names are left unlinked rather than guessed.

Beyond lookup the graph supports BFS context expansion (`neighbors`), Aider-style
context selection (`personalized_pagerank`), editing (`add_edge`/`remove_edge`/
`find_ids`), and rendering (graph/viz.py — DOT / Mermaid / interactive HTML).

Empirical note: the graph's *retrieval* value is conditional — significant on a
typed-language repo with a dense call graph (Go), null on Python where strong
retrieval already finds the files (RESULTS.md §3a). Expansion is opt-in.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

EDGE_CONTAINS = "contains"
EDGE_CALLS = "calls"
EDGE_IMPORTS = "imports"
# Human-readable inverse labels for the reverse direction.
_INVERSE = {EDGE_CONTAINS: "contained_by", EDGE_CALLS: "called_by", EDGE_IMPORTS: "imported_by"}


@dataclass
class GraphNode:
    chunk_id: str
    qualified_name: str
    simple_name: str
    kind: str            # function | method | class | module
    file_path: str
    start_line: int
    end_line: int

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class Neighbor:
    chunk_id: str
    relation: str        # e.g. "calls", "called_by", "contains", "imported_by"
    node: GraphNode


@dataclass
class CodeGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    # out[src] = list of (dst, edge_type); _in maintained for reverse traversal
    out_edges: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    in_edges: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, file_parses: Iterable) -> "CodeGraph":
        """Resolve symbol/call/import records (from chunk_file) into a graph."""
        g = cls()
        by_simple: dict[str, list[str]] = defaultdict(list)
        by_qualified: dict[str, list[str]] = defaultdict(list)
        by_file: dict[str, str] = {}  # file_path -> module chunk id

        # Pass 1: register every symbol as a node.
        for fp in file_parses:
            for sym in fp.symbols:
                node = GraphNode(
                    chunk_id=sym.chunk_id, qualified_name=sym.qualified_name,
                    simple_name=sym.simple_name, kind=sym.kind,
                    file_path=sym.file_path, start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                g.nodes[sym.chunk_id] = node
                by_simple[sym.simple_name].append(sym.chunk_id)
                by_qualified[sym.qualified_name].append(sym.chunk_id)
                if sym.kind == "module":
                    by_file[sym.file_path] = sym.chunk_id

        # Pass 2: containment (class -> method) from qualified-name nesting.
        # Only link when the parent resolves uniquely, to stay high-precision.
        for cid, node in g.nodes.items():
            qualified = node.qualified_name
            if "." in qualified:
                parent = qualified.rsplit(".", 1)[0]
                parent_ids = by_qualified.get(parent, [])
                if len(parent_ids) == 1 and parent_ids[0] != cid:
                    g._add_edge(parent_ids[0], cid, EDGE_CONTAINS)

        node_file = {cid: n.file_path for cid, n in g.nodes.items()}

        # Pass 3: imports FIRST — link module -> imported symbol/file, and record a
        # per-file import map (file -> imported simple_name -> source file[s]) so the
        # call pass can resolve ambiguous names the way the language actually does.
        import_map: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        for fp in file_parses:
            for module_id, name in fp.imports:
                target = cls._resolve(name, by_qualified, by_simple)
                if target is None:
                    # try dotted module path -> file (fastapi.routing -> fastapi/routing.py)
                    cand = name.replace(".", "/")
                    for suffix in (".py", "/__init__.py"):
                        target = by_file.get(cand + suffix)
                        if target:
                            break
                if target and target != module_id:
                    g._add_edge(module_id, target, EDGE_IMPORTS)
                    tnode = g.nodes.get(target)
                    if tnode and node_file.get(module_id):
                        import_map[node_file[module_id]][tnode.simple_name].add(tnode.file_path)

        # Pass 4: calls — resolve the callee. Disambiguation, high-precision first:
        # (1) caller's own file, (2) a file the caller IMPORTS the name from, else skip.
        for fp in file_parses:
            for caller_id, callee in fp.calls:
                caller_file = node_file.get(caller_id)
                simple = callee.rsplit(".", 1)[-1]
                import_files = import_map.get(caller_file, {}).get(simple)
                target = cls._resolve(callee, by_qualified, by_simple,
                                      prefer_file=caller_file, node_file=node_file,
                                      import_files=import_files)
                if target and target != caller_id:
                    g._add_edge(caller_id, target, EDGE_CALLS)

        return g

    @staticmethod
    def _resolve(name: str, by_qualified: dict, by_simple: dict,
                 prefer_file: Optional[str] = None,
                 node_file: Optional[dict] = None,
                 import_files: Optional[set] = None) -> Optional[str]:
        def disambiguate(cands: list[str]) -> Optional[str]:
            if prefer_file and node_file:               # (1) caller's own file wins
                same = [c for c in cands if node_file.get(c) == prefer_file]
                if len(same) == 1:
                    return same[0]
            if import_files and node_file:              # (2) a file the caller imports from
                imp = [c for c in cands if node_file.get(c) in import_files]
                if len(imp) == 1:
                    return imp[0]
            return None                                 # still ambiguous -> don't guess

        exact = by_qualified.get(name, [])
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return disambiguate(exact)  # qualified collision — same-file/import or skip
        candidates = by_simple.get(name.rsplit(".", 1)[-1], [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return disambiguate(candidates)
        return None  # unknown name — don't guess; an over-linked graph adds noise

    # ------------------------------------------------------------------ #
    def _adjacency(self) -> "dict[str, set]":
        """Undirected adjacency, cached (rebuilt lazily after edge/node mutations).
        Rebuilding this per call was the dominant cost of repeated PPR at scale."""
        adj = getattr(self, "_adj_cache", None)
        if adj is None:
            adj = defaultdict(set)
            nodes = self.nodes
            for src, edges in self.out_edges.items():
                if src not in nodes:
                    continue
                for dst, _ in edges:
                    if dst in nodes:
                        adj[src].add(dst)
                        adj[dst].add(src)
            self._adj_cache = adj
        return adj

    def personalized_pagerank(self, seeds: Iterable[str], alpha: float = 0.85,
                              iters: int = 30, max_frontier: int = 2000) -> dict[str, float]:
        """Personalized PageRank over the (undirected) graph, restarting at `seeds`.

        Nodes well-connected to the retrieved seeds score high — the Aider-style use
        of the graph: rank the connected subgraph to *select* context, rather than
        blindly expanding 1-hop neighbors. Pure power iteration (no extra deps).

        Sparse + bounded: we propagate only over nonzero nodes and keep the top
        `max_frontier` each iteration. PPR mass concentrates near the seeds, so this
        preserves the high-scoring (rankable) nodes while staying fast on a 40k-node
        graph — without it, repeated per-query PPR is seconds/query.

        Scores are an *unnormalized relative connectivity* signal, not a probability
        distribution: mass on neighbour-less (dangling) nodes is dropped rather than
        redistributed. That only scales scores, and every consumer uses their rank
        order, so it doesn't affect results — it just keeps the iteration cheap."""
        if not self.nodes:
            return {}
        adj = self._adjacency()
        seeds = [s for s in seeds if s in self.nodes]
        if not seeds:
            return {}
        seed_set = set(seeds)
        tele = 1.0 / len(seeds)
        score: dict[str, float] = {s: tele for s in seeds}
        for _ in range(iters):
            new: dict[str, float] = defaultdict(float)
            for nid, sc in score.items():
                nbrs = adj.get(nid)
                if nbrs:
                    share = alpha * sc / len(nbrs)
                    for m in nbrs:
                        new[m] += share
            for s in seeds:
                new[s] += (1 - alpha) * tele
            if len(new) > max_frontier:   # keep the high-mass frontier; the tail can't rank
                kept = dict(sorted(new.items(), key=lambda kv: kv[1], reverse=True)[:max_frontier])
                for s in seed_set:        # a seed must never be evicted from the frontier
                    if s in new:
                        kept[s] = new[s]
                new = kept
            # PPR mass concentrates fast; stop once an iteration barely moves so we
            # don't burn the full `iters` on an already-converged ranking.
            delta = sum(abs(new.get(k, 0.0) - score.get(k, 0.0)) for k in set(new) | set(score))
            score = new
            if delta < 1e-9:
                break
        return dict(score)

    def _add_edge(self, src: str, dst: str, edge_type: str) -> None:
        if (dst, edge_type) not in self.out_edges[src]:
            self.out_edges[src].append((dst, edge_type))
            self.in_edges[dst].append((src, edge_type))
            self._adj_cache = None      # invalidate cached adjacency

    # Public edit API (used by the graph editor / `graph-edit` CLI).
    def add_edge(self, src: str, dst: str, edge_type: str = EDGE_CALLS) -> None:
        if src in self.nodes and dst in self.nodes:
            self._add_edge(src, dst, edge_type)

    def remove_edge(self, src: str, dst: str, edge_type: Optional[str] = None) -> int:
        """Remove edge(s) src→dst (optionally of a given type). Returns count removed."""
        before = len(self.out_edges.get(src, []))
        self.out_edges[src] = [(d, t) for d, t in self.out_edges.get(src, [])
                               if not (d == dst and (edge_type is None or t == edge_type))]
        self.in_edges[dst] = [(s, t) for s, t in self.in_edges.get(dst, [])
                              if not (s == src and (edge_type is None or t == edge_type))]
        self._adj_cache = None          # invalidate cached adjacency
        return before - len(self.out_edges[src])

    def find_ids(self, name: str) -> list[str]:
        """Chunk ids whose qualified or simple name matches `name`."""
        return [cid for cid, n in self.nodes.items()
                if n.qualified_name == name or n.simple_name == name]

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def neighbors(self, chunk_id: str, depth: int = 1,
                  relations: Optional[set[str]] = None) -> list[Neighbor]:
        """Connected chunks up to `depth` hops, nearest first, de-duplicated.

        This is the call that lets us expand a single retrieved chunk into its
        relevant context without touching the rest of the repo.
        """
        seen = {chunk_id}
        result: list[Neighbor] = []
        frontier = [chunk_id]
        for _ in range(max(1, depth)):
            nxt: list[str] = []
            for cid in frontier:
                for dst, etype in self.out_edges.get(cid, []):
                    if relations and etype not in relations:
                        continue
                    if dst in seen or dst not in self.nodes:
                        continue
                    seen.add(dst)
                    result.append(Neighbor(dst, etype, self.nodes[dst]))
                    nxt.append(dst)
                for src, etype in self.in_edges.get(cid, []):
                    rel = _INVERSE.get(etype, etype)
                    if relations and rel not in relations and etype not in relations:
                        continue
                    if src in seen or src not in self.nodes:
                        continue
                    seen.add(src)
                    result.append(Neighbor(src, rel, self.nodes[src]))
                    nxt.append(src)
            frontier = nxt
            if not frontier:
                break
        return result

    def remove_file(self, file_path: str) -> None:
        """Drop all nodes/edges for a file (for incremental reindex, §7)."""
        doomed = [cid for cid, n in self.nodes.items() if n.file_path == file_path]
        doomed_set = set(doomed)
        for cid in doomed:
            self.nodes.pop(cid, None)
            self.out_edges.pop(cid, None)
            self.in_edges.pop(cid, None)
        for edges in self.out_edges.values():
            edges[:] = [(d, t) for (d, t) in edges if d not in doomed_set]
        for edges in self.in_edges.values():
            edges[:] = [(s, t) for (s, t) in edges if s not in doomed_set]
        self._adj_cache = None          # invalidate cached adjacency

    def merge(self, other: "CodeGraph") -> None:
        """Merge another graph in (used when re-adding changed files)."""
        self.nodes.update(other.nodes)
        for src, edges in other.out_edges.items():
            for dst, etype in edges:
                self._add_edge(src, dst, etype)

    def stats(self) -> dict:
        n_edges = sum(len(v) for v in self.out_edges.values())
        by_type: dict[str, int] = defaultdict(int)
        for edges in self.out_edges.values():
            for _, etype in edges:
                by_type[etype] += 1
        return {"nodes": len(self.nodes), "edges": n_edges, "by_type": dict(by_type)}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        data = {
            "nodes": {cid: vars(n) for cid, n in self.nodes.items()},
            "out_edges": {cid: edges for cid, edges in self.out_edges.items() if edges},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "CodeGraph":
        g = cls()
        if not os.path.isfile(path):
            return g
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for cid, nd in data.get("nodes", {}).items():
            g.nodes[cid] = GraphNode(**nd)
        for src, edges in data.get("out_edges", {}).items():
            for dst, etype in edges:
                g._add_edge(src, dst, etype)
        return g
