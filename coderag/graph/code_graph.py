"""The code graph — connections between files, symbols, and calls.

Why this exists: so the model doesn't have to scan the whole codebase. Retrieval
finds the *entry point* chunk; the graph then tells us exactly which other chunks
are connected (callees, callers, imports, the enclosing class) so we can hand the
model a few precise neighbors plus a compact structural map — instead of dumping
entire files and burning tokens.

Nodes are keyed by chunk id (so a graph node *is* a retrievable/citable chunk).
Edges are resolved heuristically by name, which is cheap and good enough to be a
real token-saver without a full type-resolver.

Edge types (stored on the *source* node, with reverse links maintained too):
  contains   class  -> its methods
  calls      symbol -> the symbol it calls
  imports    module -> a symbol/module it imports
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

        # Pass 3: calls — resolve callee name to a defined symbol.
        for fp in file_parses:
            for caller_id, callee in fp.calls:
                target = cls._resolve(callee, by_qualified, by_simple)
                if target and target != caller_id:
                    g._add_edge(caller_id, target, EDGE_CALLS)

        # Pass 4: imports — link a module to symbols/files it imports.
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

        return g

    @staticmethod
    def _resolve(name: str, by_qualified: dict, by_simple: dict) -> Optional[str]:
        exact = by_qualified.get(name, [])
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return None  # qualified name collides across files — don't guess
        simple = name.rsplit(".", 1)[-1]
        candidates = by_simple.get(simple, [])
        if len(candidates) == 1:
            return candidates[0]
        # Ambiguous (or unknown) name — don't guess; an over-linked graph adds noise.
        return None

    def _add_edge(self, src: str, dst: str, edge_type: str) -> None:
        if (dst, edge_type) not in self.out_edges[src]:
            self.out_edges[src].append((dst, edge_type))
            self.in_edges[dst].append((src, edge_type))

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
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cid, nd in data.get("nodes", {}).items():
            g.nodes[cid] = GraphNode(**nd)
        for src, edges in data.get("out_edges", {}).items():
            for dst, etype in edges:
                g._add_edge(src, dst, etype)
        return g
