"""Lightweight query routing (heuristic).

Classify a question so the caller can route it to the cheapest config that still
answers it: a direct lookup doesn't need the same machinery as a multi-hop trace.
Heuristic (no LLM) so it's free and deterministic; returns config *overrides* the
caller can pass to Retriever.retrieve(...)."""
from __future__ import annotations

import re

_MULTIHOP = re.compile(
    r"\b(trace|end[ -]?to[ -]?end|flow|lifecycle|across|pipeline|sequence|"
    r"from .+ to |how .+ (reach|become|propagat|travel|flow))", re.I)
_HOWTO = re.compile(r"^\s*(how\b|how do i\b|how to\b)", re.I)
_WHERE = re.compile(r"^\s*(where\b|which file|what file|locate\b)", re.I)


def classify_query(query: str) -> str:
    """Return one of: 'multihop' | 'howto' | 'where' | 'lookup'."""
    if _MULTIHOP.search(query):
        return "multihop"
    if _WHERE.search(query):
        return "where"
    if _HOWTO.search(query):
        return "howto"
    return "lookup"


def route(query: str) -> dict:
    """Suggested retrieve() overrides for the query class. Conservative: never turns
    OFF accuracy levers, only scales effort up for harder questions."""
    kind = classify_query(query)
    if kind == "multihop":
        # widen the net + give the reranker more to work with
        return {"use_dense": True, "use_bm25": True, "use_rerank": True, "k": 8}
    if kind == "where":
        # exact-identifier lookups: lexical matters most, rerank still cheap-win
        return {"use_dense": True, "use_bm25": True, "use_rerank": True, "k": 4}
    return {"use_dense": True, "use_bm25": True, "use_rerank": True}
