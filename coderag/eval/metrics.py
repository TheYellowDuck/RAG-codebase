"""Retrieval metrics (outline §6.2), exact formulas.

Evaluated at file granularity by default — a retrieved chunk "hits" if its
file_path is in relevant_files. That's robust to chunk-boundary choices.

  Recall@k  — did the answer-bearing file make the top-k? (the headline metric)
  Precision@k
  MRR       — how high the first hit ranks
  NDCG@k    — rewards relevant files near the top, not just present
"""
from __future__ import annotations

import math
from typing import Iterable


def retrieved_files(results) -> list[str]:
    """Ordered, de-duplicated file paths from retrieval results (best first).

    `results` items may be RetrievedChunk, generate.Source, or plain strings.
    """
    out: list[str] = []
    seen: set[str] = set()
    for r in results:
        fp = _file_of(r)
        if fp and fp not in seen:
            seen.add(fp)
            out.append(fp)
    return out


def _file_of(r) -> str:
    if isinstance(r, str):
        return r
    if hasattr(r, "chunk"):
        return r.chunk.file_path
    if hasattr(r, "file_path"):
        return r.file_path
    if isinstance(r, dict):
        return r.get("file_path", "")
    return ""


def recall_at_k(retrieved: list, relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: list, relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    top = retrieved[:k]
    return sum(c in relevant for c in top) / len(top) if top else 0.0


def reciprocal_rank(retrieved: list, relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    for i, c in enumerate(retrieved, 1):
        if c in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list, relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    dcg = sum((c in relevant) / math.log2(i + 1)
              for i, c in enumerate(retrieved[:k], 1))
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------------------- #
# Symbol granularity (optional, §6.2). A retrieved chunk hits a relevant symbol
# if the qualified names match, share a simple name, or one is a suffix of the
# other (so "add_api_route" matches "APIRouter.add_api_route").
# --------------------------------------------------------------------------- #
def retrieved_symbols(results) -> list[str]:
    """Ordered, de-duplicated symbol names from retrieval results (best first).
    Window-part suffixes (`name#2`) are normalized to the base symbol."""
    out: list[str] = []
    seen: set[str] = set()
    for r in results:
        sym = _symbol_of(r)
        if not sym:
            continue
        sym = sym.split("#", 1)[0]
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _symbol_of(r) -> str:
    if isinstance(r, str):
        return r
    if hasattr(r, "chunk"):
        return r.chunk.symbol_name or ""
    if hasattr(r, "symbol_name"):
        return r.symbol_name or ""
    if isinstance(r, dict):
        return r.get("symbol_name") or ""
    return ""


def symbol_match(a: str, b: str) -> bool:
    if a == b:
        return True
    sa, sb = a.rsplit(".", 1)[-1], b.rsplit(".", 1)[-1]
    return sa == sb or a.endswith("." + b) or b.endswith("." + a)


def recall_at_k_symbol(retrieved_syms: list[str], relevant_syms: Iterable[str],
                       k: int) -> float:
    relevant = list(relevant_syms)
    if not relevant:
        return 0.0
    top = retrieved_syms[:k]
    hits = sum(any(symbol_match(r, rel) for r in top) for rel in relevant)
    return hits / len(relevant)


def reciprocal_rank_symbol(retrieved_syms: list[str],
                           relevant_syms: Iterable[str]) -> float:
    relevant = list(relevant_syms)
    for i, r in enumerate(retrieved_syms, 1):
        if any(symbol_match(r, rel) for rel in relevant):
            return 1.0 / i
    return 0.0
