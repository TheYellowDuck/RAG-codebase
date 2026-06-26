"""External-validity benchmark: CodeSearchNet-style code retrieval.

A self-made golden set proves you can build an eval; an *external* benchmark proves
the retriever generalizes beyond your own labels. CodeSearchNet pairs a function's
docstring (query) with its code (target): index all snippets, query with each
docstring, and measure whether the matching snippet ranks in the top-k.

This adapter is dataset-format-tolerant (accepts the common field names) and reuses
the project's Embedder + metrics. Running on the real corpus needs the dataset
(it's not bundled — download a `.jsonl` split from the CodeSearchNet release); the
logic is unit-tested on a tiny synthetic sample so the harness itself is verified.
"""
from __future__ import annotations

import json
from typing import Optional

from ..config import Settings

_QUERY_FIELDS = ("query", "docstring", "func_documentation_string", "doc")
_CODE_FIELDS = ("code", "func_code_string", "function", "whole_func_string")


def _first(d: dict, names) -> str:
    for n in names:
        v = d.get(n)
        if v:
            return v if isinstance(v, str) else " ".join(v)
    return ""


def load_codesearchnet(path: str) -> list[dict]:
    """Load (query, code) pairs from a CodeSearchNet-style JSONL file."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q, c = _first(d, _QUERY_FIELDS), _first(d, _CODE_FIELDS)
            if q and c:
                out.append({"id": str(d.get("id", i)), "query": q, "code": c})
    return out


def evaluate_codesearchnet(path: str, embedder=None, k: int = 10,
                           limit: Optional[int] = None) -> dict:
    """recall@k + MRR for retrieving each example's own code from its docstring.

    `embedder` defaults to the configured model; pass a stub for tests."""
    examples = load_codesearchnet(path)
    if limit:
        examples = examples[:limit]
    if not examples:
        return {"n": 0}
    if embedder is None:
        from ..embed import Embedder
        embedder = Embedder.from_settings(Settings.from_env())

    code_vecs = embedder.encode([e["code"] for e in examples])
    query_vecs = embedder.encode([e["query"] for e in examples], is_query=True)
    sims = query_vecs @ code_vecs.T   # vectors are L2-normalized -> cosine

    n = len(examples)
    hits = 0
    mrr = 0.0
    for i in range(n):
        # rank = 1 + (# docs strictly outscoring the gold). Deterministic under
        # score ties (np.argsort's tie order is undefined, so the gold's position
        # in it could drift); identical to argsort when there are no ties.
        rank = int((sims[i] > sims[i, i]).sum()) + 1   # gold for query i is code i
        if rank <= k:
            hits += 1
        mrr += 1.0 / rank
    return {"n": n, f"recall@{k}": hits / n, "mrr": mrr / n}
