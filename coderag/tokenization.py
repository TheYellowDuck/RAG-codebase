"""Token counting (for budgets) and code-aware identifier tokenization (§3.2).

`token_len` is a cheap heuristic used only for chunk-size and context budgets —
not for billing — so an approximation is fine and avoids a network round-trip per
chunk. `code_tokens` is the domain-specific piece: it splits identifiers so that
`get_current_user` and `getCurrentUser` both match a "get current user" query.
"""
from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def token_len(text: str) -> int:
    """Rough token estimate (~4 chars/token). Good enough for budgeting."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def code_tokens(text: str) -> list[str]:
    """Tokenize code/text for BM25 so identifier variants all match (§3.2).

    Emits the whole lowercased identifier plus its camelCase pieces; snake_case
    is already split by the non-alphanumeric split.
    """
    out: list[str] = []
    for raw in _NON_ALNUM.split(text):
        if not raw:
            continue
        low = raw.lower()
        out.append(low)                               # whole identifier
        for part in _CAMEL.sub(" ", raw).split():     # camelCase pieces
            p = part.lower()
            if p != low:
                out.append(p)
    return out
