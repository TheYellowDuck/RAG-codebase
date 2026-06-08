"""Lexical (BM25) index — and why it matters more for code than prose (§3.2).

Developers query with exact identifiers (`HTTPException`, `solve_dependencies`).
Embeddings smear these together; BM25 nails exact matches. The trick is
tokenizing code identifiers (see tokenization.code_tokens) so `get_current_user`
and `getCurrentUser` both match a "get current user" query.

We index over the same chunks as the dense store, keyed by chunk.id, so fusion
(§3.3) is trivial. rank-bm25 has no incremental add, so we keep the tokenized
docs and rebuild the scorer when the corpus changes — cheap at this scale.
"""
from __future__ import annotations

import json
import os

from ..tokenization import code_tokens


class BM25Index:
    def __init__(self):
        self.ids: list[str] = []
        self._tokens: dict[str, list[str]] = {}
        self._bm25 = None
        self._doc_sets: list[set] = []
        self._dirty = True

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], texts: list[str]) -> None:
        for cid, text in zip(ids, texts):
            if cid not in self._tokens:
                self.ids.append(cid)
            self._tokens[cid] = code_tokens(text)
        self._dirty = True

    def remove(self, ids: set[str]) -> None:
        if not ids:
            return
        self.ids = [cid for cid in self.ids if cid not in ids]
        for cid in ids:
            self._tokens.pop(cid, None)
        self._dirty = True

    def _ensure_built(self) -> None:
        if not self._dirty and self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "rank-bm25 is not installed. Run: pip install -r requirements.txt"
            ) from e
        corpus = [self._tokens[cid] for cid in self.ids]
        # BM25Okapi needs a non-empty corpus with at least one token.
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._doc_sets = [set(toks) for toks in corpus]
        self._dirty = False

    def search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        """Return up to top_n (chunk_id, bm25_score), best first.

        We restrict to documents that share at least one query token, then rank
        those by BM25. Filtering by token overlap (rather than score > 0) is the
        right semantics for code identifier search: on small corpora BM25's IDF
        can go to zero/negative for common terms, which would otherwise drop
        genuine exact-identifier hits.
        """
        self._ensure_built()
        if self._bm25 is None or not self.ids:
            return []
        qtokens = code_tokens(query)
        if not qtokens:
            return []
        qset = set(qtokens)
        scores = self._bm25.get_scores(qtokens)
        ranked = [(cid, float(s)) for cid, s, docset in zip(self.ids, scores, self._doc_sets)
                  if qset & docset]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    # --- persistence ------------------------------------------------------
    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "bm25.json"), "w") as f:
            json.dump({"ids": self.ids, "tokens": self._tokens}, f)

    @classmethod
    def load(cls, dir_path: str) -> "BM25Index":
        idx = cls()
        path = os.path.join(dir_path, "bm25.json")
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            idx.ids = data["ids"]
            idx._tokens = data["tokens"]
            idx._dirty = True
        return idx
