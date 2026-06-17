"""Local embedding backend via sentence-transformers (the default stack).

The same model embeds both chunks (`embed_text`) and queries — that symmetry is
what makes cosine similarity meaningful. Vectors are L2-normalized so cosine
similarity reduces to a dot product in the vector store.

To swap in a code-specialized model, set CODERAG_EMBED_MODEL (e.g.
`jinaai/jina-embeddings-v2-base-code`). The model is loaded lazily so indexing
metadata and retrieval plumbing can be exercised without paying the import cost.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def infer_prefixes(model_name: str) -> tuple[str, str]:
    """Many modern retrieval embedders are *asymmetric* — they need a query prefix
    and a passage prefix or recall craters. We infer them from the model name so
    swapping models 'just works' (override via Settings.embed_query/doc_prefix).
    Returns (query_prefix, doc_prefix); empty for symmetric models like the default."""
    n = model_name.lower()
    if "coderank" in n or ("nomic" in n and "code" in n):   # nomic-ai/CodeRankEmbed
        return ("Represent this query for searching relevant code: ", "")
    if "e5" in n:                       # intfloat/e5-*, multilingual-e5-*
        return ("query: ", "passage: ")
    if "bge" in n and "-en" in n:       # BAAI/bge-*-en-v1.5
        return ("Represent this sentence for searching relevant passages: ", "")
    return ("", "")


def infer_max_seq_len(model_name: str) -> Optional[int]:
    """Some long-context embedders (e.g. CodeRankEmbed, 8192 tok) explode memory on
    batched padding ('Invalid buffer size') when indexing a real repo. Code chunks
    are small, so cap such models to a sane length. None = use the model's default."""
    n = model_name.lower()
    if "coderank" in n or ("nomic" in n and "code" in n):
        return 512
    return None


class Embedder:
    def __init__(self, model_name: str, *, query_prefix: Optional[str] = None,
                 doc_prefix: Optional[str] = None, max_seq_len: Optional[int] = None):
        self.model_name = model_name
        auto_q, auto_d = infer_prefixes(model_name)
        self.query_prefix = auto_q if query_prefix is None else query_prefix
        self.doc_prefix = auto_d if doc_prefix is None else doc_prefix
        self.max_seq_len = max_seq_len if max_seq_len is not None else infer_max_seq_len(model_name)
        self._model = None

    @classmethod
    def from_settings(cls, settings) -> "Embedder":
        return cls(settings.embed_model,
                   query_prefix=getattr(settings, "embed_query_prefix", None),
                   doc_prefix=getattr(settings, "embed_doc_prefix", None),
                   max_seq_len=getattr(settings, "embed_max_seq_len", None))

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "sentence-transformers is not installed. Run:\n"
                    "    pip install -r requirements.txt"
                ) from e
            # The strongest code embedders (jina-embeddings-v2-base-code,
            # nomic-embed-text) ship custom model code; opt in via env.
            import os
            trust = os.environ.get(
                "CODERAG_EMBED_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")
            kwargs = {"trust_remote_code": True} if trust else {}
            # Model download/load has no SDK-level retry — wrap it so a flaky
            # network doesn't fail the whole index/query (bad model names still
            # fail fast: they're not transient).
            from ..resilience import with_retry
            self._model = with_retry(
                lambda: SentenceTransformer(self.model_name, **kwargs),
                desc=f"load embed model '{self.model_name}'")
            if self.max_seq_len:        # cap long-context models to avoid memory blowups
                self._model.max_seq_length = self.max_seq_len
        return self._model

    @property
    def dim(self) -> int:
        # get_embedding_dimension() is the current name; fall back for older
        # sentence-transformers that only have get_sentence_embedding_dimension().
        m = self.model
        getter = (getattr(m, "get_embedding_dimension", None)
                  or m.get_sentence_embedding_dimension)
        return int(getter())

    def encode(self, texts: list[str], batch_size: int = 64,
               show_progress: bool = False, is_query: bool = False) -> np.ndarray:
        """Embed a batch. Returns float32, L2-normalized (N, D). `is_query` selects
        the query vs document prefix (a no-op for symmetric models)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.query_prefix if is_query else self.doc_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=show_progress, convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query with the same model (§3.1). Returns (D,)."""
        return self.encode([query], is_query=True)[0]
