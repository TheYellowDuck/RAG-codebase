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


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

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
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        """Embed a batch of documents. Returns float32, L2-normalized (N, D)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=show_progress, convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query with the same model (§3.1). Returns (D,)."""
        return self.encode([query])[0]
