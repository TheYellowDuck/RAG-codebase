"""Cross-encoder reranking (outline §3.4).

Cross-encoders score each (query, chunk) pair jointly, so they're far more
precise than bi-encoder cosine — at the cost of latency (you run the model once
per candidate). We rerank the fused top ~30 down to the final k. Measure the lift
against eval; if marginal, it stays optional (the latency/cost knob).

Local model by default (cross-encoder/ms-marco-MiniLM-L-6-v2); loaded lazily.
"""
from __future__ import annotations

from typing import Optional

from ..schema import Chunk


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "sentence-transformers is not installed. Run:\n"
                    "    pip install -r requirements.txt"
                ) from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[Chunk],
               top_k: int) -> list[tuple[Chunk, float]]:
        """Return the top_k candidates by cross-encoder score, best first.

        We score against the embed_text (header + code) so the signature/location
        context informs the relevance judgment, same as retrieval.
        """
        if not candidates:
            return []
        pairs = [(query, c.embed_text) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_k]]
