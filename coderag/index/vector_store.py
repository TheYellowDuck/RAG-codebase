"""Dense (semantic) index — a normalized-vector store with cosine top-N (§3.1).

Kept deliberately simple: a single in-memory float32 matrix plus an id list,
persisted with numpy. For the repo sizes this project targets (tens of thousands
of chunks) a brute-force matmul is fast, exact, and dependency-free. For large
indexes there's an opt-in approximate backend (`hnsw_store.HNSWVectorStore`,
selected by `make_vector_store`) behind this same interface — see CODERAG_VECTOR_BACKEND.
Vectors are assumed L2-normalized by the embedder, so cosine similarity == dot product.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def make_vector_store(settings):
    """Pick the dense backend from settings: 'exact' (default, brute-force) or
    'hnsw' (approximate ANN for large indexes). Same interface either way."""
    backend = (getattr(settings, "vector_backend", "exact") or "exact").lower()
    if backend == "hnsw":
        from .hnsw_store import HNSWVectorStore
        return HNSWVectorStore(M=getattr(settings, "hnsw_m", 16),
                               ef_construction=getattr(settings, "hnsw_ef_construction", 200),
                               ef_search=getattr(settings, "hnsw_ef_search", 64))
    return VectorStore()


def load_vector_store(settings, dir_path: str):
    """Load whichever backend `settings` selects from `dir_path`."""
    backend = (getattr(settings, "vector_backend", "exact") or "exact").lower()
    if backend == "hnsw":
        from .hnsw_store import HNSWVectorStore
        return HNSWVectorStore.load(dir_path)
    return VectorStore.load(dir_path)


class VectorStore:
    def __init__(self, dim: Optional[int] = None):
        self.dim = dim
        self.ids: list[str] = []
        self._matrix: Optional[np.ndarray] = None  # (N, D) float32

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.dim is None:
            self.dim = vectors.shape[1]
        if self._matrix is None:
            self._matrix = vectors.copy()
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        self.ids.extend(ids)

    def remove(self, ids: set[str]) -> None:
        """Drop chunks by id (incremental reindex). Rebuilds the matrix — fine
        at this scale, and keeps the row/id mapping trivially consistent."""
        if not ids or self._matrix is None:
            return
        keep = [i for i, cid in enumerate(self.ids) if cid not in ids]
        self._matrix = self._matrix[keep] if keep else None
        self.ids = [self.ids[i] for i in keep]

    def search(self, query_vec: np.ndarray, top_n: int) -> list[tuple[str, float]]:
        """Return up to top_n (chunk_id, cosine_score), best first."""
        if self._matrix is None or len(self.ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        scores = self._matrix @ q  # cosine, since both sides are normalized
        n = min(top_n, len(self.ids))
        # Stable sort so equal-cosine ties (duplicated/near-identical code, exact
        # 1.0 matches) resolve by row/insertion order — deterministic across runs
        # and NumPy versions, for both membership AND order. (argpartition + a
        # non-stable argsort left the tie order — and which tied chunk made the
        # cut — undefined, which propagates through RRF/rerank into final results.)
        idx = np.argsort(-scores, kind="stable")[:n]
        return [(self.ids[i], float(scores[i])) for i in idx]

    # --- persistence ------------------------------------------------------
    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        if self._matrix is not None:
            np.save(os.path.join(dir_path, "vectors.npy"), self._matrix)
        with open(os.path.join(dir_path, "vector_ids.json"), "w") as f:
            json.dump({"ids": self.ids, "dim": self.dim}, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        store = cls()
        ids_path = os.path.join(dir_path, "vector_ids.json")
        vec_path = os.path.join(dir_path, "vectors.npy")
        if os.path.isfile(ids_path):
            with open(ids_path) as f:
                meta = json.load(f)
            store.ids = meta["ids"]
            store.dim = meta.get("dim")
        if os.path.isfile(vec_path):
            store._matrix = np.load(vec_path)
        return store
