"""Approximate dense index via HNSW (hnswlib) — the scale backend.

Drop-in alternative to the exact `VectorStore`, behind the *same* interface
(`add` / `remove` / `search` / `save` / `load` / `len`). HNSW gives sublinear
queries on large corpora at the cost of being *approximate* (a little recall for a
lot of speed), so it's opt-in (`CODERAG_VECTOR_BACKEND=hnsw`) and not the default —
exact stays default because this project measures recall and small indexes don't
need ANN. Vectors are L2-normalized by the embedder; we use cosine space, and
report similarity = 1 - distance so scores match the exact backend's semantics.

Requires `hnswlib` (`pip install 'coderag[ann]'`), imported lazily.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


def _import_hnswlib():
    try:
        import hnswlib
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The HNSW backend needs hnswlib. Install it with:\n"
            "    pip install 'coderag[ann]'   (or: pip install hnswlib)\n"
            "or use the default exact backend (CODERAG_VECTOR_BACKEND=exact)."
        ) from e
    return hnswlib


class HNSWVectorStore:
    def __init__(self, dim: Optional[int] = None, *, M: int = 16,
                 ef_construction: int = 200, ef_search: int = 64):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ids: list[str] = []
        self._id_to_label: dict[str, int] = {}
        self._label_to_id: dict[int, str] = {}
        self._next_label = 0
        self._index = None

    def __len__(self) -> int:
        return len(self.ids)

    def _ensure_index(self, dim: int, capacity: int) -> None:
        if self._index is None:
            hnswlib = _import_hnswlib()
            self.dim = dim
            self._index = hnswlib.Index(space="cosine", dim=dim)
            self._index.init_index(max_elements=max(capacity, 16),
                                   ef_construction=self.ef_construction, M=self.M)
            self._index.set_ef(self.ef_search)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        v = np.asarray(vectors, dtype=np.float32)
        self._ensure_index(v.shape[1], len(ids))
        need = self._next_label + len(ids)
        if need > self._index.get_max_elements():
            self._index.resize_index(int(need * 1.5) + 16)
        labels = []
        for cid in ids:
            lbl = self._next_label
            self._next_label += 1
            self._id_to_label[cid] = lbl
            self._label_to_id[lbl] = cid
            self.ids.append(cid)
            labels.append(lbl)
        self._index.add_items(v, np.asarray(labels, dtype=np.int64))

    def remove(self, ids: set[str]) -> None:
        """Mark chunks deleted (incremental reindex). Marked elements are excluded
        from results; the graph isn't rebuilt (cheap, and consistent)."""
        if not ids or self._index is None:
            return
        for cid in ids:
            lbl = self._id_to_label.pop(cid, None)
            if lbl is not None:
                try:
                    self._index.mark_deleted(lbl)
                except Exception:  # already deleted / not present
                    pass
                self._label_to_id.pop(lbl, None)
        doomed = set(ids)
        self.ids = [c for c in self.ids if c not in doomed]

    def search(self, query_vec: np.ndarray, top_n: int) -> list[tuple[str, float]]:
        """Return up to top_n (chunk_id, cosine_score), best first (approximate)."""
        if self._index is None or not self.ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        n = min(top_n, len(self.ids))
        self._index.set_ef(max(self.ef_search, n))   # ef must be >= k
        labels, distances = self._index.knn_query(q, k=n)
        out = []
        for lbl, dist in zip(labels[0], distances[0], strict=True):
            cid = self._label_to_id.get(int(lbl))
            if cid is not None:
                out.append((cid, 1.0 - float(dist)))  # cosine distance -> similarity
        return out

    # --- persistence ------------------------------------------------------
    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        if self._index is not None:
            self._index.save_index(os.path.join(dir_path, "hnsw.bin"))
        with open(os.path.join(dir_path, "hnsw_meta.json"), "w") as f:
            json.dump({"ids": self.ids, "dim": self.dim, "M": self.M,
                       "ef_construction": self.ef_construction,
                       "ef_search": self.ef_search,
                       "id_to_label": self._id_to_label,
                       "next_label": self._next_label}, f)

    @classmethod
    def load(cls, dir_path: str) -> "HNSWVectorStore":
        with open(os.path.join(dir_path, "hnsw_meta.json")) as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"], M=meta["M"],
                    ef_construction=meta["ef_construction"], ef_search=meta["ef_search"])
        store.ids = meta["ids"]
        store._id_to_label = {k: int(v) for k, v in meta["id_to_label"].items()}
        store._label_to_id = {int(v): k for k, v in store._id_to_label.items()}
        store._next_label = meta["next_label"]
        bin_path = os.path.join(dir_path, "hnsw.bin")
        if os.path.isfile(bin_path):
            hnswlib = _import_hnswlib()
            idx = hnswlib.Index(space="cosine", dim=meta["dim"])
            idx.load_index(bin_path, max_elements=max(store._next_label, 16))
            idx.set_ef(meta["ef_search"])
            store._index = idx
        return store
