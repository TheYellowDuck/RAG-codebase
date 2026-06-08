"""Evaluation harness (§6) — the centerpiece."""
from .metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, retrieved_files
from .bootstrap import bootstrap_ci

__all__ = [
    "recall_at_k", "precision_at_k", "reciprocal_rank", "ndcg_at_k",
    "retrieved_files", "bootstrap_ci",
]
