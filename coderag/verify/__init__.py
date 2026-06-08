"""Citation verification & faithfulness (§5)."""
from .faithfulness import (
    structural_check,
    faithfulness_score,
    verify_answer,
    parse_citations,
)

__all__ = ["structural_check", "faithfulness_score", "verify_answer", "parse_citations"]
