"""Generation: context assembly + grounded answering with [n] citations (§4)."""
from .generator import (
    AnswerResult, Source, assemble_context, generate_answer, answer_with_repair,
)

__all__ = ["AnswerResult", "Source", "assemble_context", "generate_answer",
           "answer_with_repair"]
