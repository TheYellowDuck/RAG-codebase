"""Generation metrics (outline §6.3).

  Faithfulness          — supported claims / total (see verify/faithfulness.py)
  Answer correctness    — LLM-as-judge vs reference_answer on a small rubric
  Citation precision/recall — of cited files, how many are relevant (precision);
                              of relevant files, how many got cited (recall)
"""
from __future__ import annotations

from ..config import Settings
from ..llm import get_llm_client

_CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["correct", "partially_correct", "wrong"]},
        "reason": {"type": "string"},
    },
    "required": ["grade", "reason"],
    "additionalProperties": False,
}

_GRADE_SCORE = {"correct": 1.0, "partially_correct": 0.5, "wrong": 0.0}


def answer_correctness(question: str, answer: str, reference: str,
                       settings: Settings, client=None) -> dict:
    """LLM-as-judge against the reference answer (§6.3)."""
    client = client or get_llm_client()
    prompt = (
        "You are grading an answer about a codebase against a reference answer.\n"
        "Grade as 'correct', 'partially_correct', or 'wrong', with a one-line reason.\n"
        "Judge factual agreement with the reference, not wording.\n\n"
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answer: {answer}\n\n"
        'Return JSON: {"grade": ..., "reason": ...}'
    )
    data = client.judge_json(prompt, _CORRECTNESS_SCHEMA, max_tokens=512)
    grade = data.get("grade", "wrong")
    if grade not in _GRADE_SCORE:
        grade = "wrong"
    return {"grade": grade, "score": _GRADE_SCORE[grade],
            "reason": data.get("reason", "")}


def citation_precision_recall(sources, cited_ns: list[int],
                              relevant_files: list[str]) -> dict:
    """File-level citation precision/recall (§6.3).

    sources: list of generate.Source (carry n + file_path).
    cited_ns: the [n] actually cited in the answer.
    """
    relevant = set(relevant_files)
    by_n = {s.n: s.file_path for s in sources}
    cited_files = {by_n[n] for n in set(cited_ns) if n in by_n}
    if cited_files:
        precision = sum(f in relevant for f in cited_files) / len(cited_files)
    else:
        precision = 0.0
    recall = (len(cited_files & relevant) / len(relevant)) if relevant else 0.0
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "cited_files": sorted(cited_files),
    }
