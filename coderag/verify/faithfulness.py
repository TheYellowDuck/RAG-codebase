"""Citation verification & faithfulness (outline §5). Two layers, cheap → expensive.

§5.1 Structural check (free): parse [n] markers, confirm each maps to a real
source, and flag claim sentences with no citation. Zero LLM cost.

§5.2 Faithfulness (LLM-as-judge, RAGAS-style): decompose the answer into atomic
claims, then for each claim ask whether its cited source(s) support it.
faithfulness = supported_claims / total_claims. The literal deliverable behind
"accurate citation."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..llm import get_llm_client

_CITATION = re.compile(r"\[(\d+)\]")
# Sentences that are honest non-answers shouldn't be flagged as uncited claims.
_ABSTAIN_HINTS = (
    "do not contain", "don't contain", "not contain", "no information",
    "cannot determine", "can't determine", "not covered", "not present",
    "does not appear", "doesn't appear", "no source", "not found in",
    "unable to", "insufficient", "not enough information",
)


def parse_citations(answer: str) -> list[int]:
    """Every [n] cited in the answer (with duplicates), in order."""
    return [int(m) for m in _CITATION.findall(answer)]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def structural_check(answer: str, num_sources: int) -> dict:
    """Free structural validation (§5.1)."""
    cited = parse_citations(answer)
    valid = sorted({n for n in cited if 1 <= n <= num_sources})
    invalid = sorted({n for n in cited if not (1 <= n <= num_sources)})

    uncited_claims: list[str] = []
    for sent in _split_sentences(answer):
        if _CITATION.search(sent):
            continue
        low = sent.lower()
        if any(h in low for h in _ABSTAIN_HINTS):
            continue  # honest abstention, not an unsupported claim
        if len(sent.split()) >= 5:  # ignore short fragments / headers
            uncited_claims.append(sent)

    return {
        "n_citations": len(cited),
        "valid_citations": valid,
        "invalid_citations": invalid,
        "uncited_claim_sentences": uncited_claims,
        "all_citations_valid": not invalid,
    }


# --------------------------------------------------------------------------- #
# LLM-as-judge (§5.2)
# --------------------------------------------------------------------------- #
@dataclass
class Claim:
    text: str
    sources: list[int]
    supported: Optional[bool] = None
    reason: str = ""


_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "sources"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]},
                    "reason": {"type": "string"},
                },
                "required": ["index", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def _extract_claims(client, answer: str) -> list[Claim]:
    prompt = (
        "Decompose the following answer into atomic factual claims. For each claim, "
        "list the source numbers it cites via [n] markers (empty list if none).\n\n"
        f"Answer:\n{answer}\n\n"
        "Return JSON: {\"claims\": [{\"text\": ..., \"sources\": [n, ...]}]}"
    )
    data = client.judge_json(prompt, _CLAIMS_SCHEMA)
    return [Claim(text=c["text"], sources=list(c.get("sources", [])))
            for c in data.get("claims", [])]


def _verify_claims(client, claims: list[Claim],
                   source_by_n: dict[int, str]) -> None:
    """Mutate claims in place with supported/reason. One batched call."""
    to_check = [(i, c) for i, c in enumerate(claims) if c.sources]
    for c in claims:
        if not c.sources:  # a claim with no citation can't be supported by a source
            c.supported = False
            c.reason = "No source cited."
    if not to_check:
        return

    blocks = []
    for i, c in to_check:
        cited_text = "\n---\n".join(
            f"[{n}] {source_by_n.get(n, '(missing source)')}" for n in c.sources
        )
        blocks.append(f"Claim {i}: \"{c.text}\"\nSource(s):\n{cited_text}")
    prompt = (
        "For each claim, decide whether it is fully supported by its cited "
        "source(s). Answer SUPPORTED or UNSUPPORTED with a one-line reason.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn JSON: {\"results\": [{\"index\": i, \"verdict\": ..., \"reason\": ...}]}"
    )
    data = client.judge_json(prompt, _VERIFY_SCHEMA)
    by_index = {r["index"]: r for r in data.get("results", [])}
    for i, c in to_check:
        r = by_index.get(i)
        if r:
            c.supported = (r["verdict"] == "SUPPORTED")
            c.reason = r.get("reason", "")
        else:
            c.supported = False
            c.reason = "Judge returned no verdict."


def faithfulness_score(answer: str, sources, settings: Settings,
                       client=None) -> dict:
    """RAGAS-style faithfulness (§5.2). `sources` is a list of generate.Source
    (or any object/dict with .n and .code)."""
    client = client or get_llm_client()
    source_by_n = {_get(s, "n"): _get(s, "code") for s in sources}

    claims = _extract_claims(client, answer)
    if not claims:
        return {"faithfulness": 1.0, "n_claims": 0, "n_supported": 0, "claims": []}

    _verify_claims(client, claims, source_by_n)
    n_supported = sum(1 for c in claims if c.supported)
    return {
        "faithfulness": n_supported / len(claims),
        "n_claims": len(claims),
        "n_supported": n_supported,
        "claims": [vars(c) for c in claims],
        "unsupported": [c.text for c in claims if not c.supported],
    }


def verify_answer(answer: str, sources, settings: Settings, client=None,
                  run_llm_judge: bool = True) -> dict:
    """Run both layers and return a combined report.

    The structural check (free, no LLM) always runs and is always returned. If the
    LLM faithfulness judge fails (e.g. a small local model emits invalid JSON),
    that's recorded under `faithfulness_error` instead of discarding the structural
    result — the cheap check is exactly what you want when the judge is flaky."""
    report = {"structural": structural_check(answer, len(sources))}
    if run_llm_judge:
        try:
            report["faithfulness"] = faithfulness_score(answer, sources, settings, client)
        except Exception as e:
            report["faithfulness_error"] = f"{type(e).__name__}: {e}"
    return report


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key)
