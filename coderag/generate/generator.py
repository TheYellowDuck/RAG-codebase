"""Generation: context assembly + grounded answering (outline §4).

Assembly (§4.1): fill a token budget greedily by final rank, drop overlapping /
duplicate spans (oversized-function windows can overlap), and number the sources
so the model can cite [n] compactly and we can verify mechanically. Generation
goes through the provider-agnostic LLM layer (coderag/llm) — Claude by default,
or any provider the user configures.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from ..config import Settings
from ..tokenization import token_len, code_tokens
from ..retrieve import RetrievedChunk
from ..llm import get_llm_client
from .prompts import SYSTEM_TEMPLATE, USER_TEMPLATE, GRAPH_SECTION_TEMPLATE


@dataclass
class Source:
    n: int
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    code: str
    symbol_name: Optional[str] = None
    source_type: str = "retrieved"
    relation: Optional[str] = None

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class AnswerResult:
    answer: str
    sources: list[Source]
    model: str
    usage: dict = field(default_factory=dict)
    context_tokens: int = 0   # tokens of assembled source context (cost lever)


def _overlaps(a: Source, file_path: str, start: int, end: int) -> bool:
    """True if [start,end] is contained in / duplicates an accepted same-file span."""
    if a.file_path != file_path:
        return False
    return start >= a.start_line and end <= a.end_line


def _compact_code(code: str) -> str:
    """Collapse runs of blank lines and strip trailing whitespace. Lossless for
    meaning — never removes code or comments — so it's quality-safe."""
    lines = [ln.rstrip() for ln in code.split("\n")]
    out: list[str] = []
    blank = False
    for ln in lines:
        if not ln:
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(ln)
    return "\n".join(out).strip("\n")


def _trim_to_query(code: str, query: str, max_tokens: int) -> str:
    """Opt-in: for an over-budget chunk, keep its signature line plus a window
    around the lines that match query identifiers; ellipsize the rest. Only used
    when settings.trim_sources is on (it can drop code, so it's off by default)."""
    if token_len(code) <= max_tokens:
        return code
    lines = code.split("\n")
    qtokens = set(code_tokens(query))
    if not qtokens:
        return code  # nothing to anchor on — don't guess; keep full
    hits = [i for i, ln in enumerate(lines) if qtokens & set(code_tokens(ln))]
    if not hits:
        return code
    pad = 6
    keep = set()
    keep.add(0)  # signature line
    for h in hits:
        keep.update(range(max(0, h - pad), min(len(lines), h + pad + 1)))
    out, prev = [], -2
    for i in sorted(keep):
        if i - prev > 1:
            out.append("    # ...")
        out.append(lines[i])
        prev = i
    trimmed = "\n".join(out)
    return trimmed if token_len(trimmed) < token_len(code) else code


def _fit_to_budget(code: str, query: str, budget: int) -> str:
    """No single source may exceed the whole budget. Trim to query-relevant lines
    first (keeps the part the answer needs), then hard-cap as a backstop. This
    bounds pathological chunks like a 4k-line class summary."""
    if token_len(code) <= budget:
        return code
    fitted = _trim_to_query(code, query, budget)
    if token_len(fitted) <= budget:
        return fitted
    return fitted[: budget * 4].rstrip() + "\n    # ...(truncated to budget)"


def _gate_by_score(retrieved: list[RetrievedChunk], min_sources: int) -> list[RetrievedChunk]:
    """Drop sources the reranker scored as irrelevant (<0), but always keep at
    least `min_sources`. With no reranker, scores are synthetic positives, so this
    is a no-op — it only ever removes a clearly-irrelevant tail."""
    kept = [r for i, r in enumerate(retrieved) if i < min_sources or r.score >= 0]
    return kept or list(retrieved)


def _merge_adjacent(sources: list[Source]) -> list[Source]:
    """Fold a source into an already-accepted one from the same file when their
    line spans are contiguous (gap ≤ 2), saving a repeated header. Preserves rank
    order and only joins truly adjacent spans, so no lines are misrepresented."""
    out: list[Source] = []
    for s in sources:
        merged = False
        for t in out:
            if t.file_path != s.file_path:
                continue
            if 0 <= s.start_line - t.end_line <= 2:        # s follows t
                t.code = t.code + "\n" + s.code
                t.end_line = max(t.end_line, s.end_line)
                merged = True
                break
            if 0 <= t.start_line - s.end_line <= 2:        # s precedes t
                t.code = s.code + "\n" + t.code
                t.start_line = min(t.start_line, s.start_line)
                merged = True
                break
        if not merged:
            out.append(s)
    return out


def assemble_context(retrieved: list[RetrievedChunk], settings: Settings,
                     query: str = "") -> tuple[str, list[Source]]:
    """Select, shrink, dedup, and number the sources for the prompt (§4.1).

    Token savers (all quality-preserving by default): drop reranked-irrelevant
    tail, content-dedup, whitespace-compact, merge contiguous same-file spans, and
    budget-fill by rank. Per-source trimming is opt-in (settings.trim_sources)."""
    budget = settings.context_token_budget
    candidates = (_gate_by_score(retrieved, settings.min_sources)
                  if settings.drop_negative_rerank else list(retrieved))

    accepted: list[Source] = []
    used = 0
    seen_hashes: set[str] = set()
    for r in candidates:
        c = r.chunk
        code = c.code
        if settings.trim_sources:
            code = _trim_to_query(code, query, settings.max_source_tokens)
        if settings.compact_source_code:
            code = _compact_code(code)

        code = _fit_to_budget(code, query, budget)  # no source may blow the budget
        if any(_overlaps(a, c.file_path, c.start_line, c.end_line) for a in accepted):
            continue
        if settings.dedup_sources:
            h = hashlib.sha1(re.sub(r"\s+", "", code).encode()).hexdigest()
            if h in seen_hashes:
                continue
        cost = token_len(code)
        if accepted and used + cost > budget:
            continue  # keep scanning — a smaller later chunk might still fit
        accepted.append(Source(
            n=0, chunk_id=c.id, file_path=c.file_path, start_line=c.start_line,
            end_line=c.end_line, code=code, symbol_name=c.symbol_name,
            source_type=r.source, relation=r.relation,
        ))
        if settings.dedup_sources:
            seen_hashes.add(h)
        used += cost

    if settings.merge_adjacent_sources:
        accepted = _merge_adjacent(accepted)
    for i, s in enumerate(accepted, 1):     # number after merging
        s.n = i

    block = "\n\n".join(_render_source(s) for s in accepted)
    return block, accepted


def _render_source(s: Source) -> str:
    tag = f"[{s.n}] {s.citation}"
    if s.symbol_name:
        tag += f"  ({s.symbol_name})"
    if s.source_type == "graph" and s.relation:
        tag += f"  [graph: {s.relation}]"
    return f"{tag}\n{s.code}"


def generate_answer(question: str, retrieved: list[RetrievedChunk],
                    settings: Settings, *, repo: str = "this",
                    graph_context: Optional[str] = None,
                    stream: bool = False, client=None,
                    extra_system: str = "") -> AnswerResult:
    """Answer `question` grounded in the retrieved chunks, with [n] citations.

    `client` is any coderag.llm.LLMClient; if omitted, the configured provider is
    resolved from the environment (Claude by default). `extra_system` appends an
    extra instruction (used by the self-repair retry)."""
    numbered_sources, sources = assemble_context(retrieved, settings, query=question)

    graph_section = ""
    if settings.include_graph_context and graph_context:
        graph_section = GRAPH_SECTION_TEMPLATE.format(graph_context=graph_context)

    system = SYSTEM_TEMPLATE.format(repo=repo)
    if extra_system:
        system = f"{system}\n\n{extra_system}"
    user = USER_TEMPLATE.format(
        question=question, numbered_sources=numbered_sources, graph_section=graph_section,
    )

    client = client or get_llm_client()
    completion = client.generate(system, user, max_tokens=settings.gen_max_tokens,
                                 stream=stream)
    return AnswerResult(answer=completion.text, sources=sources,
                        model=client.gen_model, usage=completion.usage,
                        context_tokens=token_len(numbered_sources))


_REPAIR_INSTRUCTION = (
    "IMPORTANT: cite a source [n] for EVERY factual sentence. If a claim is not "
    "directly supported by the provided sources, omit it or say the sources don't "
    "cover it. Do not speculate."
)


def answer_with_repair(question: str, retrieved: list[RetrievedChunk],
                       settings: Settings, *, repo: str = "this",
                       graph_context: Optional[str] = None, client=None) -> AnswerResult:
    """generate_answer + an opt-in self-repair retry: if the first answer's
    faithfulness is below settings.self_repair_threshold, regenerate once with a
    stricter cite-or-drop instruction and keep whichever scores higher."""
    from ..verify import faithfulness_score  # lazy import avoids a cycle
    client = client or get_llm_client()
    first = generate_answer(question, retrieved, settings, repo=repo,
                            graph_context=graph_context, client=client)
    if settings.self_repair_threshold <= 0:
        return first
    score1 = faithfulness_score(first.answer, first.sources, settings, client=client)
    if score1.get("faithfulness", 1.0) >= settings.self_repair_threshold:
        return first
    retry = generate_answer(question, retrieved, settings, repo=repo,
                            graph_context=graph_context, client=client,
                            extra_system=_REPAIR_INSTRUCTION)
    score2 = faithfulness_score(retry.answer, retry.sources, settings, client=client)
    return retry if score2.get("faithfulness", 0) >= score1.get("faithfulness", 0) else first
