"""Generation: context assembly + grounded answering (outline §4).

Assembly (§4.1): fill a token budget greedily by final rank, drop overlapping /
duplicate spans (oversized-function windows can overlap), and number the sources
so the model can cite [n] compactly and we can verify mechanically. Generation
goes through the provider-agnostic LLM layer (coderag/llm) — Claude by default,
or any provider the user configures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import Settings
from ..tokenization import token_len
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


def _overlaps(a: Source, file_path: str, start: int, end: int) -> bool:
    """True if [start,end] is contained in / duplicates an accepted same-file span."""
    if a.file_path != file_path:
        return False
    return start >= a.start_line and end <= a.end_line


def assemble_context(retrieved: list[RetrievedChunk],
                     token_budget: int) -> tuple[str, list[Source]]:
    """Dedup + budget-fill, then number the sources (§4.1)."""
    accepted: list[Source] = []
    used = 0
    n = 0
    for r in retrieved:
        c = r.chunk
        if any(_overlaps(a, c.file_path, c.start_line, c.end_line) for a in accepted):
            continue
        cost = token_len(c.code)
        if accepted and used + cost > token_budget:
            continue  # keep scanning — a smaller later chunk might still fit
        n += 1
        accepted.append(Source(
            n=n, chunk_id=c.id, file_path=c.file_path, start_line=c.start_line,
            end_line=c.end_line, code=c.code, symbol_name=c.symbol_name,
            source_type=r.source, relation=r.relation,
        ))
        used += cost
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
                    stream: bool = False, client=None) -> AnswerResult:
    """Answer `question` grounded in the retrieved chunks, with [n] citations.

    `client` is any coderag.llm.LLMClient; if omitted, the configured provider is
    resolved from the environment (Claude by default)."""
    numbered_sources, sources = assemble_context(retrieved, settings.context_token_budget)

    graph_section = ""
    if settings.include_graph_context and graph_context:
        graph_section = GRAPH_SECTION_TEMPLATE.format(graph_context=graph_context)

    system = SYSTEM_TEMPLATE.format(repo=repo)
    user = USER_TEMPLATE.format(
        question=question, numbered_sources=numbered_sources, graph_section=graph_section,
    )

    client = client or get_llm_client()
    completion = client.generate(system, user, max_tokens=settings.gen_max_tokens,
                                 stream=stream)
    return AnswerResult(answer=completion.text, sources=sources,
                        model=client.gen_model, usage=completion.usage)
