"""The data model everything hangs off (outline §0).

Get the chunk schema right first — citations, metadata filtering, reranking, the
code graph, and incremental indexing all depend on it. The `id` doubling as
file:line means a retrieved chunk *is* a citation; no separate bookkeeping.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Optional


def chunk_id(file_path: str, start_line: int, end_line: int) -> str:
    """Stable id: sha1(file:start-end). Survives reindexing as long as the span
    is unchanged, which is what lets incremental reindex (§7) be precise."""
    return hashlib.sha1(f"{file_path}:{start_line}-{end_line}".encode()).hexdigest()


def build_embed_text(context_header: str, code: str, use_context_header: bool) -> str:
    """What actually gets embedded (§2.3). With the header on, the embedding
    carries location + signature — exactly what code queries key off."""
    if use_context_header and context_header:
        return f"{context_header}\n\n{code}"
    return code


@dataclass
class Chunk:
    id: str                      # stable: sha1(f"{file_path}:{start_line}-{end_line}")
    repo: str
    file_path: str               # relative to repo root — the citation anchor
    language: str
    symbol_name: Optional[str]   # qualified, e.g. "Router.dispatch"
    symbol_type: str             # "function" | "method" | "class" | "module" | "window"
    start_line: int              # 1-indexed, inclusive
    end_line: int
    code: str                    # raw source of the span (shown in the UI)
    context_header: str          # synthesized location/signature context (§2.3)
    embed_text: str              # context_header + "\n\n" + code — what gets embedded
    git_sha: str                 # commit (or content hash) indexed at — for staleness

    @property
    def citation(self) -> str:
        """Human-readable citation anchor, e.g. fastapi/routing.py:412-455."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def make_chunk(
    *,
    repo: str,
    file_path: str,
    language: str,
    symbol_name: Optional[str],
    symbol_type: str,
    start_line: int,
    end_line: int,
    code: str,
    context_header: str,
    git_sha: str,
    use_context_header: bool = True,
) -> Chunk:
    cid = chunk_id(file_path, start_line, end_line)
    return Chunk(
        id=cid,
        repo=repo,
        file_path=file_path,
        language=language,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        start_line=start_line,
        end_line=end_line,
        code=code,
        context_header=context_header,
        embed_text=build_embed_text(context_header, code, use_context_header),
        git_sha=git_sha,
    )
