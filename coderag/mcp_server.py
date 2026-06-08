"""MCP server — expose the code RAG over stdio so an agent (Claude Code, Cursor,
Claude Desktop) can search and ask questions about an indexed repo.

This is what makes the system *used* rather than demoed: any MCP client can call
`search_code` (key-free retrieval) and `answer_question` (grounded, cited answer —
needs an LLM provider/key).

Run:
    pip install 'coderag[mcp]'
    CODERAG_INDEX_DIR=.coderag_index_code python -m coderag.mcp_server

Register with a client (e.g. Claude Desktop config):
    "coderag": {"command": "python", "args": ["-m", "coderag.mcp_server"],
                "env": {"CODERAG_INDEX_DIR": "/abs/path/.coderag_index_code"}}
"""
from __future__ import annotations

import os

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The MCP SDK isn't installed. Run:  pip install 'coderag[mcp]'"
    ) from e

from .index import CodeIndex
from .retrieve import Retriever

_INDEX_DIR = os.environ.get("CODERAG_INDEX_DIR", ".coderag_index")
mcp = FastMCP("coderag")

_retriever: "Retriever | None" = None


def _retr() -> Retriever:
    """Load the index once, lazily (first tool call)."""
    global _retriever
    if _retriever is None:
        index = CodeIndex.load(_INDEX_DIR)
        _retriever = Retriever(index, index.settings)
    return _retriever


@mcp.tool()
def search_code(query: str, k: int = 6) -> str:
    """Search the indexed codebase and return the top matching code chunks, each
    with its `file:line` citation and symbol. Retrieval only — no LLM/key needed."""
    results = _retr().retrieve(query, k=k)
    if not results:
        return "No matching code found."
    blocks = []
    for x in results:
        head = f"[{x.citation}] {x.chunk.symbol_name or x.chunk.file_path}"
        blocks.append(f"{head}\n{x.chunk.code}")
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
def answer_question(query: str) -> str:
    """Answer a question about the codebase with verifiable [n] citations, grounded
    only in retrieved code (abstains if the answer isn't in the repo). Needs an LLM
    provider configured (ANTHROPIC_API_KEY or CODERAG_LLM_PROVIDER=...)."""
    from .generate import generate_answer
    r = _retr()
    results = r.retrieve(query, k=r.settings.final_k)
    graph_ctx = r.graph_context(results) if r.settings.include_graph_context else None
    ans = generate_answer(query, results, r.settings, repo=r.index.repo,
                          graph_context=graph_ctx)
    sources = "\n".join(f"[{s.n}] {s.citation}" for s in ans.sources)
    return f"{ans.answer}\n\nSources:\n{sources}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
