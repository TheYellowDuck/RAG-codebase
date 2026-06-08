"""Code RAG — a code-aware retrieval-augmented generation system.

Indexes a codebase on AST boundaries, retrieves with hybrid dense+lexical search
and a code graph, then answers questions with Claude using verifiable [n] citations.

See outline.md for the full technical design; each module maps to a numbered section.
"""

__version__ = "0.1.0"
