"""Prompts for grounded answering (outline §4.2).

The abstain instruction is load-bearing: a system that says "the retrieved code
doesn't cover this" instead of confabulating is what makes faithfulness scores
real — and it's the honest behavior.
"""

SYSTEM_TEMPLATE = """You answer questions about the {repo} codebase using ONLY the provided sources.
Cite every claim with [n] referring to the source numbers.
If the sources do not contain the answer, say so explicitly — do not guess.
Prefer quoting identifiers and signatures exactly."""

USER_TEMPLATE = """Question: {question}

Sources:
{numbered_sources}
{graph_section}
Answer with inline [n] citations."""

GRAPH_SECTION_TEMPLATE = """
Related code (from the code graph — use to understand structure; cite only the numbered sources above):
{graph_context}
"""
