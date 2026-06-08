"""Scaffold golden questions from a built index — labels verified by construction.

Two stages:

1. `build_questions` mines the index + code graph for question candidates whose
   `relevant_files`/`relevant_symbols` are *structurally guaranteed* (derived from
   real `calls` edges and symbol definitions), weighted toward cross-file. Phrasing
   is templated and names the symbols (`How does X use Y?`).

2. `paraphrase_questions` (optional, LLM) rewrites each question to describe the
   behavior in natural language **without any identifiers or file names** — removing
   the lexical/BM25 confound that symbol-named questions introduce, while keeping the
   verified labels untouched.
"""
from __future__ import annotations

import random
import re

_STOP = {"self", "cls", "init", "main", "call", "run", "get", "set", "add", "len",
         "str", "int", "list", "dict", "type", "name", "value", "data", "args",
         "kwargs", "super", "print", "format", "join", "split", "append"}
_DOC = re.compile(r'"""(.+?)"""', re.DOTALL)


def _interesting(node) -> bool:
    s = node.simple_name
    return (node.kind in ("function", "method", "class")
            and len(s) >= 4 and not s.startswith("__")
            and s.lower() not in _STOP and "test" not in s.lower())


def _docstring(chunk) -> "str | None":
    if chunk is None:
        return None
    m = _DOC.search(chunk.context_header or "")
    if m:
        first = m.group(1).strip().splitlines()[0].strip()
        return first or None
    return None


def build_questions(index, n: int = 55, start_id: int = 46,
                    exclude_symbols: "set[str] | None" = None,
                    holdout_every: int = 4, seed: int = 7) -> list[dict]:
    """Generate `n` questions with construction-verified labels (see module doc)."""
    g = index.graph
    used = set(exclude_symbols or set())
    rng = random.Random(seed)

    cross: list[dict] = []
    seen_pairs: set[tuple] = set()
    for src, edges in g.out_edges.items():
        sn = g.nodes.get(src)
        if sn is None or not _interesting(sn) or sn.kind == "class":
            continue
        for dst, etype in edges:
            if etype != "calls":
                continue
            dn = g.nodes.get(dst)
            if dn is None or not _interesting(dn) or sn.file_path == dn.file_path:
                continue
            key = (sn.simple_name, dn.simple_name)
            if key in seen_pairs or sn.qualified_name in used or dn.qualified_name in used:
                continue
            seen_pairs.add(key)
            cross.append({
                "question": f"How does `{sn.simple_name}` use `{dn.simple_name}`?",
                "type": "cross-file",
                "relevant_files": sorted({sn.file_path, dn.file_path}),
                "relevant_symbols": [sn.qualified_name, dn.qualified_name],
                "reference_answer": f"{sn.qualified_name} ({sn.file_path}) calls "
                                    f"{dn.simple_name} ({dn.file_path}).",
            })

    single: list[dict] = []
    for cid, node in g.nodes.items():
        if not _interesting(node) or node.qualified_name in used:
            continue
        doc = _docstring(index.get_chunk(cid))
        if node.kind == "class":
            q = f"What does the `{node.simple_name}` class do?"
        else:
            q = f"Where is `{node.simple_name}` defined and what does it do?"
        single.append({
            "question": q,
            "type": "where" if not doc else "how-to",
            "relevant_files": [node.file_path],
            "relevant_symbols": [node.qualified_name],
            "reference_answer": doc or f"{node.qualified_name} is a {node.kind} "
                                       f"defined in {node.file_path}.",
        })

    rng.shuffle(cross)
    rng.shuffle(single)
    n_cross = min(len(cross), round(n * 0.65))
    chosen = cross[:n_cross] + single[: n - n_cross]
    rng.shuffle(chosen)

    out = []
    for i, q in enumerate(chosen[:n]):
        q = {"id": f"q{start_id + i:03d}", **q}
        if holdout_every and (i % holdout_every == holdout_every - 1):
            q["holdout"] = True
        out.append(q)
    return out


# --------------------------------------------------------------------------- #
# LLM paraphrase — remove identifiers/file names (de-confound), keep labels
# --------------------------------------------------------------------------- #
_PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "question": {"type": "string"}},
                "required": ["id", "question"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rewrites"],
    "additionalProperties": False,
}

_PARAPHRASE_INSTRUCTION = (
    "You are rewriting evaluation questions about a codebase. Rewrite EACH question "
    "so it describes the behavior in natural language and is still answerable, but "
    "WITHOUT naming any function, class, method, variable, or file (no identifiers, "
    "no `code_names`, no paths). Keep the meaning. Keep it one sentence.\n\n"
    "Questions:\n{listing}\n\n"
    'Return JSON: {{"rewrites": [{{"id": "qNNN", "question": "..."}}]}}'
)


def paraphrase_questions(questions: list[dict], client, batch_size: int = 12,
                         progress: bool = False) -> list[dict]:
    """Rewrite question text to be identifier-free via the LLM, preserving every
    other field. On any failure for a batch, the originals are kept."""
    by_id = {q["id"]: q for q in questions}
    items = list(questions)
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        listing = "\n".join(f'{q["id"]}: {q["question"]}' for q in batch)
        prompt = _PARAPHRASE_INSTRUCTION.format(listing=listing)
        if progress:
            print(f"[paraphrase] batch {i // batch_size + 1} ({len(batch)} questions)...")
        try:
            data = client.judge_json(prompt, _PARAPHRASE_SCHEMA, max_tokens=2048)
            for rw in data.get("rewrites", []):
                q = by_id.get(rw.get("id"))
                if q is not None and rw.get("question"):
                    q["question"] = rw["question"].strip()
        except Exception:
            continue  # keep originals for this batch
    return items
