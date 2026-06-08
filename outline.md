# Code RAG — Technical Design & Implementation Guide

A companion to the project plan. This goes down to schemas, algorithms, code, prompts, and metric formulas — the level you implement from. Code is illustrative and idiomatic; verify exact library APIs (especially `tree-sitter`, which changes binding signatures between versions) at build time.

---

## 0. The data model everything hangs off

Get the chunk schema right first — citations, metadata filtering, reranking, and incremental indexing all depend on it.

```python
from dataclasses import dataclass

@dataclass
class Chunk:
    id: str               # stable: sha1(f"{file_path}:{start_line}-{end_line}")
    repo: str
    file_path: str        # relative to repo root — the citation anchor
    language: str
    symbol_name: str|None # qualified, e.g. "Router.dispatch"
    symbol_type: str      # "function" | "method" | "class" | "module"
    start_line: int       # 1-indexed, inclusive
    end_line: int
    code: str             # raw source of the span (shown in the UI)
    context_header: str   # synthesized location/signature context (see §2.3)
    embed_text: str       # context_header + "\n\n" + code  — what gets embedded
    git_sha: str          # commit indexed at — for staleness / incremental reindex
```

Two non-obvious fields earn their keep:

- **`context_header`** — code chunks are ambiguous in isolation (a method `dispatch` could be anywhere). Prepending structural context before embedding measurably improves retrieval *and* gives you the citation location for free.
- **`git_sha`** — lets you do incremental reindexing (§7) instead of rebuilding the whole index on every `git pull`.

The `id` doubling as `file:line` means a retrieved chunk *is* a citation. No separate bookkeeping.

---

## 1. Ingestion: file discovery

Before parsing, decide what's even a candidate. This step quietly determines index quality.

- Walk the repo, respect `.gitignore`.
- **Skip**: binaries, lockfiles, generated code, vendored deps (`node_modules/`, `vendor/`, `dist/`, `.venv/`), minified files, anything over a size cap (e.g. 1 MB single file or >5k lines — usually generated).
- Detect language by extension (and shebang for extensionless scripts).
- Keep a manifest: `{file_path: (git_sha, language, n_lines)}` — useful for incremental reindex and for sanity-checking coverage ("did I actually index the whole repo?").

A quiet failure mode: indexing test fixtures, generated protobufs, or huge data files drowns the real signal. Spot-check what got chunked.

---

## 2. Ingestion: code-aware chunking

This is the core technical contribution. Prose RAG splits on character/token windows; that shreds code — half a function retrieves as noise. You chunk on **AST boundaries**.

### 2.1 Parse with tree-sitter

```python
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY = Language(tspython.language())
parser = Parser(PY)
tree = parser.parse(source_bytes)   # source as bytes; tree-sitter is byte-oriented
```

You want one grammar per language. Start with Python only; adding a grammar later is the cheap way to demo generality.

### 2.2 The chunking algorithm

Walk the tree, emit a chunk at each definition boundary, recurse into classes so methods become their own chunks, and split functions that exceed your token budget.

```python
DEF_TYPES = {"function_definition", "class_definition"}

def chunk_tree(root, source: bytes, file_path, max_tokens=800):
    chunks = []

    def emit(node, qualified, symbol_type):
        span = source[node.start_byte:node.end_byte].decode("utf-8", "replace")
        if token_len(span) <= max_tokens:
            chunks.append(build_chunk(node, span, qualified, symbol_type, file_path, source))
        else:
            # oversized function: window it, but PREPEND the signature to every window
            chunks.extend(split_oversized(node, qualified, symbol_type, file_path, source, max_tokens))

    def walk(node, class_path):
        for child in node.children:
            if child.type == "class_definition":
                name = node_name(child, source)
                # a "class summary" chunk: signature + docstring + method signatures only
                chunks.append(class_summary_chunk(child, class_path, file_path, source))
                walk(child, class_path + [name])          # recurse → methods as chunks
            elif child.type == "function_definition":
                name = node_name(child, source)
                qualified = ".".join(class_path + [name]) if class_path else name
                kind = "method" if class_path else "function"
                emit(child, qualified, kind)
            else:
                walk(child, class_path)                   # keep descending

    walk(root, [])
    chunks.append(module_chunk(root, source, file_path))   # imports + top-level code
    return chunks
```

Design choices worth being able to defend in an interview:

- **Methods are chunks, classes get a summary chunk.** A giant class shouldn't be one blob. But you also want a "what is this class" chunk → emit signature + docstring + the list of method signatures. Now "what does class `Router` do?" and "what does `Router.dispatch` do?" both retrieve well.
- **Oversized functions get windowed *with the signature re-prepended* to every window.** A window from the middle of a 300-line function is useless without knowing which function it's in.
- **A module-level chunk** captures imports and top-level code, and is the fallback unit for files tree-sitter can't fully parse.
- **Fallback path**: if parsing fails (syntax error, unsupported construct), fall back to line-window chunking for that file so nothing is silently dropped.

### 2.3 The context header (cheap, big lift)

Before embedding, prepend a short synthesized header to each chunk. For a method, something like:

```
File: fastapi/routing.py
Class: APIRouter
def add_api_route(self, path, endpoint, *, methods=None, ...)
"""Register a route on the router."""
```

Then `embed_text = context_header + "\n\n" + code`. You embed the header+code; you store and display the raw `code`. This is the "contextual chunk" idea applied to code — the embedding now carries location and signature, which is exactly what queries like "where do I register a route" key off. Run your eval with and without it; the lift is usually large enough to be a slide on its own.

### 2.4 Embed and index

- Batch embeds (respect the model's batch limits) over `embed_text`.
- Store vectors + full `Chunk` metadata in the vector DB.
- **Build a BM25 index over the *same* chunks** (§3.2). Keep both indices keyed by `chunk.id` so fusion is trivial.

---

## 3. Retrieval

### 3.1 Dense (semantic) search

Embed the query with the *same* model, cosine top-N (N ≈ 30–50 before reranking, not your final k). Optional metadata pre-filters (language, path glob) if the UI offers scoping.

### 3.2 Lexical (BM25) — and why it matters more for code than prose

Developers query with exact identifiers (`HTTPException`, `solve_dependencies`). Embeddings smear these together; BM25 nails exact matches. The trick is **tokenizing code identifiers** so `get_current_user` and `getCurrentUser` both match a "get current user" query:

```python
import re
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

def code_tokens(text: str) -> list[str]:
    out = []
    for raw in _NON_ALNUM.split(text):
        if not raw:
            continue
        low = raw.lower()
        out.append(low)                                  # whole identifier
        for part in _CAMEL.sub(" ", raw).split():        # camelCase pieces
            p = part.lower()
            if p != low:
                out.append(p)
    return out                                            # snake_case already split by _NON_ALNUM
```

Tokenize both the chunk text and the query this way. This single function is a legitimately good interview talking point because it's specific to the domain and obviously correct once stated.

### 3.3 Fusion (Reciprocal Rank Fusion)

Don't try to normalize cosine scores against BM25 scores — they're not comparable. Fuse by *rank*:

```python
from collections import defaultdict

def rrf(*ranked_lists, k=60):
    # each ranked_list is an ordered list of chunk_ids, best first
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

`k=60` is the standard default and robust. RRF is parameter-light, needs no score calibration, and consistently beats either retriever alone — say exactly that.

### 3.4 Rerank

Take the fused top ~30, score each `(query, chunk)` pair with a cross-encoder reranker, keep the final top-k (k ≈ 5–8). Cross-encoders are slower (they attend over query+doc jointly) but far more precise than bi-encoder cosine. Measure the lift against eval; if it's marginal, keep it optional and talk about the latency/cost tradeoff (also a good answer to "how would you make this faster?").

Final retrieval output per chunk should carry full provenance (`file_path`, `start_line`, `end_line`, `code`) so generation can cite without another lookup.

---

## 4. Generation

### 4.1 Context assembly

- **Token budget**: cap context (e.g. 4–8k tokens). Fill greedily by final rank.
- **Dedup overlap**: oversized-function windows can overlap; merge adjacent spans from the same file or drop near-duplicates so you don't burn budget.
- **Number the sources** so the model can cite compactly and you can verify mechanically:

```
[1] fastapi/routing.py:412-455
<code>

[2] fastapi/dependencies/utils.py:88-140
<code>
```

### 4.2 Prompt

System:

```
You answer questions about the {repo} codebase using ONLY the provided sources.
Cite every claim with [n] referring to the source numbers.
If the sources do not contain the answer, say so explicitly — do not guess.
Prefer quoting identifiers and signatures exactly.
```

User:

```
Question: {question}

Sources:
{numbered_sources}

Answer with inline [n] citations.
```

The **abstain instruction is load-bearing**: a system that says "the retrieved code doesn't cover this" instead of confabulating is what makes faithfulness scores real, and it's the honest behavior. Demo a question the repo *can't* answer and show it declining — interviewers notice.

(Stricter alternative: ask for JSON `{"answer": "...", "claims": [{"text": "...", "sources": [1,3]}]}`. Easier to verify per-claim; slightly worse prose. Worth A/B-ing.)

---

## 5. Citation verification & faithfulness

Two layers, cheap → expensive:

**5.1 Structural check (free).** Parse `[n]` markers, confirm each maps to a real source you actually sent, and flag any sentence making a claim with *no* citation. Catches the easy failures with zero LLM cost.

**5.2 Faithfulness (LLM-as-judge, RAGAS-style).** Decompose the answer into atomic claims, then for each claim ask whether the cited source(s) support it:

```
Claim: "{claim}"
Source(s): {cited_source_text}
Is the claim fully supported by the source(s)? Answer SUPPORTED or UNSUPPORTED with a one-line reason.
```

`faithfulness = supported_claims / total_claims`. Surface unsupported claims in the UI as "low confidence." This is the literal deliverable behind "accurate citation," and most candidates have nothing here.

---

## 6. Evaluation — the centerpiece

If you build one thing well, build this. It converts "I made a RAG thing" into "I improved recall@5 from 0.61 → 0.88 and can show you the table."

### 6.1 Golden set schema

`data/golden_questions.jsonl`, one object per line:

```json
{
  "id": "q017",
  "question": "How does FastAPI coerce a path parameter to an int?",
  "type": "how-to",
  "relevant_files": ["fastapi/routing.py", "fastapi/dependencies/utils.py"],
  "relevant_symbols": ["get_request_handler", "solve_dependencies"],
  "reference_answer": "Path params declared with a type hint are turned into Pydantic fields; the hint drives validation/coercion, and a bad value yields a 422."
}
```

30–50 questions spanning *factual / how-to / where / cross-file* (the cross-file ones stress retrieval hardest). Label `relevant_files` by actually reading the repo — this is a few focused hours and it's the foundation of every number you'll report.

### 6.2 Retrieval metrics (exact formulas)

Evaluate at **file granularity** first (a retrieved chunk "hits" if its `file_path` is in `relevant_files`) — it's robust to chunk-boundary choices. Optionally also do chunk/symbol granularity.

```python
import math

def recall_at_k(retrieved, relevant, k):
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant) if relevant else 0.0

def precision_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    return sum(c in relevant for c in top) / len(top) if top else 0.0

def reciprocal_rank(retrieved, relevant):
    for i, c in enumerate(retrieved, 1):
        if c in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    dcg = sum((c in relevant) / math.log2(i + 1)
              for i, c in enumerate(retrieved[:k], 1))
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0
```

- **Recall@k** is the headline RAG metric: did the answer-bearing file make the top-k? If recall is low, no amount of prompt-tuning saves you — the LLM never saw the answer.
- **MRR** captures how high the first hit ranks.
- **NDCG@k** rewards getting relevant files near the top, not just present.

### 6.3 Generation metrics

- **Faithfulness** (§5.2): supported claims / total.
- **Answer correctness**: LLM-as-judge against `reference_answer` on a small rubric (correct / partially correct / wrong + reason). Average the scores.
- **Citation precision/recall**: of cited spans, how many are actually relevant (precision); of relevant info, how much got cited (recall).

### 6.4 The runner and the table

`eval/run.py` loops the golden set, runs retrieval + generation, computes every metric, and prints/saves a table. Run it after *each* change so improvements are attributable:

| Config | Recall@5 | MRR | NDCG@5 | Faithfulness |
|---|---|---|---|---|
| Baseline (window chunks, dense only) | 0.61 | 0.48 | 0.55 | 0.79 |
| + code-aware chunking | 0.72 | 0.57 | 0.66 | 0.83 |
| + context headers | 0.79 | 0.64 | 0.72 | 0.85 |
| + hybrid (BM25 + RRF) | 0.85 | 0.71 | 0.80 | 0.88 |
| + reranking | 0.88 | 0.76 | 0.83 | 0.90 |

(Numbers illustrative — yours will differ; the *shape* is the deliverable.)

### 6.5 Rigor touches that signal maturity

- **Hold out** ~10 questions you never look at while tuning, to check you didn't overfit the dev set.
- With N ≈ 40, report **bootstrap confidence intervals** on the means rather than bare point estimates — resample questions with replacement, recompute the metric, take the 2.5/97.5 percentiles. Cheap, and it shows you know a 0.85 vs 0.88 difference on 40 questions might be noise.
- Log per-question results so you can eyeball *which* questions regressed when a change helps on average.

---

## 7. Incremental indexing (high-value stretch)

Rebuilding the whole index on every commit is the obvious-but-wrong approach. Instead:

1. `git diff --name-status {old_sha}..{new_sha}` → added/modified/deleted files.
2. For deleted/modified files: delete all chunks where `chunk.file_path == path` (this is why file_path is a first-class field).
3. For added/modified files: re-chunk, re-embed, re-insert; update the BM25 index.
4. Store the new `git_sha` in the manifest.

This turns "reindex 150k LOC" into "reindex the 3 files that changed," and it's a clean systems-thinking story.

---

## 8. Common pitfalls (so you skip the painful version)

- **Skipping eval and tuning on vibes.** You'll "improve" things that regress and never know. Eval first.
- **Indexing junk** (generated code, fixtures, huge data files) — it dilutes retrieval. Audit coverage.
- **Char/token-window chunking on code.** The whole point is not doing this.
- **Dense-only retrieval.** You'll miss exact-identifier queries constantly; hybrid is not optional for code.
- **No abstain path.** A model that always answers will confabulate on out-of-scope questions and tank faithfulness.
- **Comparing raw cosine vs BM25 scores directly.** Fuse by rank (RRF), not by score.
- **One mega-chunk per file.** Kills precision; chunk by symbol.

---

## 9. First-week concrete checklist

1. Choose the repo; clone it; print LOC and file/language breakdown.
2. Write 30+ golden questions with labeled `relevant_files`. (Do this *before* building — it forces you to understand the repo and gives you your ruler.)
3. Naive window-chunk → embed → dense top-k → Claude answer with `[n]` citations (end-to-end, deliberately dumb).
4. Stand up `eval/run.py`; record **baseline** Recall@5 / MRR / NDCG@5 / faithfulness.

After week one you have a working, *measured* system. Everything after that is improving numbers you can already see — which is exactly the position you want to be in, both for the project and for the interview story.