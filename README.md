# Code RAG

A code-aware retrieval-augmented generation system. It indexes a codebase on
**AST boundaries** (not character windows) across **~all languages** — precise,
empirically-derived specs for **18 mainstream languages** (Python, JS, TS, Go,
Rust, Ruby, Java, C, C++, C#, PHP, Kotlin, Scala, Swift, Lua, Bash, Perl,
Objective-C) plus a generic pattern-based chunker for every other tree-sitter
grammar (~165 via `tree-sitter-language-pack`), with a graceful line-window
fallback when no grammar is available — retrieves with **hybrid dense + BM25
search fused by RRF and reranked by a cross-encoder**, builds a **code graph**
(imports / calls / class-method containment) so the model gets connected context
without scanning whole files, and answers questions with **Claude** using
verifiable `[n]` citations plus a **faithfulness** check. Ships with an
**evaluation harness** that turns "I made a RAG thing" into "here is the
recall@5 / MRR / NDCG@5 / faithfulness table."

This is the implementation of [`outline.md`](outline.md); every module maps to a
numbered section there.

---

## Why it's built this way

- **AST chunking (§2).** Prose RAG splits on token windows; that shreds code —
  half a function retrieves as noise. We chunk on definition boundaries: methods
  are their own chunks, classes get a summary chunk (signature + docstring +
  method signatures), oversized functions are windowed with the signature carried
  in a context header, and a module chunk captures imports + top-level code.
- **Context headers (§2.3).** Before embedding, each chunk is prefixed with
  `File / Class / signature / docstring`. The embedding then carries location and
  signature — exactly what code queries key off.
- **Hybrid retrieval (§3).** Developers query exact identifiers (`HTTPException`)
  that embeddings smear together; BM25 nails them. A code-aware tokenizer splits
  `get_current_user` / `getCurrentUser` so both match "get current user". Dense
  and lexical results are fused by **Reciprocal Rank Fusion** (no score
  calibration) and reranked by a cross-encoder.
- **Code graph (token-saver).** Retrieval finds the *entry-point* chunk; the
  graph tells us exactly which other chunks connect to it (callees, callers,
  imports, enclosing class). We hand the model a few precise neighbors plus a
  compact structural map instead of dumping entire files — so tokens are spent
  only on code that matters.
- **Abstain + faithfulness (§4–5).** The prompt forbids guessing; an LLM-judge
  decomposes the answer into atomic claims and checks each against its cited
  source. `faithfulness = supported / total`.
- **Evaluation first (§6).** Retrieval metrics at file granularity, generation
  metrics, bootstrap confidence intervals, a holdout split, and per-question
  logs — so every change is attributable.

---

## Install

```bash
cd RAG-codebase
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or install as a package (gives you the `coderag` command):
pip install -e .                 # add '.[langs]' for ~all languages, '.[dev]' for tests
```

The default stack is **local**: `sentence-transformers` for embeddings and a
local cross-encoder for reranking (first run downloads a few hundred MB). The
only API key needed is for generation/verification:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # get one at https://console.anthropic.com/
# or drop it in a local .env file (auto-loaded): ANTHROPIC_API_KEY=sk-ant-...
```

Indexing and retrieval work **without** any key; only generation (`query`,
`eval --generate`) needs one, and it fails with a clear message if it's missing.

### Setting your API key (kept out of git)

Put the key in a local **`.env`** file — never `export` it (that leaks into shell
history) and never paste it into a tracked file. `.env` is gitignored and the CLI
auto-loads it.

```bash
cp .env.example .env          # template → your local, ignored copy
# edit .env and set your real key, e.g.:  ANTHROPIC_API_KEY=sk-ant-...
# (put real keys ONLY in .env — never in .env.example, which is committed)

git status                    # confirm .env is NOT listed (it's ignored)
bash scripts/check_secrets.sh # confirm no key-shaped string in tracked files
python -m coderag.cli query "How does FastAPI register a route?" --index .coderag_index_code
```

> ⚠️ If a real key ever lands in a **committed** file (e.g. `.env.example`) or your
> shell history, treat it as compromised and **rotate it** in the provider console —
> deleting the line isn't enough once it's in git history. Note: exported shell
> variables override `.env`, so `unset` any stale ones (or open a fresh terminal).

### Bring your own LLM provider

Generation and judging go through a small provider layer ([coderag/llm/](coderag/llm/)),
so you're not locked to one vendor — the provider is resolved at runtime from the
environment (it's **not** baked into the index, so others can point the same index
at their own key). Default is Anthropic (Claude). Pick one:

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export CODERAG_LLM_PROVIDER=openai OPENAI_API_KEY=sk-...

# Any OpenAI-compatible endpoint — OpenRouter (→ Claude/Gemini/Llama/…),
# Together, Groq, Azure, or local Ollama / LM Studio / vLLM (free, offline):
export CODERAG_LLM_PROVIDER=openai \
       CODERAG_LLM_BASE_URL=http://localhost:11434/v1 \
       OPENAI_API_KEY=ollama CODERAG_GEN_MODEL=qwen2.5-coder
```

OpenAI/compatible needs the extra: `pip install 'coderag[openai]'`. Per-provider
model defaults apply (`claude-opus-4-8`, `gpt-4o`); override with
`CODERAG_GEN_MODEL` / `CODERAG_JUDGE_MODEL`. See [.env.example](.env.example).

**Keeping your keys out of git (any provider).** No key is ever hardcoded —
every provider reads its key only from the environment or a local `.env`. The
`.gitignore` excludes **all** dotenv variants (`.env`, `.env.*`, `*.env`) plus
`*.pem`/`*.key`/`secrets.json`, so a key for any provider stays local; only the
[.env.example](.env.example) placeholder is committed (`cp .env.example .env`).
Before pushing, run the provider-agnostic scan (catches OpenAI/Anthropic/Google/
AWS/GitHub key shapes), or wire it as a pre-commit hook:

```bash
bash scripts/check_secrets.sh
# optional: ln -s ../../scripts/check_secrets.sh .git/hooks/pre-commit
```

If a key ever lands in a commit, treat it as compromised and rotate it — git
history retains it. (GitHub's push-protection secret scanning is a second net.)

**The embedding model matters a lot for code.** The default is a code-search
model (`flax-sentence-embeddings/st-codesearch-distilroberta-base`), chosen
because it dominates a general-text model: on the FastAPI golden set (dense-only,
recall@5) the default scores **0.95** vs **0.70** for `all-MiniLM-L6-v2`. Override:

```bash
# strongest (needs custom model code):
export CODERAG_EMBED_MODEL=jinaai/jina-embeddings-v2-base-code
export CODERAG_EMBED_TRUST_REMOTE_CODE=1
# or tiny/fast general model:
export CODERAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

> Commands below use `python -m coderag.cli ...`; if you `pip install -e .`,
> the equivalent `coderag ...` console command works too.

---

## Quick start

```bash
# 1. Index a repo (clone one first, e.g. git clone https://github.com/tiangolo/fastapi)
python -m coderag.cli index /path/to/fastapi --out .coderag_index

# 2. Check coverage — did it index the right things? (spot-check for junk)
python -m coderag.cli stats --index .coderag_index

# 3. Ask a question (retrieval + Claude answer + faithfulness check)
python -m coderag.cli query "How does FastAPI coerce a path param to an int?"

# 4. Inspect the code graph around a symbol
python -m coderag.cli graph --symbol APIRouter.add_api_route --depth 1

# 5. Reindex only what changed after a git pull
python -m coderag.cli update /path/to/fastapi
#    or, precisely, between two commits:
python -m coderag.cli update /path/to/fastapi --git <old_sha> <new_sha>
```

### Retrieval-only / ablation flags on `query`

```bash
python -m coderag.cli query "..." --retrieve-only            # no Claude call
python -m coderag.cli query "..." --dense-only               # disable BM25
python -m coderag.cli query "..." --no-rerank                # skip cross-encoder
python -m coderag.cli query "..." --expand-graph             # add graph neighbors
```

---

## Evaluation

Build a golden set first (this is the ruler for everything): edit
[`data/golden_questions.jsonl`](data/golden_questions.jsonl) with 30–50 questions
for **your** repo, labeling `relevant_files` by actually reading the code. Include
a few out-of-scope questions (empty `relevant_files`) to test the abstain path and
mark ~10 as `"holdout": true`.

```bash
# Retrieval-only ablation table (no Claude key needed):
python -m coderag.cli eval --index .coderag_index --configs dense,hybrid,rerank,graph

# Full table incl. generation faithfulness + answer correctness (uses Claude):
python -m coderag.cli eval --generate
```

Real run on the bundled FastAPI golden set (30 questions; 18 answerable dev,
default code embedder, retrieval-only):

| Config | Recall@5 | MRR | NDCG@5 | P@5 | SymR@5 |
|---|---|---|---|---|---|
| dense  | 0.97 | 0.77 | 0.81 | 0.29 | 0.75 |
| hybrid | 0.92 | 0.87 | 0.87 | 0.33 | 0.83 |
| rerank | 0.97 | 0.82 | 0.86 | 0.41 | 0.77 |
| graph  | 0.97 | 0.82 | 0.86 | 0.28 | 0.77 |

What the harness reveals here: dense recall is near the ceiling with the code
embedder, so file-recall@5 can't separate the configs — but **hybrid wins MRR and
symbol recall** (BM25 fusion sharpens ranking), and **rerank wins precision**
(P@5). Swapping the embedder (`all-MiniLM-L6-v2` → code) is what moved dense
recall@5 from 0.83 → 0.97 in the first place. recall@k is reported with a bootstrap
CI; runs are saved to `eval_runs/<timestamp>/` (`summary.md` + `results.json`,
per-question rows). Full write-up with both embedders and per-component analysis:
[RESULTS.md](RESULTS.md).
When questions carry `relevant_symbols`, extra **symbol-granularity** columns
(`SymR@k`, `SymMRR`) appear automatically — a chunk hits a symbol if the qualified
names match, share a simple name, or one is a suffix of the other.

### Ablating chunking and context headers

`dense / hybrid / rerank / graph` run on a single index. The chunking-strategy
and context-header ablations change *indexing*, so build separate indexes:

```bash
# Baseline: window chunks, no header  →  eval it as "dense"
python -m coderag.cli index /path/to/repo --out .idx_baseline --window-chunk --no-context-header
python -m coderag.cli eval --index .idx_baseline --configs dense

# + code-aware chunking, no header
python -m coderag.cli index /path/to/repo --out .idx_nohdr --no-context-header

# + context headers (the default)
python -m coderag.cli index /path/to/repo --out .idx_full
```

Comparing `dense` across these three indexes gives the "+ code-aware chunking"
and "+ context headers" rows of the outline's table; the `hybrid/rerank/graph`
configs on `.idx_full` give the rest.

---

## Project layout

```
coderag/
  schema.py            §0  Chunk data model (id = file:line = citation)
  config.py                central settings + ANTHROPIC_API_KEY guard
  tokenization.py      §3.2 token_len + code-aware identifier tokenizer
  ingest/
    discovery.py       §1  .gitignore-aware file discovery + skip rules + manifest
    chunker.py         §2  tree-sitter AST chunking, context headers, fallbacks
    languages.py       §2.1 grammar-pluggable LanguageSpec registry (py/js/ts)
  graph/
    code_graph.py           imports/calls/containment graph + neighbor queries
  embed/embedder.py    §2.4 local sentence-transformers embeddings
  index/
    vector_store.py    §3.1 normalized-vector cosine store
    bm25_index.py      §3.2 BM25 over the same chunks
    store.py           §2.4 CodeIndex: build/save/load, manifest, graph records
  retrieve/
    retriever.py       §3   dense + BM25 + RRF + graph expansion
    rerank.py          §3.4 cross-encoder reranker
  generate/
    generator.py       §4   context assembly + grounded answering (Claude)
    prompts.py         §4.2 system/user prompts (the abstain instruction)
  verify/faithfulness.py §5 structural check + RAGAS-style LLM judge
  llm/                      provider layer: Anthropic + OpenAI-compatible adapters
  eval/
    metrics.py         §6.2 recall/precision/MRR/NDCG
    gen_metrics.py     §6.3 answer correctness + citation precision/recall
    bootstrap.py       §6.5 bootstrap confidence intervals
    run.py             §6.4 the runner and the table
  incremental.py       §7   content-hash + git-diff incremental reindex
  cli.py                    index / query / update / eval / stats / graph
data/golden_questions.jsonl §6.1 golden set schema + template
tests/                      pytest suite (torch-free; stub embedder)
```

## Tests

The suite covers the pure logic — tokenizer, RRF, metrics (file + symbol),
bootstrap, AST chunking (incl. JS), the code graph, index build/save/load,
incremental reindex, context assembly, and citation checks — with a stub embedder
so it needs no torch and no API key.

```bash
pip install -e '.[dev,langs]'   # pytest + JS/TS grammars
pytest
```

---

## How the code graph saves tokens

Without the graph, answering a cross-file question means either retrieving (and
paying for) many chunks or dumping whole files. With it:

1. Retrieval finds the best entry-point chunk(s).
2. `CodeGraph.neighbors()` returns the *connected* chunks (callees/callers/
   imports/methods) — resolved at index time, so it's a dictionary lookup, not a
   scan.
3. Generation includes a few of those neighbors (`--expand-graph`) **and** a
   compact one-line-per-edge structural map (`X calls Y (file:line)`), which costs
   a handful of tokens but tells the model how the pieces fit.

Inspect it directly with `coderag.cli graph --symbol <name>` or
`--file <path>`.

---

## Notes & limitations

- **AST chunking covers any tree-sitter language.** 18 mainstream languages have
  **precise specs** (exact node-type sets derived by parsing real samples, not
  guessed — verified by `tests/test_precise_languages.py`): Python, JS, TS, Go,
  Rust, Ruby, Java, C, C++, C#, PHP, Kotlin, Scala, Swift, Lua, Bash, Perl,
  Objective-C. Every other grammar uses a **generic** pattern-based classifier
  (nodes matched by type-name: `*function*`, `*class*/struct/impl/trait/...`,
  `*call*`, `*import*`). Name/body/callee extraction is shared robust logic, so a
  precise spec is just a list of node types — add one in
  `coderag/ingest/languages.py`. Install grammars with `pip install
  'coderag[langs]'` (~165 via `tree-sitter-language-pack`). Languages with no
  installed grammar, or any parse failure, fall back to line-window chunking so
  nothing is dropped.
- **Graph edges are name-resolved heuristically** (no full type inference). This
  is intentionally cheap; ambiguous names (including qualified-name collisions
  across files) are left unlinked rather than guessed, to keep the graph
  high-precision.
- **Embedding truncation:** small general models (e.g. all-MiniLM, 256 tokens)
  truncate large chunks — the context header goes first so signature/location
  survive. Use a longer-context code model via `CODERAG_EMBED_MODEL` if needed.
