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
numbered section there. New to RAG / embeddings / the eval methodology? Start with
[**docs/LEARN.md**](docs/LEARN.md) — a from-scratch walkthrough of every concept.

---

## Headline results

| Dimension | Result |
|---|---|
| Embedder lever (golden set, recall@5) | general `all-MiniLM` 0.83 → code-trained `st-codesearch` **0.97** (CIs disjoint) |
| External — CodeSearchNet (800 docstring→code) | recall@10 **0.985** · MRR 0.948 |
| External — HumanEval (164 problem→solution) | hybrid recall@10 **0.860** · MRR 0.543 |
| Scale — Django (521k LOC, ~40k chunks) | BM25 adds **+0.10** recall@5 (paired test significant) |
| Test suite | **129 passing** |

Everything is reproducible (`coderag eval`, `coderag bench …`) and reported with
**bootstrap confidence intervals + paired significance** — full detail in
[RESULTS.md](RESULTS.md).

## What's interesting (the honest findings)

The point isn't "it works" — it's **measuring what moves the needle and reporting
the negatives**:

1. **The embedder is the dominant lever** — a code-trained model lifted recall@5
   `0.83 → 0.97` (and `0.70 → 0.97` on direct lookups), bigger than every other
   component combined; confirmed externally on CodeSearchNet.
2. **A hybrid "win" that was a confound — caught and killed.** Auto-generated
   questions leaked function names into the query, unfairly helping BM25; paraphrasing
   the names out erased the advantage (recall@5 `0.922 → 0.861`).
3. **Components mostly don't separate** on a strong embedder + small repo (reported as
   "no separation," not a fake winner) — but **BM25 becomes decisive at scale**
   (Django +0.10) and on HumanEval's near-duplicate solutions (+0.05).
4. **Three features measured useless/harmful, then turned off:** HyDE (hurt the common
   case), self-repair (gain was regression-to-the-mean, proven with a control), and a
   **general-domain cross-encoder reranker** (hurt HumanEval recall@10 `0.86 → 0.63` —
   a web reranker doesn't transfer to code).
5. **The code graph helps only conditionally** — significant on Go (p=0.019), null on
   Python (p=1.000), even after improving its edges.

Lever hierarchy: **chunking > embedder > BM25 (at scale) > rerank/fusion > prompt >
graph (conditional) > judge.**

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
# strongest (needs custom model code + a compatible transformers — see note):
export CODERAG_EMBED_MODEL=jinaai/jina-embeddings-v2-base-code
export CODERAG_EMBED_TRUST_REMOTE_CODE=1
# or tiny/fast general model:
export CODERAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

> Note: Jina v2's custom model code needs an older `transformers` (it imports
> `find_pruneable_heads_and_indices`, removed in recent versions). On a current
> stack it errors on load — pin `transformers<4.40` in a separate env to try it.
> The override mechanism is provider-agnostic; this is a Jina-specific dep pin.

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

# 4. Interactive REPL — load the index once, ask many (with slash commands)
python -m coderag.cli chat

# 5. Inspect / visualize / edit the code graph
python -m coderag.cli graph --symbol APIRouter.add_api_route --depth 1
python -m coderag.cli graph-export --symbol APIRouter --format html --out graph.html  # static, editable
python -m coderag.cli graph-serve  --symbol APIRouter --port 8000                     # LIVE edits + reset
python -m coderag.cli graph-edit --add-edge caller_symbol callee_symbol --type calls
python -m coderag.cli graph-rebuild     # remake the graph from stored records (no re-embed)

# 6. Reindex only what changed after a git pull
python -m coderag.cli update /path/to/fastapi
#    or, precisely, between two commits:
python -m coderag.cli update /path/to/fastapi --git <old_sha> <new_sha>
```

**Interactive chat** (`chat`) loads the index once and runs a REPL: ask questions
(cited answers), or use `/retrieve <q>`, `/graph <symbol>`, `/sources`, `/stats`,
`/help`, `/quit`.

**Graph visualizer/editor.** `graph-export` renders a focused subgraph (or a
degree-capped overview) as a standalone **interactive HTML** page with a **node
search box** (matches highlight gold + dim the rest + zoom), or `dot`/`mermaid`.
**Layout** is auto-chosen: a fragmented **overview** defaults to **force** (ForceAtlas2
— spreads into clusters, auto-settles), while a **focused** view (`--symbol Foo
--depth 2`) is connected so it defaults to the layered **tree** (LR). Both are toggle
buttons in the page; override with `--layout force|hierarchical`. For
publication-quality layered output, `--format dot` → Graphviz is cleanest.

*Do edits persist?* The static `graph-export` page is **scratch** — edits aren't
saved; download the JSON and apply it with **`graph-import <file>`** (reconciles only
the viewed subgraph's edges). For **live persistence**, use **`graph-serve`**: a
localhost server where each add/delete writes `graph.json` immediately (survives
closing the page), with a **reset** button to revert. `graph-edit` does the same
non-interactively from the CLI.

### Retrieval-only / ablation flags on `query`

```bash
python -m coderag.cli query "..." --retrieve-only            # no Claude call
python -m coderag.cli query "..." --dense-only               # disable BM25
python -m coderag.cli query "..." --no-rerank                # skip cross-encoder
python -m coderag.cli query "..." --expand-graph             # graph neighbors → rerank pool
python -m coderag.cli query "..." --hyde                     # HyDE query expansion (needs a key)
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

Real run on the bundled FastAPI golden set (**100 questions**, 75 answerable dev,
default code embedder, retrieval-only; **recall@5 with bootstrap CIs**, questions
paraphrased identifier-free to avoid a lexical confound):

| Config | Recall@5 | MRR | P@5 (ceiling 0.32) |
|---|---|---|---|
| dense  | 0.900 [0.86–0.94] | 0.841 [0.78–0.90] | 0.279 |
| hybrid | 0.861 [0.80–0.92] | 0.802 [0.73–0.87] | 0.266 |
| rerank | 0.900 [0.85–0.94] | 0.833 [0.77–0.89] | 0.279 |
| graph  | 0.920 [0.88–0.96] | 0.833 [0.77–0.89] | 0.288 |

**What the data supports — stated to its actual confidence.** Two findings are
robust; the rest is "no separation detected," and the table says so.

1. **A phrasing confound, caught and killed (the headline).** The scaffolded
   questions originally *named their target symbols* (`How does X use Y?`); on that
   version hybrid "won" recall@5 (0.922). Paraphrasing them identifier-free
   (`scripts/scaffold_golden.py --paraphrase`) dropped hybrid to **0.861 — below
   dense** — because BM25 had been matching the names in the question, not
   retrieving better. This is robust: a within-design change with a clear mechanism.
2. **The embedder is the dominant lever.** general→code moved dense recall@5
   0.83 → 0.97 on direct-lookup questions — a gap wide enough to clear the CIs.

Everything **among dense / rerank / graph is within overlapping CIs** (recall@5
0.86–0.92), so the honest call is *no separation detected*, not a winner. P@5 reads
low (~0.28) but is **near its ceiling**: questions average 1.6 relevant files (max
3), so the most P@5 can be is ~0.32 — precision is bounded by label sparsity, not
poor retrieval. **The code graph's value is conditional — measured across three repos
with paired significance tests.** Cross-file recall@5, graph vs no-graph:
**[cobra](https://github.com/spf13/cobra) (Go, dense call graph) 0.824 → 0.941
(Δ +0.118, p=0.019 — significant)**; FastAPI (Python) 0.829 → 0.829 (p=1.000, null);
Django (Python, 40k chunks) +0.03 (noise). The graph helps on **typed languages with
dense call graphs** (Go) and is null on Python. It's an **opt-in, decide-per-repo**
knob — `eval --configs dense,graph`.

**Scale (Django, 521k LOC → 40k chunks):** indexes in **84 s** at **2.2 GB** peak
RAM; **47 ms/query** for dense+BM25 over 40k chunks (brute-force numpy holds — FAISS
only needed ~10× larger). And scale surfaced the real lesson: dense-only recall falls
to 0.667 on a big repo, and **BM25/hybrid recovers +0.10** — exact-identifier matching
is what scales. Lever order: **embedder → BM25 → graph (conditional)**. (Full
three-repo + scaling write-up: RESULTS §3a–3c.)

recall@k CIs are shown by default; runs save to `eval_runs/<timestamp>/`. Full
per-component write-up: [RESULTS.md](RESULTS.md).
With `--generate`, extra columns appear — **Faithfulness, Cite-P/R, Correctness,
and CtxTok** (assembled-context tokens, so the graph is judged on its real
cost/benefit, not recall). `SymR@k`/`SymMRR` appear when questions label
`relevant_symbols`.

### Generation quality + the abstain path

`eval --generate` measures the parts that differentiate this project — grounded,
cited answers and honest refusal. On the de-confounded 100-q set (Haiku 4.5 judge,
dense config): answer-correctness **0.747 [0.67–0.82]**, citation precision **0.84**,
faithfulness **0.79** (i.e. ~1 in 5 claims the judge didn't match to a cited source —
judge strictness vs. genuine ungrounding is worth probing). And an **honest
non-result worth keeping**: auto-expanding the code graph's neighbors showed *no
measured benefit* — cross-file correctness trended down (Δ **−0.10, 95% CI
[−0.23, +0.04]**, includes 0) at ~13% fewer tokens. No upside, a downward trend → it
stays **opt-in (`--expand-graph`), not default-on** (full numbers + mechanism in
[RESULTS.md](RESULTS.md)). The headline behavior to see live is **abstention** — ask
something the repo can't answer and it declines instead of confabulating:

```bash
python -m coderag.cli query "What is the default database connection pool size?" \
  --index .coderag_index_code
# → "the sources do not contain this" — and the structural check flags any
#   citation that doesn't map to a real source.
```

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
    languages.py       §2.1 LanguageSpec registry: 18 precise + generic fallback
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
pip install -e '.[dev,langs]'   # pytest + tree-sitter grammars (~165 languages)
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

## Token efficiency

Beyond the graph, generation spends tokens carefully. The context budget is
**always enforced** — no single source can exceed it (an oversized chunk, e.g. a
4k-line class summary, is trimmed to the query-relevant lines), which alone took
one FastAPI query from ~22k to ~6k tokens. On top of that, quality-preserving
savers run by default (toggle in [config.py](coderag/config.py)):

| Saver | Default | What it does |
|---|---|---|
| `drop_negative_rerank` | on | drops sources the cross-encoder scored irrelevant (<0), keeping ≥ `min_sources` |
| `dedup_sources` | on | drops content-identical chunks |
| `merge_adjacent_sources` | on | merges contiguous same-file spans into one source |
| `compact_source_code` | on | collapses blank lines / trailing whitespace (lossless) |
| `trim_sources` | **off** | proactively trim large chunks to query-relevant lines (can drop code) |
| `faithfulness_single_call` | on | one judge call (extract+verify) instead of two |
| `judge_source_tokens` | 300 | caps source text sent to the faithfulness judge |
| `faithfulness_skip_when_clean` | **off** | skip the LLM judge when the structural check is clean |

Measured ~24% further context reduction across sample FastAPI queries with
relevant sources preserved. The riskier levers (`trim_sources`,
`faithfulness_skip_when_clean`) are off by default; turn them on and re-run
`eval --generate` to confirm faithfulness/correctness hold for your repo.

---

## Use it from an agent (MCP server)

Expose the index to any MCP client (Claude Code, Cursor, Claude Desktop) so an
agent can search and ask about the codebase:

```bash
pip install 'coderag[mcp]'
CODERAG_INDEX_DIR=.coderag_index_code python -m coderag.mcp_server
```

Two tools: `search_code(query, k)` (key-free retrieval) and `answer_question(query)`
(grounded, cited answer — needs a provider/key). Client registration example:

```json
"coderag": {"command": "python", "args": ["-m", "coderag.mcp_server"],
            "env": {"CODERAG_INDEX_DIR": "/abs/path/.coderag_index_code"}}
```

**Auto-reindex on `git pull`.** Incremental reindex (§7) can run from a git hook —
symlink [scripts/git_post_merge_reindex.sh](scripts/git_post_merge_reindex.sh) into
the indexed repo's `.git/hooks/post-merge`; it re-embeds only the files that changed
between the old and new HEAD.

---

## Advanced options & extras

All opt-in via [config.py](coderag/config.py) (so the eval can ablate each):

| Option | What it does | Status on FastAPI |
|---|---|---|
| `use_hyde` / `--hyde` | embed a hypothetical snippet for dense search | **measured: hurts strong-retrieval repos** (FastAPI −0.05, p=0.014); off by default |
| `dense_weight` / `bm25_weight` | weight dense vs lexical before RRF | tunable |
| `use_mmr` | MMR diversity on the final chunks | reduces near-dups |
| `graph_expand_depth`, `graph_pagerank`, `graph_rerank_boost` | multi-hop traversal / personalized-PageRank / connectivity-boost graph wirings | **measured: no retrieval gain** (RESULTS §3) |
| `self_repair_threshold` | retry once with a stricter cite-or-drop prompt if faithfulness is low | **controlled test: no effect** — the lift was regression-to-the-mean (Δ vs re-roll −0.015, p=0.77); off by default |
| `coderag.retrieve.routing` | heuristic query classifier (lookup / howto / multihop) | cost/effort routing |

**External benchmark (CodeSearchNet).** Independent validation beyond the self-made
golden set: on **800 real docstring→code pairs**, the default embedder scores
**recall@10 0.985 / MRR 0.948** ([coderag/eval/codesearchnet.py](coderag/eval/codesearchnet.py)).
Reproduce with `pip install 'coderag[bench]'` then stream a sample (or pass any
CodeSearchNet-style `.jsonl` to `evaluate_codesearchnet(path)`).

**External benchmark (CodeRAG-Bench-style) — the *full* retriever, not just the
embedder.** CodeSearchNet scores the embedder alone; this runs dense + BM25 + RRF
over a shared corpus where each query has relevant document(s)
([coderag/eval/coderag_bench.py](coderag/eval/coderag_bench.py)):

```bash
python scripts/fetch_humaneval_bench.py                 # HumanEval -> data/humaneval_bench.jsonl
coderag bench data/humaneval_bench.jsonl --suite coderag --mode hybrid -k 10
```

On HumanEval (164 problems, retrieve each problem's canonical solution from the
pool of all 164): **hybrid recall@10 0.860 / MRR 0.543**, vs dense-only 0.811 /
0.527 — a genuinely harder task than CodeSearchNet (0.985), and a case where BM25
measurably helps (+0.049). Adding the default cross-encoder (`--mode rerank`) *hurts*
(recall@10 → 0.634): a web-search reranker doesn't transfer to code — a measured
reason it's off by default. The harness is format-tolerant and unit-tested on a
synthetic corpus; datasets aren't bundled (fetch via the script or pass any
matching `.jsonl`).

**Generalize to another repo/language.** The golden-set scaffolder is repo-agnostic:
```bash
coderag index /path/to/other-repo --out .idx_other --install-grammars
python scripts/scaffold_golden.py --index .idx_other --paraphrase > data/other.jsonl
coderag eval --index .idx_other --golden data/other.jsonl
```

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
  'coderag[langs]'` (~165 via `tree-sitter-language-pack`), or let indexing fetch
  them on demand: `coderag index <repo> --install-grammars` detects the languages
  present and installs what's missing (also via `CODERAG_AUTO_INSTALL_GRAMMARS=1`).
  Languages with no available grammar, or any parse failure, fall back to
  line-window chunking so nothing is dropped.
- **Graph edges are name-resolved heuristically** (no full type inference), but with
  principled disambiguation for ambiguous names, in precision order: unique match →
  the **caller's own file** (language scoping) → a **file the caller imports the name
  from** (import-aware) → otherwise left unlinked rather than guessed. Cheap, and
  high-precision by construction. (Re-index to apply resolver changes to an existing
  index.)
- **Embedding truncation:** small general models (e.g. all-MiniLM, 256 tokens)
  truncate large chunks — the context header goes first so signature/location
  survive. Use a longer-context code model via `CODERAG_EMBED_MODEL` if needed.

---

## Productionization roadmap

This is a **rigorously tested reference implementation**, not a deployed service —
and the difference is deliberate. It is production-grade on *correctness* (137 tests
+ CI, an eval harness with confidence intervals, secret hygiene) but intentionally
leaves the *serving/ops* layer out of scope. If you were to run it as a service, here
is the honest gap list, in priority order. (The first item is **done** — included as
a demonstrated slice.)

1. **Resilience on external calls — ✅ done.** LLM API calls get exponential backoff
   from the provider SDKs; `timeout` + `max_retries` are now first-class and
   configurable (`CODERAG_LLM_TIMEOUT`, `CODERAG_LLM_MAX_RETRIES`) instead of relying
   on undocumented defaults. Local model loading (HF download — no SDK retry) is
   wrapped in `with_retry` (backoff + jitter, transient-only). See
   [coderag/resilience.py](coderag/resilience.py).
2. **Observability.** A stdlib logger is wired (`CODERAG_LOG_LEVEL`); a real
   deployment would add structured logs, metrics (latency/recall/cost), and tracing
   rather than the `print()` diagnostics used for CLI UX.
3. **A real API surface.** Today it's CLI + an MCP server + a localhost graph viz.
   Production needs an ASGI service (FastAPI/uvicorn) with auth, rate limiting,
   request validation, and `/health`/`/ready` endpoints.
4. **Scalable vector store.** [vector_store.py](coderag/index/vector_store.py) is an
   in-memory brute-force matmul — O(N) per query, all in RAM. Great to ~10⁵ chunks
   (Django: 40k, 47 ms/query); beyond that, swap in FAISS/HNSW/pgvector behind the
   same `search()` interface (the code is structured for it).
5. **Concurrency & state.** Single-process, in-memory index; `graph-serve` writes
   `graph.json` live with no locking. A service needs concurrent-read safety,
   transactional index updates, and an index-migration path (`INDEX_VERSION` exists
   but has no upgrade logic yet).
6. **Deploy & cost.** Dockerfile + healthcheck + deploy manifests; pinned deps in the
   image (the runtime SDK auto-install is a dev convenience, an anti-pattern in prod);
   secrets via a manager rather than `.env`; LLM spend quotas + a response cache.

The point of listing this explicitly: knowing *what's missing and in what order* is
the production-readiness skill — more so than half-building the serving layer here.
