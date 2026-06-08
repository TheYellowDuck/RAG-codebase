# Learn this project from scratch

A from-zero explanation of the whole system — no prior background assumed. Every
term is defined as it appears, concepts build in order, and each one is tied to the
code that uses it. Read it in passes. By the end you should be able to *derive* the
design decisions, not just describe them.

> Companion docs: [README.md](../README.md) (how to use it) and
> [RESULTS.md](../RESULTS.md) (the measured findings).

---

## 0. The problem

You have a big codebase (thousands of files). You want to ask a plain-English
question — *"how does FastAPI turn a bad request into a 422 error?"* — and get a
correct answer with links to the exact lines.

Two naive approaches fail:

- **Grep / Ctrl-F** finds only exact words. Ask "how does it validate input" and you
  won't find the function named `request_params_to_args`.
- **Paste the whole repo into an AI** is impossible: an AI can only read a limited
  amount of text per request (its *context window*), and a repo is far too big.

So the whole game is: **find the few pieces of code that matter, and show only those
to the AI.** The field's name for that is **RAG**.

---

## 1. The basics

**LLM (Large Language Model)** — e.g. Claude, GPT. An extremely well-read assistant
that continues text sensibly (advanced autocomplete trained on huge text corpora).
Two facts: it only "sees" what's in the prompt you give it (its training does *not*
include your private repo), and it has a maximum input size per request.

**Token** — LLMs read *tokens*, not characters or words: chunks of text ≈ ¾ of a word.
You pay per token and the context window is counted in tokens — which is *why* the
whole repo can't fit.

**RAG (Retrieval-Augmented Generation)** — the strategy:

1. **Retrieve** the few relevant pieces of your data,
2. **Augment** the prompt with them,
3. **Generate** the answer from them.

It's an **open-book exam**: don't memorize the textbook — flip to the 3 relevant
pages, then answer. This project is a code-specialized open-book exam system.

The remaining parts: cut the book into pages (chunking), find the right pages
(retrieval), answer from them (generation), check the answer (verification), and
**prove it works** (evaluation — the heart).

---

## 2. Retrieval — finding the relevant code

Two complementary ways to match a question to code; the project uses both.

### 2a. Meaning-based search — embeddings & vectors

The trick: **turn text into numbers that capture meaning**, so "compute average"
lands near `def mean(xs): return sum(xs)/len(xs)` even with no shared words.

- A **vector** is a list of numbers, e.g. `[0.12, -0.7, 0.4, …]` — think *coordinates
  of a point in space*. Two numbers = a point on a map (x, y); ours have ~768 numbers
  = a point in 768-dimensional space. Can't picture it, but the math is the same.
- An **embedding model** is a neural net that reads text and outputs such a vector,
  trained so **similar meaning → nearby coordinates.** This "map of meaning" is the
  *embedding space*.
- To search: embed the question, find the chunks whose vectors are **closest** by
  **cosine similarity** — the *angle* between the two vectors-as-arrows. Same
  direction → ~1 (similar); perpendicular → 0; opposite → −1. We normalize vectors to
  length 1 so cosine = a fast dot product.

Code: [coderag/embed/embedder.py](../coderag/embed/embedder.py) produces vectors;
[coderag/index/vector_store.py](../coderag/index/vector_store.py) stores them all in
one matrix and finds the closest with a single matrix multiply. This is **dense
retrieval**.

> **Biggest finding of the project lives here:** the *choice of embedding model*
> dominates. A general-text model scored recall@5 = 0.70; a **code-trained** model
> scored 0.97 on the same test — bigger than every other component combined.

### 2b. Keyword search — BM25

Embeddings *blur* exact names (ask about `HTTPException`, drift to "error handling").
Developers query exact identifiers, so we also need literal matching.

**BM25** is the classic keyword-ranking algorithm: count the query's words in each
document, weighting **rare words more** (a chunk containing `RequestValidationError`
is a strong signal; "the" is useless). This is **lexical / sparse** retrieval.

Code twist ([coderag/tokenization.py](../coderag/tokenization.py)): a **code-aware
tokenizer** splits `get_current_user` / `getCurrentUser` into `get / current / user`,
so "get current user" matches both naming styles. That's *why* BM25 works on code.
([coderag/index/bm25_index.py](../coderag/index/bm25_index.py))

### 2c. Combining both — RRF

Two ranked lists (meaning + keywords) with incomparable scores (cosine 0.8 vs BM25
14.3). **Reciprocal Rank Fusion** combines **ranks, not scores**: a chunk's fused
score = Σ over lists of `1 / (k + rank)`, k=60. Being near the top of *either* helps;
of *both* helps a lot. No score calibration needed. (`rrf` in
[coderag/retrieve/retriever.py](../coderag/retrieve/retriever.py).) Dense + BM25 =
**hybrid retrieval**.

### 2d. A precise second pass — reranking

Fusion gives ~30 rough candidates. A **cross-encoder reranker**
([coderag/retrieve/rerank.py](../coderag/retrieve/rerank.py)) reads each
`(question, chunk)` *together* and scores relevance — slow but accurate, so it only
runs on the shortlist, keeping the best ~6.

- **Bi-encoder** (embedder): encodes question and chunk *separately*; fast,
  approximate; good for scanning thousands.
- **Cross-encoder** (reranker): reads them *together*; slow, precise; good for a
  shortlist.

Pipeline: **dense + BM25 → RRF → cross-encoder rerank → top few.** Every stage is a
toggle so each one's value can be measured (§7).

---

## 3. Chunking — cutting code into "pages"

Retrieval works on **chunks**. *How you cut* matters as much as how you search.

**Naive (bad):** split every N lines → slices a function mid-body → retrieves as
noise.

**This project — AST-boundary chunking:**

- **AST (Abstract Syntax Tree)** = the grammatical structure of code as a tree (this
  is a *function definition* with *parameters* and a *body* containing *calls*…).
- **tree-sitter** parses source into its AST for ~165 languages.
- Cut at meaningful boundaries: each function/method = its own chunk; a class = a
  *summary* chunk (signature + its method signatures); oversized functions are
  windowed but each window carries the function signature.

([coderag/ingest/chunker.py](../coderag/ingest/chunker.py),
[coderag/ingest/languages.py](../coderag/ingest/languages.py))

**Context headers** ([coderag/schema.py](../coderag/schema.py)): before embedding,
prepend `File / Class / signature / docstring`. You *embed* `header + code` but
*display* `code`. The header injects location + signature into the vector — exactly
what questions key off. Measurable win.

**Chunk id = location:** `sha1("routing.py:1358-1450")`. Because the id *is* the
citation, a retrieved chunk already knows where it lives — which makes citations,
dedup, the graph, and incremental updates all simple.

Coverage: 18 languages have hand-tuned *precise* specs; everything else uses a
*generic* pattern-matcher; unparseable files fall back to line-windows (nothing is
dropped).

---

## 4. The code graph

Functions **call** functions, files **import** files, classes **contain** methods —
that's a graph.

- **Graph** = nodes (chunks) + edges (`calls` / `imports` / `contains`).
  ([coderag/graph/code_graph.py](../coderag/graph/code_graph.py))
- **Why:** for a cross-file question, find the entry point by retrieval, then *follow
  edges* to connected chunks instead of dumping whole files. Built once at index time
  → following edges is a dict lookup.
- Edges resolve **conservatively** (ambiguous name → prefer caller's own file, then a
  file it imports from, else leave unlinked). Better to miss an edge than invent one.

**Honest result (§8):** measured, the graph's *retrieval* value is **conditional** —
helped on a Go repo (dense graph), null on Python (retrieval already found
everything). So expansion is **off by default**. A visualizer/editor ships with it
(`graph-export` / `graph-serve`).

---

## 5. Generation — answering from the chunks

([coderag/generate/generator.py](../coderag/generate/generator.py),
[coderag/generate/prompts.py](../coderag/generate/prompts.py))

1. **Assemble context:** number the chunks `[1] file:line / <code>`, drop
   duplicates/overlaps, stay under a token budget (a huge chunk gets trimmed to the
   relevant lines).
2. **Prompt:** *answer using ONLY these sources, cite every claim with `[n]`, and if
   the sources don't cover it, say so — don't guess.*
3. **Abstention** (admitting "I don't know") is what makes it trustworthy: a
   confident wrong answer is worse than "not covered."

The `[n]` citations let us *mechanically* check the answer (§6).

**Provider-agnostic** ([coderag/llm/](../coderag/llm/base.py)): a tiny interface with
adapters for Anthropic (Claude) and any OpenAI-compatible endpoint (OpenAI,
OpenRouter, local Ollama…), chosen from environment variables at runtime.

---

## 6. Verification — checking the answer

LLMs can "hallucinate" (make things up). Two layers
([coderag/verify/faithfulness.py](../coderag/verify/faithfulness.py)):

1. **Structural (free, no AI):** confirm every `[n]` points to a real source (catches
   a fabricated `[7]`); flag claim-sentences with no citation.
2. **Faithfulness (LLM-as-judge):** split the answer into atomic claims; for each,
   check the cited source actually supports it. `faithfulness = supported / total`
   (abstentions excluded). Standard "RAGAS-style" technique.

Using an LLM to grade an LLM is **LLM-as-judge** — imperfect (the judge is noisy,
§8), but with the free structural check it's a real safety net.

---

## 7. Evaluation — proving it works (the heart)

This is what separates "I built a RAG demo" from "I built it and can prove how good
it is."

### The ruler — a golden set
[data/golden_questions.jsonl](../data/golden_questions.jsonl): questions with the
correct answer locations (`relevant_files`) written down — the answer key. ~45
hand-verified + the rest scaffolded from the graph = 100. Some are out-of-scope (test
abstention); some are held out (not seen while tuning, to avoid self-deception).

### The metrics
For a question whose answer is in `routing.py`:

- **Recall@k** — did the right file make the top k? *The headline* — if retrieval
  misses it, nothing downstream recovers.
- **Precision@k** — of the k returned, what fraction were relevant? With ~1.6
  relevant files/question, precision@5 can't exceed ~0.32 — so a "low" 0.28 is near
  the ceiling. (Knowing your metric's ceiling = understanding it.)
- **MRR** — how high the *first* correct result ranked (#1 → 1.0, #2 → 0.5…).
- **NDCG** — rewards several relevant results near the top, not just the first.

Generation also measures **faithfulness**, **answer correctness** (judge vs your
reference), and **citation precision/recall**.

### Real, or luck?
On ~75 questions a 0.90-vs-0.88 gap might be noise. Two tools
([coderag/eval/bootstrap.py](../coderag/eval/bootstrap.py)):

- **Bootstrap confidence interval (CI):** resample the questions many times, recompute
  the score, report the middle-95% range. "0.90 [0.85–0.94]" = if you'd asked a
  slightly different set, you'd plausibly see 0.85–0.94. Overlapping ranges ⇒ **no
  separation detected** — you may *not* crown a winner.
- **Paired significance test + p-value:** compare two configs on the *same*
  questions; the **p-value** is the chance of a difference this big purely by luck.
  p < 0.05 = real effect. (Used to confirm the Go graph win, p=0.019, and the Python
  graph null, p=1.000.)

The runner ([coderag/eval/run.py](../coderag/eval/run.py)) prints an ablation table —
the same index as `dense / hybrid / rerank / graph`, each with CIs — so every
component's contribution is *attributable*.

---

## 8. What the evaluation found (plain words)

Several findings are *negative* — the most credible kind:

1. **The embedder is the biggest lever** (0.83 → 0.97; confirmed externally on
   CodeSearchNet: recall@10 = 0.985).
2. **A fake "win," caught and killed:** hybrid looked better until you noticed the
   auto-generated questions contained the function names (unfair help to BM25). You
   paraphrased the names out, re-ran, and the advantage vanished. *Distrusting your
   own favorable result* is the most impressive thing here.
3. **Configs mostly don't separate** on a strong embedder + small repo — reported as
   "no separation," not a fake winner.
4. **The graph is conditional** — significant on Go, null on Python. You even improved
   its edges to test the "too sparse" objection, re-indexed, and it *still* didn't
   help on Python.
5. **At scale, BM25 becomes the lever:** on Django (40k chunks), dense-only recall
   fell to 0.67; BM25 recovered +0.10 (significant). Indexed in 84 s, 47 ms/query —
   it scales.
6. **Two features built, measured useless, turned off:** HyDE (hurt the common case)
   and self-repair (its gain was regression-to-the-mean — proven with a control).

**The lever hierarchy** (the most ownable takeaway):
**chunking > embedder > BM25 (at scale) > rerank/fusion > prompt > graph (conditional)
> judge.** Fixing something low can't repair a problem higher up. That ordering is the
lens for reasoning about *any* RAG system.

---

## 9. The codebase, mapped to the concepts

```
coderag/
  schema.py            the Chunk (id = file:line = citation)              §3
  tokenization.py      token counting + code-aware tokenizer             §2b
  ingest/
    discovery.py       find/curate which files to index                  §3
    chunker.py         AST-boundary chunking + context headers           §3
    languages.py       18 precise specs + generic fallback               §3
  graph/code_graph.py  calls/imports/contains graph (+ viz/server)       §4
  embed/embedder.py    text -> vectors (the dominant lever)              §2a
  index/
    vector_store.py    dense (cosine) search                             §2a
    bm25_index.py      keyword search                                    §2b
    store.py           ties chunks+vectors+bm25+graph; save/load
  retrieve/
    retriever.py       dense + BM25 + RRF + (graph) + HyDE/MMR           §2c
    rerank.py          cross-encoder reranker                           §2d
  generate/            assemble context, prompt, cite, abstain           §5
  verify/faithfulness  structural check + LLM judge                      §6
  llm/                 provider-agnostic Claude/OpenAI adapters          §5
  eval/                metrics, bootstrap, significance, scaffold, run   §7
  incremental.py       reindex only changed files
  cli.py               index / query / chat / eval / graph-* commands
```

---

## 10. How to own it

**Read in pipeline order:** `schema → chunker → embedder → vector_store/bm25 →
retriever → generator → faithfulness → eval/run`. Then run `coderag eval` and toggle
one component at a time — watching the table move is where it clicks.

**Derive, don't recite** (what interviewers probe):

- *Why embeddings AND BM25?* Embeddings capture meaning but blur exact names; BM25
  nails names but misses paraphrases — together they cover both.
- *Why RRF instead of adding scores?* Cosine and BM25 scores are on different,
  unstable scales; rank-fusion needs no calibration.
- *Why AST chunks?* A function is the unit of meaning; half a function is noise.
- *Why is recall@5 the headline?* If the answer file isn't retrieved, nothing
  downstream can recover.
- *Why confidence intervals?* On ~75 questions a 0.02 gap is likely noise; CIs make
  that visible so you don't overclaim.

**The two-minute story:** *"Open-book Q&A over a codebase. I chunk on the AST so
functions are the units, embed them with a code-trained model, and retrieve with a
hybrid of meaning-search and keyword-search fused by reciprocal rank — because
developers query exact identifiers that embeddings blur. Answers cite every claim and
abstain when unsupported, checked by a structural pass plus an LLM faithfulness judge.
The centerpiece is an evaluation harness with a hand-labeled golden set and bootstrap
+ paired-significance tests — and the findings are honest: the embedder is the
dominant lever (confirmed on CodeSearchNet), a hybrid 'win' turned out to be a
phrasing artifact I debugged out, the code graph helps only conditionally (significant
on Go, null on Python), BM25 becomes critical at half-a-million-line scale, and two
features I built I measured to be useless and turned off."*

The arc — build it, then *distrust and falsify your own results* — is the rare part.
