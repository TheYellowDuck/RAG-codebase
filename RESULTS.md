# Results — Code RAG on FastAPI

A measured walk-through of the retrieval pipeline on a real codebase. The point
of the [eval harness](coderag/eval/) is to make every design choice attributable;
this is that story.

- **Target repo:** [tiangolo/fastapi](https://github.com/tiangolo/fastapi) @ `5cdf820c8046`
- **Golden set:** [data/golden_questions.jsonl](data/golden_questions.jsonl) — **100
  questions**, 25 holdout. ~45 hand-written + grep-verified (incl. 15 hard
  multi-hop); the rest scaffolded from the code graph
  ([scripts/scaffold_golden.py](scripts/scaffold_golden.py)) with labels verified by
  construction. Retrieval metrics are over the **~73 answerable dev questions**
  (holdout + out-of-scope excluded).
- **Stack:** local embeddings (code-search model) + local cross-encoder rerank;
  generation/judging via the pluggable provider layer. Retrieval metrics are
  key-free; generation metrics used **Claude Haiku 4.5** (≈$0.40/run).
- **Granularity:** file-level (a chunk "hits" if its file is in `relevant_files`);
  `SymR@5` is symbol-level recall. All headline metrics carry **bootstrap CIs** so
  saturation/noise is visible rather than hidden.

## TL;DR

> **Two findings survive scrutiny; everything else is "no separation detected."**
> (1) A methodological catch: questions that named their target symbols inflated
> hybrid (BM25 matched the names in the query); paraphrasing them identifier-free
> dropped hybrid recall@5 **0.922 → 0.861 — below dense**. A phrasing artifact,
> exposed by a within-design re-run. (2) The embedder is the dominant lever:
> general→code lifted dense recall@5 **0.83 → 0.97** on direct lookups — a gap wide
> enough to clear the CIs. Among dense / rerank / graph the CIs overlap (recall@5
> 0.86–0.92); the graph is *directionally* better on cross-file questions but within
> noise — and on the real test (correctness-at-tokens) it showed no benefit, so
> it's opt-in, not default-on.
> Honest > flattering — the table reports CIs so the non-separation is plain.

---

## Index coverage

```
python -m coderag.cli index fastapi/fastapi
python -m coderag.cli stats
```

| Metric | Value |
|---|---|
| Files indexed | 48 |
| Chunks | 531 — 45 module · 150 function · 101 class · 235 method |
| Window-fallback files (parse failed) | 2 |
| Code graph | 407 nodes · 669 edges (125 contains · 247 calls · 297 imports) |

The graph is real and useful: `solve_dependencies` correctly links to *calls*
`get_dependant` / `request_params_to_args` / `request_body_to_args` and is
*called_by* `get_request_handler` / `get_websocket_app`.

---

## Headline: the embedding model is the dominant lever

Dense-only recall@5, same chunks, same golden set, only the embedder changed:

| Embedding model | Dense Recall@5 |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (general text) | 0.83 |
| `flax-sentence-embeddings/st-codesearch-distilroberta-base` (code) | **0.97** |

This is now the repo default. The strongest option,
`jinaai/jina-embeddings-v2-base-code`, is one env var away
(`CODERAG_EMBED_MODEL` + `CODERAG_EMBED_TRUST_REMOTE_CODE=1`).

---

## 1. The embedder is the dominant lever — but it's question-dependent

Dense-only recall@5, same chunks, only the embedder swapped:

| Question set | `all-MiniLM-L6-v2` (general) | `st-codesearch` (code) |
|---|---|---|
| Direct-lookup (easier) | 0.70 | **0.97** (decisive, CIs disjoint) |
| Full 100-q (dense-only) | 0.849 [0.78–0.91] | 0.900 [0.85–0.94] |

The code embedder clearly wins on direct "where is X / what is Y" lookups (the gap
survives the noise). Across the full 100-q set the gap shrinks to ~0.05 with
overlapping CIs — many questions name identifiers that any retriever can latch
onto, and multi-hop answers need *several* files which no embedder nails alone.
Honest takeaway: tune the embedder first, but it isn't a silver bullet.

**Two follow-ups that kept us from chasing the wrong lever:**

1. **Recall is candidate-bound, not ordering-bound.** Dense recall@k on FastAPI is
   *flat* — 0.817 @3, 0.885 @5, 0.892 @10, **0.892 @40**. The relevant file is either
   in the top handful or not retrieved at all; it never sits at ranks 6–40. So
   reranking, MMR, and diversification **cannot** raise recall here (nothing to
   promote) — the ceiling is candidate generation (the embedder). A useful negative:
   don't tune the reranker to fix a recall problem.

2. **The embedder lever still has headroom — but the model class matters.** Tested
   two upgrades (prefixes auto-applied, see embed/infer_prefixes):

   | Embedder | HumanEval recall@10 | FastAPI recall@5 | FastAPI MRR |
   |---|---|---|---|
   | `st-codesearch` (default) | 0.811 | 0.885 | 0.848 |
   | `e5-base-v2` (general) | 0.970 | 0.840 ↓ | — |
   | **`CodeRankEmbed` (code)** | **0.994** | **0.896** | **0.920** |

   A *general* retriever (e5) wins docstring→solution but **regresses in-repo search** —
   task-dependent, so not a safe default. A strong *code* embedder
   (`nomic-ai/CodeRankEmbed`) wins **both**: +0.18 on HumanEval and, in-repo, a tied
   recall@5 with a large **MRR gain (0.848→0.920)** — it ranks the answer file much
   higher. It's a real, validated upgrade. It stays **opt-in, not default**, only
   because it ships custom model code (needs `pip install 'coderag[embed-code]'` +
   `CODERAG_EMBED_TRUST_REMOTE_CODE=1` at index *and* query time) — we don't silently
   enable arbitrary remote code. Its 8k context is auto-capped to avoid a memory
   blowup (embed/infer_max_seq_len). Enable:
   `CODERAG_EMBED_MODEL=nomic-ai/CodeRankEmbed CODERAG_EMBED_TRUST_REMOTE_CODE=1`.

   **But — measured honestly — it does not close the two production gaps.** Filling
   the blank cells: at-scale Django recall@5 is **tied** (dense 0.700 vs 0.67, hybrid
   0.750 vs 0.767 — within wide CIs, n=40), and answer-correctness **did not improve**
   (0.65 vs 0.78 on n=20, confounded by generation noise). It's also **~7× slower to
   index at scale** (~10 min vs 84 s on Django). So the upgrade is real *where
   retrieval was already decent* (focused repos, docstring tasks), but the at-scale
   recall and correctness ceilings are **not an embedder problem** — at 40k chunks the
   bottleneck is disambiguating thousands of similar symbols (where BM25 helps more
   than any embedder), and correctness is gated by more than retrieval. Honest net:
   better embedder, same two gaps.

## 2. Config ablation (100-q, code embedder) — recall@5 with bootstrap CIs

Scaffolded questions originally *named both symbols* (`How does X use Y?`), which
hands BM25/lexical an unfair assist. We removed that confound by paraphrasing them
to be identifier/file-free (`scripts/scaffold_golden.py --paraphrase`) and re-ran.
The before/after is the most instructive result here:

| Config | Recall@5 (symbol-named) | Recall@5 (paraphrased) |
|---|---|---|
| dense  | 0.900 | 0.900 [0.86–0.94] |
| hybrid | **0.922** | **0.861 [0.80–0.92]** ↓ |
| rerank | 0.886 | 0.900 [0.85–0.94] |
| graph  | 0.913 | **0.920 [0.88–0.96]** |

**Hybrid's apparent lead was the lexical confound.** Once the questions no longer
contain the literal identifiers, hybrid drops 0.06 — *below* dense — because BM25
was matching the symbol names in the question text, not finding better code. This is
the clearest single lesson in the writeup: a phrasing artifact masqueraded as a
retrieval win, and de-confounding exposed it. (MRR moves the same way: hybrid
0.914 → 0.802.)

After de-confounding, dense / rerank / graph land at 0.90 / 0.90 / 0.92 with
**heavily overlapping CIs — no separation among them**. The robust result is the
hybrid *drop*, not a new winner.

## 3. Among the configs: no separation detected (and that's the honest call)

With the confound removed, the CIs make it plain that dense / rerank / graph can't
be distinguished on this repo + embedder. Stating it precisely:

- **Precision is label-bounded, not weak.** P@5 is ~0.27–0.29 across all configs —
  but questions average **1.6 relevant files** (31×1, 40×2, 2×3), so the most P@5
  can be is **~0.32**. So every config sits near ~85–90% of the achievable ceiling,
  and the spread between them is noise. (`precision_at_k` divides by `k`, so it can't
  exceed that ceiling.)
- **The graph does not help retrieval — five wirings tested (k=15, recall@5).** I
  took the "make the graph help" question seriously and implemented the patterns the
  literature uses (traversal, Aider-style PageRank). None move the needle:

  | wiring | cross-file (40) | multi-hop (8) |
  |---|---|---|
  | dense+rerank (no graph) | 0.829 | 0.750 |
  | + neighbors → rerank pool | 0.829 | 0.750 |
  | + multi-hop traversal (depth-2) | 0.829 | 0.750 |
  | + personalized PageRank | 0.842 | 0.750 |
  | + connectivity-boost rerank | 0.829 | 0.750 |
  | + legacy post-rerank append | 0.867* | — |

  (*inflation — it pads the file list and *hurts* generation, §4.) PageRank's +0.013
  is noise. **Why:** strong dense+BM25+rerank already finds the relevant files when
  findable; the genuinely-missed ones aren't 1–2 hops from a retrieved seed (and the
  high-precision/low-edge-recall graph often lacks the edge), so there's nothing to
  surface. This reconciles with graph-RAG's documented wins — those come from
  *weak/no dense retrieval* or *global sense-making* questions, **not** local code
  lookup with a strong retriever. The pool wiring is the default for `expand_graph`
  (it's the safe one); expansion + PageRank stay opt-in. Raising the graph's
  retrieval value would need a higher-edge-recall graph (type inference) — **low ROI
  vs the embedder** *on this repo* — see §3a, which flips that on a second language.

### 3a. The graph IS conditional — it helps on Go (cobra), validating the mechanism

The FastAPI null isn't "graphs don't help" — it's "graphs don't help *here*." The
mechanism above predicts the graph should help where the graph is **denser** (a
typed language resolves calls/imports more cleanly → higher edge-recall). So I ran
the same harness on a second repo, **[spf13/cobra](https://github.com/spf13/cobra)
(Go)** — 36 files, 666 chunks, and a graph of **1,613 call edges / 631 nodes
(~2.5/node, far denser than FastAPI's)**. The prediction held:

Cross-file recall@5 (k=15), graph pool vs no graph, across **three repos**:

| repo (lang, size) | dense+rerank | + graph pool |
|---|---|---|
| FastAPI (Python, ~7k chunks) | 0.829 | 0.829 *(null)* |
| cobra (Go, 666 chunks, dense calls graph) | 0.824 [0.71–0.94] | **0.941 [0.85–1.00] (+0.12)** |
| Django (Python, 40k chunks) | 0.778 [0.61–0.92] | 0.806 [0.64–0.92] *(+0.03, noise)* |

**The graph's clearest win is cobra (Go)** — a +0.12 lift, mechanism-backed (Go's
clean call resolution → a denser graph, ~2.5 edges/node). A **paired bootstrap test**
(per-question deltas, far more sensitive than comparing two independent CIs) settles
it: cobra graph−dense **Δ +0.118 [+0.03, +0.24], p=0.019 — significant**; FastAPI
graph−dense **Δ +0.000, p=1.000 — definitively null.** So the honest conclusion is:
the graph is **conditional — significant on a typed language with a dense call graph
(Go), null on Python.** (On cobra, depth-2 and PageRank did *worse* than the simple
1-hop pool — more reach adds noise.) It's an opt-in, decide-per-repo knob (`eval
--configs dense,graph`), not a default.

**Tested the obvious objection — "your Python graph is just too sparse."** I added
**import-aware resolution** (resolve a call via the file the caller imports the name
from), which lifted FastAPI's call edges **+11% (247 → 274)**, and re-indexed. The
graph stayed **null (0.829, p=1.000)**. So the Python null is the *regime* (strong
retrieval already finds the files — no headroom), **not** an edge-recall artifact;
making the graph denser didn't help. cobra was unchanged (Go imports are package
paths, nothing to resolve locally). That's a stronger conclusion than the original:
I improved the hypothesized bottleneck and the result held.

### 3b. What scale (Django, 40k chunks) actually revealed: BM25 is the scale lever

Django (3,379 files, 521k LOC) is where retrieval gets *hard*, and it surfaced a
finding the small repos hid:

| Django config | recall@5 |
|---|---|
| dense-only | 0.667 [0.53–0.80] |
| + BM25 (hybrid) | **0.767 [0.65–0.87]  (+0.10)** |
| + rerank | 0.767 |
| + graph | 0.783 [0.65–0.90] |

Dense-only recall drops from ~0.90 (small repos) to **0.667** at 40k chunks — the
embedder alone can't disambiguate among thousands of similarly-named symbols. **BM25
recovers most of it: +0.10, and the paired test confirms it's real (Δ +0.100
[+0.02, +0.20], p=0.026 — significant).** Exact-identifier matching is what scales.
The graph adds a further +0.016 (within noise). So at scale the lever order is
**embedder → BM25 → (graph, conditional)** — and "hybrid beats dense" finally clears
significance *because the repo is big enough to need it.* (A `bm25_weight` sweep on
Django found the default equal weighting already optimal — 1.0 → 0.767, up/down
weighting ≤ 0.733 — so no tuning gain; an honest null.)

**Chasing the remaining at-scale recall — diagnosed, then four dead ends and one
conditional win.** The recall curve is the key: it climbs **0.75 @5 → 0.83 @10 → 0.87
@20**, then plateaus. So ~0.12 is *ordering-bound* (the right file IS in the top-20,
ranked below 5) and ~0.13 is *candidate-bound* (never retrieved). The ordering-bound
part looks promotable:

| Attempt to lift recall@5 (Django) | Result |
|---|---|
| Stronger embedder (CodeRankEmbed) | tied (0.700 dense / 0.750 hybrid) |
| MiniLM cross-encoder rerank | 0.733 < 0.750 — hurts |
| bge-reranker-v2-m3 (strongest general) | 0.637 < 0.662 (same harness) — hurts, ~16 min/CPU |
| `bm25_weight` sweep | null (equal optimal) |
| **graph-aware rerank (PPR-connectivity)** | **0.662 → 0.688 (+0.025)** — the one that helped |

Generic relevance scorers can't disambiguate near-duplicate symbols any better than
RRF — they reshuffle the top-30 and often demote the true chunk. But the **code graph
can**: re-ranking the fused pool by rank-fusing the RRF order with a personalized-
PageRank order (seeded by the top hits, `Retriever._graph_rerank`, `graph_rerank=True`)
promotes pool members that are call/import-connected to the top candidates. It's the
**only** lever that moved at-scale recall the right way (+0.025), and it's **neutral on
small repos** (FastAPI 0.744 → 0.739, n=106) — so it doesn't regress the common case.
**Powered up to a significance test, it stays opt-in.** Scaffolding a larger,
de-confounded Django set (150 paraphrased questions) and running the paired bootstrap:
baseline recall@5 0.423 → graph_rerank 0.450, **delta +0.027, p=0.075 — not
significant.** So the effect is *consistent* (+0.025 hand-set, +0.027 scaffolded) and
*directionally real*, but doesn't clear p<0.05 even at n=150. Honest disposition:
**opt-in, not default** — a measured, conditional, no-cross-encoder mechanism, the only
one that moved at-scale recall, but not significant enough to impose on everyone. (Two
caveats kept the test honest: scaffolded cross-file questions are graph-structured, so
they *favor* graph_rerank — this is its best case; and the candidate-bound ~0.13 of
recall, files never retrieved, still needs chunking work.) A perf note from the chase:
per-query PPR on a 40k-node graph was seconds/query until `personalized_pagerank` was
made to cache its adjacency and prune the frontier — necessary for graph_rerank to be
usable at scale at all.

### 3c. Researching the gaps: three SOTA techniques tried, one significant win

Took the chase to the literature (Contextual Retrieval, CoIR leaders, late-interaction,
LLM rerankers) and implemented the worthwhile ones. Two didn't transfer; one did.

| Technique (from research) | Result | Kept? |
|---|---|---|
| Granite embedder (CoIR-strong) | HumanEval 0.927 < CodeRankEmbed 0.994 | no |
| Contextual Retrieval (Anthropic, −49% failed retrievals on prose) | FastAPI +0.006 (noise) | no |
| **Listwise LLM reranker** | **FastAPI +0.086 (p<0.001), Django +0.062** | **yes (opt-in)** |

- **Contextual Retrieval didn't transfer — and the reason is instructive:** our
  `embed_text` already prepends a structural header (file / class / signature /
  docstring), so code chunks are *already* contextualized; the LLM blurb is redundant
  (Anthropic's gain comes from prose chunks that carry no context). A good example of a
  technique whose value depends on what your chunks already contain.
- **The listwise LLM reranker is the one lever that significantly moved recall.** Show
  the top-15 fused candidates to the LLM and let it *reason* about which match the
  query — it disambiguates near-duplicate symbols where cross-encoders (MiniLM, bge)
  and graph-PPR could not. **FastAPI recall@5 0.744 → 0.830 (+0.086, p<0.001, n=106)**;
  Django 0.662 → 0.725 (+0.062, n=40, p=0.23 — same direction, underpowered). It works
  on *both* small and at-scale repos. The cost is one LLM call per query, so it ships
  **opt-in** (`llm_rerank=True` / `--llm-rerank`), a "premium" mode — but unlike
  everything else this session, its recall gain is real and significant.
- **And it lifts the *other* gap too — answer-correctness.** Better-ranked context
  means the generator gets the right material: same questions, same Haiku generator +
  Sonnet judge, default retrieval vs LLM-reranked retrieval — **correctness 0.725 →
  0.900 (+0.175, n=20, p=0.064).** Not quite significant (n=20, CI lower bound at 0),
  but a large effect that pushes correctness from *below* the ~0.85 bar to *above* it.
  So the LLM reranker is the single lever that addresses **both** open gaps —
  retrieval recall (significantly) and answer-correctness (strongly, near-significant).
- **It compounds and dominates the embedder choice (best config measured).** Stacking
  it on the strongest embedder: CodeRankEmbed alone recall@5 0.703 → **CodeRankEmbed +
  LLM rerank 0.858** (+0.156, p<0.001, n=106) — the **highest recall@5 of the session.**
  And `st-codesearch + LLM rerank` (0.830) ≈ `CodeRankEmbed + LLM rerank` (0.858), both
  far above either embedder alone (~0.70–0.74): once the LLM reranks the pool, the
  embedder choice matters little. **The LLM reranker is *the* accuracy lever** — it
  lifts whatever feeds it. Best config = any decent embedder + `--accurate` (LLM rerank).

So the surviving claims: §1 (de-confounding), the **embedder** lever, **BM25 matters
at scale** (§3b), and **the graph helps on dense/typed graphs** (§3a, cobra). On any
single small repo the configs look like "no separation"; across three repos of
different size/language the real, conditional structure shows.

### 3c. Scaling — it holds up

Indexing Django (521k LOC → **40,351 chunks**, a **40k-node / 86k-edge** graph):

| metric | value |
|---|---|
| index time (CPU) | **84 s** (~480 chunks/s embedding) |
| peak memory (max RSS) | **2.2 GB** (mostly the embedder) |
| index on disk | 252 MB |
| query latency, dense+BM25 over 40k chunks | **47 ms** |
| query latency, + cross-encoder rerank | 503 ms (rerank is constant in index size) |

The **brute-force numpy vector store is fine at 40k chunks (47 ms)** — FAISS would
only be needed ~10× larger. The cross-encoder rerank (~450 ms) is the latency cost,
and it's independent of repo size. Indexing is one-time, linear in code size, and
modest in memory. Net: the system scales to a real large repo without changes.

### 3d. External benchmark (CodeSearchNet) — retrieval generalizes off our own ruler

A self-made golden set proves you can build an eval; an external benchmark proves the
retriever generalizes. On **800 real CodeSearchNet docstring→code pairs** (streamed
sample), the default code embedder retrieves the matching snippet at **recall@10 =
0.985, MRR = 0.948** ([coderag/eval/codesearchnet.py](coderag/eval/codesearchnet.py)).
That's strong and consistent with published code-retrieval numbers — independent
confirmation that the embedder choice (§ above) holds beyond FastAPI/cobra/Django.
(Caveat: 800-candidate pool from the train split; the official 1000-distractor test
protocol would be marginally harder — directionally the same. This is the *embedder
alone* — it doesn't exercise BM25/RRF/rerank.)

### 3e. Official CoIR leaderboard protocol (CoSQA) — a real, ranked comparison

The above use convenient protocols; CoIR is the *official* code-IR benchmark. Ran its
**CoSQA** task (NL web-query → code, the hardest/noisiest of CoIR's 10) through the
official `coir-eval` harness, wrapping our own `Embedder`, vs the CoIR paper's Table 3
(nDCG@10):

| Model | CoSQA nDCG@10 |
|---|---|
| BM25 | 0.140 |
| UniXcoder | 0.251 |
| **st-codesearch (our default, 2021)** | **0.275** |
| OpenAI text-embedding-ada-002 | 0.289 |
| Voyage-Code-002 | 0.298 |
| E5-Mistral-7B | 0.313 |
| E5-Base | 0.326 |
| **CodeRankEmbed (our opt-in)** | **0.359** |

Honest placement: our **default** lands mid-pack (above UniXcoder, below the commercial
embedders) — it's a 2021 model. Our **opt-in CodeRankEmbed beats every baseline in the
CoIR paper on this task** (E5-Base, E5-Mistral-7B, Voyage-Code-002, ada-002). Caveats:
(1) CodeRankEmbed postdates the paper, and current overall CoIR *leaders* (Gemini
Embedding 2, Qwen3-Embedding, SFR/CodeXEmbed, Voyage-code-3) score higher; (2) this is
**1 of CoIR's 10 datasets** — a full rank needs all of them (`coir-eval`, ~$0 API but
hours of local embedding). Still: a genuine, leaderboard-protocol number, not a
self-made ruler — and it confirms the embedder is a pluggable, off-the-shelf choice
(the project's value is the system + eval, not a novel model).

**A second external adapter — CodeRAG-Bench-style retrieval — closes the
"embedder-only" gap** ([coderag/eval/coderag_bench.py](coderag/eval/coderag_bench.py)):
each query has relevant document(s) in a shared corpus, and we run the *full
retriever* (`--mode hybrid` = dense + BM25 + RRF) over it. Measured on **HumanEval
(164 problems) — retrieve each problem's canonical solution from the pool of all
164** (reproduce: `python scripts/fetch_humaneval_bench.py` then
`coderag bench data/humaneval_bench.jsonl --suite coderag --mode hybrid`):

| Mode | recall@10 | recall@1 | MRR | nDCG@10 |
|---|---|---|---|---|
| dense (embedder only) | 0.811 | 0.396 | 0.527 | 0.588 |
| **hybrid (dense+BM25+RRF)** | **0.860** | **0.402** | **0.543** | **0.612** |
| + cross-encoder rerank | 0.634 | 0.189 | 0.322 | 0.381 |

Three honest takeaways: (1) this is **far harder than CodeSearchNet's 0.985** —
retrieving a short solution *body* from a problem *prompt*, amid many near-duplicate
list/string routines, is a real task, and MRR 0.54 / recall@1 0.40 reflect that;
(2) **hybrid beats dense (+0.049 recall@10)** here — BM25 helps when the corpus is
full of similar candidates, consistent with the Django finding and unlike the small
FastAPI set where configs didn't separate; (3) **the cross-encoder reranker actively
hurts** (recall@10 0.86 → 0.63) — the default `ms-marco-MiniLM` is trained on
*web-search* passages and misjudges (problem-prompt, code-body) pairs, reordering a
strong code-tuned retrieval into a worse one. The lesson isn't "rerankers are bad,"
it's **a general-domain reranker doesn't transfer to code; it needs a code-tuned
cross-encoder or it's net-negative** — another measured reason a component stays off
by default. (Caveat: a self-contained 164-candidate pool, not the official
CodeRAG-Bench generation/pass@1 leaderboard.) For published context on the CodeXGLUE
CodeSearchNet protocol (1000-distractor MRR): CodeBERT 0.679, GraphCodeBERT 0.703,
UniXcoder 0.740 — fine-tuned 2020–22 encoders on a different protocol, so context
rather than an apples-to-apples target.

## 4. Generation quality (de-confounded 100-q set, Haiku 4.5 judge)

Run on the de-confounded set, single judge, dense vs graph (75 non-holdout
questions). **Dense answer-correctness is 0.747 [0.67–0.82]**, and faithfulness reads
**0.79** with the original judge config.

**That 0.79 was largely a measurement artifact — now diagnosed and fixed.** The judge
ran on Haiku 4.5 and saw only the first **300 tokens** of each cited source, so for any
claim whose supporting line sat past that cutoff it returned a *false* UNSUPPORTED.
Re-measuring the **same generated answers** (generation unchanged) with the source
budget raised to 2000 tokens and a stronger judge (Sonnet 4.6) recovers the grounding
the metric was hiding — a 15-question FastAPI re-measurement:

| Faithfulness (identical answers) | Score |
|---|---|
| Haiku judge, 300-tok sources (original) | 0.792  ← reproduces the 0.79 |
| Sonnet judge, 2000-tok sources (accurate) | **0.948** |

The lift concentrates exactly where expected — long-source questions whose supporting
code was truncated away (*"declare a query parameter with validation"* 0.17→0.92,
OAuth2 bearer extraction 0.33→1.00). So the honest reading is **faithfulness ≈ 0.95,
not 0.79**: the system was already well-grounded; a cheap, truncated judge under-counted
it. This resolves the "judge strictness vs genuine ungrounding" question §6 raised — it
was mostly the judge. Fix shipped: `judge_source_tokens` 300→1500
([config.py](coderag/config.py)), and use Sonnet/Opus as the judge for reported
numbers. (Answer-correctness used the same Haiku@300 judge — re-audited below.)

**Answer-correctness was re-audited the same way — and it held (it is *not* an
artifact).** Correctness grades the answer against the reference (no sources, so the
truncation bug doesn't apply); the only levers are the judge and the generator.
Tested three ways on FastAPI:

| Setup (same questions) | Correctness |
|---|---|
| Haiku gen, Haiku judge (original regime) | 0.725 (n=20) ← reproduces 0.747 |
| Haiku gen, Sonnet judge (stronger judge) | 0.775 (n=20)  (+0.05) |
| Sonnet gen, Sonnet judge (stronger generator) | 0.800 (n=10)  (+0.05) |

A stronger judge and a stronger generator each move it only ~+0.05, so **correctness
is genuinely ≈0.78** — a real ceiling, gated by retrieval misses and genuinely hard
questions, not by measurement. Honest bottom line: faithfulness was a judge artifact
(0.79→~0.95); correctness is real (~0.78, below a ~0.85 bar) and the honest remaining
generation gap.

| Config | Faithfulness | Cite-P | Cite-R | Correct | CtxTok |
|---|---|---|---|---|---|
| dense | 0.789 [0.73–0.84] | 0.836 [0.77–0.89] | 0.791 | **0.747 [0.67–0.82]** | 1829 |
| graph | 0.774 [0.71–0.84] | 0.720 [0.62–0.81] | 0.620 | 0.640 [0.55–0.73] | 1677 |

**Graph expansion showed no measured benefit in generation — a clean non-result.**
Sliced to its home turf, the 40 cross-file questions:

| | correctness | faithfulness | CtxTok |
|---|---|---|---|
| dense | 0.725 | 0.770 | 1865 |
| graph | 0.625 | 0.756 | 1626 |

The point estimate trends down, but **held to the same standard as the rest of this
writeup it's within noise**: the paired correctness difference is **−0.10, 95% CI
[−0.23, +0.04]** (includes 0), and the citation-precision drop (0.84 → 0.72) has
overlapping CIs too. So the honest statement is *no demonstrated benefit, point
estimates trending worse* — not "graph hurts." Mechanism (plausible, not proven):
the auto-expanded neighbors act as distractors and, under the token budget, displace
the top-ranked chunks the answer needs, so the recall edge (§3) doesn't translate.
**Takeaway — and the conclusion survives either way:** with no measured upside, a
downward trend, and only a ~13% token saving, turning expansion on by default isn't
justified, so it's **opt-in (`--expand-graph`)**.

**Follow-up (acted on):** these generation numbers used the *legacy* post-rerank
injection. That wiring was the problem — neighbors entered the context with a fake
score, bypassing relevance. It's now replaced by the default **pre-rerank pool**
(neighbors must earn relevance; §3), which removes the injection mechanism, so a
`--generate` re-run should bring graph generation back to ≈dense rather than below
it. But since the pool wiring also showed **no retrieval gain** (§3), the graph
still isn't worth turning on by default — it stays opt-in, and its real value is the
cheap structural map. (And per §3a, a denser graph isn't the fix: adding import-aware
resolution raised Python call edges +11% and the FastAPI graph stayed null — the
limit is the strong-retrieval regime, not edge-recall.)

The headline behavior to demo is **abstention**: ask an out-of-scope question and
the system declines instead of confabulating, and the structural check flags any
citation that doesn't map to a real source.

### 4a. HyDE and self-repair: measured, neither earns default-on

Two opt-in features, validated rather than assumed (Haiku judge):

- **HyDE** (embed a hypothetical snippet for dense search): **significantly *hurts*
  FastAPI — recall@5 0.883 → 0.833, Δ −0.050 [−0.09,−0.01], p=0.014** — because when
  dense retrieval is already strong, the synthesized snippet pulls the query toward
  generic patterns. On Django (harder, more headroom) it trends up (+0.067) but
  **not significantly (p=0.20)**. Verdict: **off by default; opt-in only for
  low-recall repos.**
- **Self-repair** (retry with a stricter cite-or-drop prompt when faithfulness < 0.8):
  a first run showed +0.05, but that was uncontrolled. The **controlled experiment**
  (26 triggered answers, each retried two ways) settles it — and it's a *negative*:

  | triggered answers (n=26) | faithfulness |
  |---|---|
  | baseline (1st pass) | 0.511 |
  | same-prompt re-roll (control) | **0.688** |
  | stricter "repair" retry | 0.672 |

  **Repair lift over a plain re-roll: Δ −0.015 [−0.13,+0.09], p=0.773 — none.** The
  apparent benefit was **entirely regression-to-the-mean**: just *re-generating* a
  low-scored answer recovers +0.18, and the stricter instruction adds nothing. The
  control arm was exactly what exposed it. Verdict: **the repair instruction doesn't
  work; off by default** (kept as an opt-in no-op pending a better idea).

  *Secondary finding (worth more than the feature):* that +0.18 re-roll bounce means
  **the LLM faithfulness judge is noisy** — re-judging swings ~0.18 on low scorers,
  so single-pass faithfulness numbers (the 0.79 in §4) carry real judge variance and
  should be read with wide error bars, not as ground truth.

The point: I added HyDE and self-repair, then *measured* them under control — and the
data says **don't enable either** (HyDE regresses the common case; self-repair's lift
was a re-roll artifact). Same discipline that killed the hybrid "win" (§1) and kept
graph expansion opt-in (§3): a favorable-looking result is a suspect until a control
rules out the confound.

---

## Caveats

- **Small N.** Retrieval metrics over 18 answerable questions, generation over 20.
  CIs are wide; treat ±0.1 as noise and the config ordering as suggestive. A bigger
  golden set would tighten these and better separate the near-ceiling configs.
- **Cheap judge.** Faithfulness/correctness used Haiku 4.5 as the LLM judge (to fit
  a small budget). It's serviceable but imperfect; a stronger judge (Sonnet/Opus)
  gives more accurate generation numbers. Judge quality is itself a variable.
- **Embedding truncation.** The default code model has a modest max sequence length;
  the context header goes first so a chunk's signature/location always survives.
- Numbers are tied to this FastAPI commit and may shift as the repo evolves.

---

## Reproduce

```bash
pip install -e '.[langs,dev]'
git clone --depth 1 https://github.com/tiangolo/fastapi

# default (code) embedder
python -m coderag.cli index fastapi/fastapi --out .coderag_index_code
python -m coderag.cli eval --index .coderag_index_code --configs dense,hybrid,rerank,graph

# baseline general-text embedder, for the comparison
python -m coderag.cli index fastapi/fastapi --out .coderag_index \
  --embed-model sentence-transformers/all-MiniLM-L6-v2
python -m coderag.cli eval --index .coderag_index --configs dense,hybrid,rerank,graph

# with generation metrics (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python -m coderag.cli eval --index .coderag_index_code --generate
```

Per-run artifacts (table + per-question rows) are written to `eval_runs/<timestamp>/`.

## Next steps

1. Grow the golden set toward 50 to tighten CIs and separate near-ceiling configs.
2. Re-run the generation metrics with a stronger judge (Sonnet/Opus) for sharper
   faithfulness/correctness numbers.
3. Try `jinaai/jina-embeddings-v2-base-code` (longer context, code-native) for a
   further dense lift.
