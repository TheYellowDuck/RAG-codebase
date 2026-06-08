# Results — Code RAG on FastAPI

A measured walk-through of the retrieval pipeline on a real codebase. The point
of the [eval harness](coderag/eval/) is to make every design choice attributable;
this is that story.

- **Target repo:** [tiangolo/fastapi](https://github.com/tiangolo/fastapi) @ `5cdf820c8046`
- **Golden set:** [data/golden_questions.jsonl](data/golden_questions.jsonl) — 30
  questions (factual / how-to / where / cross-file / out-of-scope), 10 holdout,
  every `relevant_files`/`relevant_symbols` grep-verified against the source.
  Metrics below are over the **18 answerable dev questions** (holdout excluded;
  out-of-scope excluded from retrieval metrics — they test abstention, not recall).
- **Stack:** local embeddings + local cross-encoder rerank; generation/judging go
  through the pluggable provider layer (default Claude `claude-opus-4-8`, or any
  OpenAI-compatible provider). All numbers here are **retrieval-only** (no API key
  needed); generation metrics are not yet measured (see Caveats).
- **Granularity:** file-level (a chunk "hits" if its file is in `relevant_files`);
  `SymR@5` is symbol-level recall.

## TL;DR

> **Switching the embedding model from a general-text model to a code-search model
> lifted dense recall@5 from 0.83 → 0.97 — a bigger gain than reranking and the
> code graph combined.** Once dense retrieval saturates, hybrid (BM25+RRF) wins on
> ranking (MRR, symbol recall) and reranking wins on precision.

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

## Full ablation (18 dev questions, recall@5 with bootstrap CI)

### General-text embedder (`all-MiniLM-L6-v2`)

| Config | Recall@5 | MRR | NDCG@5 | P@5 | SymR@5 |
|---|---|---|---|---|---|
| dense  | 0.83 [0.67–1.00] | 0.74 | 0.76 | 0.23 | 0.49 |
| hybrid | 0.78 [0.61–0.94] | 0.70 | 0.71 | 0.28 | 0.62 |
| rerank | 0.89 [0.72–1.00] | 0.78 | 0.81 | 0.35 | 0.71 |
| graph  | 0.89 [0.72–1.00] | 0.78 | 0.81 | 0.22 | 0.77 |

Here the ceiling is lower, so the ladder is visible: rerank lifts recall
(0.83→0.89) and the graph lifts symbol recall (0.49→0.77).

### Code-search embedder (default)

| Config | Recall@5 | MRR | NDCG@5 | P@5 | SymR@5 |
|---|---|---|---|---|---|
| dense  | 0.97 [0.92–1.00] | 0.77 | 0.81 | 0.29 | 0.75 |
| hybrid | 0.92 [0.78–1.00] | **0.87** | **0.87** | 0.33 | **0.83** |
| rerank | 0.97 [0.92–1.00] | 0.82 | 0.86 | **0.41** | 0.77 |
| graph  | 0.97 [0.92–1.00] | 0.82 | 0.86 | 0.28 | 0.77 |

---

## Reading the results

- **Embedder choice dwarfs everything else.** +0.14 recall@5 from a one-line model
  swap vs. +0.06 from reranking. For code RAG, the embedder is the first thing to
  tune, not the last.
- **Dense recall saturates** with the code embedder (0.97), so file-recall@5 can't
  separate the configs — a ceiling effect, not equivalence. The other metrics do
  separate them:
  - **Hybrid (BM25 + RRF) wins ranking** — best MRR (0.87) and symbol recall
    (0.83). Exact-identifier matching sharpens *where* the right chunk lands, which
    is what BM25 is for (queries like `RequestValidationError` / `solve_dependencies`).
  - **Reranking wins precision** — P@5 0.41, the highest. Fewer, more on-target
    chunks per answer (good for a tight generation budget).
- **An honest trade-off:** hybrid recall@5 (0.92) dips *below* dense (0.97). When
  dense is already excellent, BM25 fusion occasionally bumps a relevant file out of
  the top-5 even as it improves overall ranking. Worth knowing before turning every
  knob to "on."
- **The graph's job is symbol/structure, not raw file recall** — its value shows in
  symbol recall and in handing generation connected context (callees/callers) cheaply;
  it slightly lowers P@5 because expansion adds neighbor chunks that aren't all
  answer-files.

---

## Caveats

- **N = 18 dev questions.** CIs are wide; treat ±0.1 as noise. Expanding the golden
  set further would tighten these and better separate near-ceiling configs.
- **Retrieval-only.** Faithfulness, answer-correctness, and citation precision/recall
  need a Claude key and are **not yet measured** — run `eval --generate` to fill them in.
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

1. Run `eval --generate` to add faithfulness / answer-correctness / citation metrics.
2. Grow the golden set toward 50 to tighten CIs and separate near-ceiling configs.
3. Try `jinaai/jina-embeddings-v2-base-code` (longer context, code-native) for a
   further dense lift.
