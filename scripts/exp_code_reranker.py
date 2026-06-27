"""Experiment: does a CODE-capable cross-encoder reranker beat the hybrid baseline?

The general-domain cross-encoders the project shipped/tested (ms-marco-MiniLM, bge-m3)
HURT code retrieval (RESULTS §3b/§3d), which is why use_rerank defaults False. This
tests the open cell: a code-capable cross-encoder (jinaai/jina-reranker-v2-base-multilingual).

Two-env procedure (jina can't load in the project's Python 3.14 + transformers 5.x env):

  # 1) DUMP candidate pools from the project's own fused retriever (main .venv):
  python scripts/exp_code_reranker.py dump .idx_fastapi data/golden_questions.jsonl fastapi
  python scripts/exp_code_reranker.py dump-humaneval data/humaneval_bench.jsonl

  # 2) RERANK + score in an isolated Python 3.11 env (jina needs old transformers):
  python3.11 -m venv /tmp/ce311
  /tmp/ce311/bin/pip install torch 'transformers==4.44.2' einops numpy
  /tmp/ce311/bin/python scripts/exp_code_reranker.py rerank fastapi
  /tmp/ce311/bin/python scripts/exp_code_reranker.py rerank humaneval

Pools are scored on (query, embed_text) — exactly matching coderag/retrieve/rerank.py:45
for the golden sets; HumanEval scores (query, raw solution code) with NO file header, so
that result is free of the file-path-in-header confound. Baseline = the SAME fused pool in
fused (RRF) order. Significance = the repo's paired bootstrap.
"""
import json
import random
import sys

POOL = 50
RERANKER = "jinaai/jina-reranker-v2-base-multilingual"


def _paired(a, b, n=2000, seed=0):
    d = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    N = len(d)
    res = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    p = 2 * min(sum(1 for x in res if x <= 0) / n, sum(1 for x in res if x >= 0) / n)
    return sum(d) / N, min(1.0, p)


def _hit(ids, gold_set, k):
    got = [i for i in ids[:k] if i in gold_set]
    return len(set(got)) / len(gold_set) if gold_set else 0.0


def _rr(ids, gold_set):
    for i, x in enumerate(ids, 1):
        if x in gold_set:
            return 1.0 / i
    return 0.0


def _dedup(items):
    out = []
    for it in items:
        if it not in out:
            out.append(it)
    return out


def cmd_dump(index_path, golden_path, tag):
    from coderag.config import Settings, load_dotenv
    from coderag.eval.run import load_golden
    from coderag.index import CodeIndex
    from coderag.retrieve.retriever import rrf
    load_dotenv()
    s = Settings.from_env()
    idx = CodeIndex.load(index_path)
    emb = idx.embedder
    out = []
    for q in (q for q in load_golden(golden_path) if not q.holdout and q.relevant_files):
        qv = emb.encode_query(q.question)
        fused = rrf([c for c, _ in idx.vector_store.search(qv, s.dense_top_n)],
                    [c for c, _ in idx.bm25.search(q.question, s.bm25_top_n)], k=s.rrf_k)[:POOL]
        cands = [{"key": ch.file_path, "text": ch.embed_text[:2000]}
                 for cid in fused if (ch := idx.get_chunk(cid))]
        out.append({"q": q.question, "gold": q.relevant_files, "cands": cands, "unit": "file"})
    json.dump(out, open(f"/tmp/rerank_pool_{tag}.json", "w"))
    print(f"{tag}: dumped {len(out)} questions")


def cmd_dump_humaneval(bench_path):
    import numpy as np
    from rank_bm25 import BM25Okapi

    from coderag.config import Settings, load_dotenv
    from coderag.embed import Embedder
    from coderag.retrieve.retriever import rrf
    from coderag.tokenization import code_tokens
    load_dotenv()
    recs = [json.loads(l) for l in open(bench_path) if l.strip()]
    ids = [r["id"] for r in recs]
    docs = [r["code"] for r in recs]
    queries = [r["query"] for r in recs]
    emb = Embedder.from_settings(Settings.from_env())
    dv = np.asarray(emb.encode(docs), dtype=np.float32)
    qv = np.asarray(emb.encode(queries, is_query=True), dtype=np.float32)
    bm = BM25Okapi([code_tokens(d) for d in docs])
    id2 = dict(zip(ids, docs, strict=True))
    out = []
    for i, q in enumerate(queries):
        dorder = [ids[j] for j in np.argsort(-(dv @ qv[i]))[:POOL]]
        border = [ids[j] for j in np.argsort(-bm.get_scores(code_tokens(q)))[:POOL]]
        fused = rrf(dorder, border, k=Settings().rrf_k)[:POOL]
        # gold = the query's own solution id; doc text is RAW code (no file header)
        out.append({"q": q, "gold": [ids[i]],
                    "cands": [{"key": cid, "text": id2[cid][:2000]} for cid in fused],
                    "unit": "doc"})
    json.dump(out, open("/tmp/rerank_pool_humaneval.json", "w"))
    print(f"humaneval: dumped {len(out)} queries")


def cmd_rerank(tag):
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(
        RERANKER, torch_dtype="auto", trust_remote_code=True)
    m.to("mps")
    m.eval()
    data = json.load(open(f"/tmp/rerank_pool_{tag}.json"))
    ks = (5, 10) if data[0]["unit"] == "doc" else (5,)
    base = {k: [] for k in ks}
    rer = {k: [] for k in ks}
    base_rr, rer_rr = [], []
    for r in data:
        gold = set(r["gold"])
        base_keys = _dedup([c["key"] for c in r["cands"]])
        sc = m.compute_score([[r["q"], c["text"]] for c in r["cands"]], max_length=512)
        rer_keys = _dedup([c["key"] for c, _ in
                           sorted(zip(r["cands"], sc, strict=True), key=lambda x: float(x[1]), reverse=True)])
        for k in ks:
            base[k].append(_hit(base_keys, gold, k))
            rer[k].append(_hit(rer_keys, gold, k))
        base_rr.append(_rr(base_keys, gold))
        rer_rr.append(_rr(rer_keys, gold))
    n = len(data)
    print(f"\n=== {tag} (n={n}), {RERANKER} (pool {POOL}) ===")
    for k in ks:
        d, p = _paired(rer[k], base[k])
        print(f"  recall@{k}: {sum(base[k])/n:.3f} -> {sum(rer[k])/n:.3f}  (Δ{d:+.3f}, paired p={p:.3f})")
    dm, pm = _paired(rer_rr, base_rr)
    print(f"  MRR:      {sum(base_rr)/n:.3f} -> {sum(rer_rr)/n:.3f}  (Δ{dm:+.3f}, paired p={pm:.3f})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "dump":
        cmd_dump(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "dump-humaneval":
        cmd_dump_humaneval(sys.argv[2])
    elif cmd == "rerank":
        cmd_rerank(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
