"""Ad-hoc experiment harness for two not-yet-tried retrieval levers (run once,
not part of the package). Uses the project's own metrics + bootstrap so the
numbers are comparable to `coderag eval`.

  E3  query-adaptive fusion weights — boost BM25 when the query names an indexed
       symbol, boost dense for pure-prose queries (vs the fixed equal-weight RRF).
  E4  query decomposition — split a multi-file question into atomic sub-queries
       (1 LLM call), retrieve each, and RRF-union with the original.

Reports file-level recall@5 / MRR with bootstrap CIs + paired-bootstrap p vs the
equal-weight hybrid baseline, sliced by question type.
"""
import sys

from coderag.config import Settings, load_dotenv
from coderag.eval.bootstrap import bootstrap_ci, paired_bootstrap
from coderag.eval.metrics import recall_at_k, reciprocal_rank
from coderag.eval.run import load_golden
from coderag.index import CodeIndex
from coderag.retrieve.retriever import rrf
from coderag.tokenization import code_tokens

load_dotenv()
INDEX = sys.argv[1] if len(sys.argv) > 1 else ".idx_fastapi"
GOLDEN = "data/golden_questions.jsonl"

s = Settings.from_env()
index = CodeIndex.load(INDEX)
embedder = index.embedder
questions = [q for q in load_golden(GOLDEN) if not q.holdout]

# --- symbol vocabulary for identifier detection (E3) -----------------------
# A token only signals "the user named a specific symbol" if it is RARE — common
# English (function/from/class/request/type/model) lives in the symbol vocab too
# and made the v1 gate fire on 100% of queries (== a constant BM25 reweight, not
# adaptive). Gate on symbol document-frequency + a stopword set instead.
from collections import Counter

_STOP = {"function", "class", "method", "request", "response", "model", "type",
         "value", "field", "param", "params", "config", "handler", "data", "from",
         "with", "that", "this", "does", "build", "create", "into", "used", "uses"}
_sym_df: Counter = Counter()
for ch in index.chunks.values():
    if ch.symbol_name:
        for t in {t.lower() for t in code_tokens(ch.symbol_name)}:
            _sym_df[t] += 1


def _rare_identifier_hit(query) -> bool:
    for t in {t.lower() for t in code_tokens(query)}:
        if len(t) >= 5 and t not in _STOP and 1 <= _sym_df.get(t, 0) <= 3:
            return True
    return False


def files_from_ids(ids):
    out = []
    for cid in ids:
        ch = index.get_chunk(cid)
        if ch and ch.file_path not in out:
            out.append(ch.file_path)
    return out


def fuse(query, w_dense, w_bm25):
    qvec = embedder.encode_query(query)
    dense = index.vector_store.search(qvec, s.dense_top_n)
    lexical = index.bm25.search(query, s.bm25_top_n)
    fused = rrf([c for c, _ in dense], [c for c, _ in lexical],
                k=s.rrf_k, weights=[w_dense, w_bm25])
    return files_from_ids(fused)


_fire = [0, 0]   # [identifier-queries, total]


def adaptive_weights(query):
    _fire[1] += 1
    if _rare_identifier_hit(query):
        _fire[0] += 1
        return 1.0, 1.5      # query names a rare symbol → lean BM25
    return 1.5, 1.0          # pure prose → lean dense (the stronger retriever here)


# --- E4: LLM query decomposition -------------------------------------------
_llm = None


def decompose(query):
    global _llm
    if _llm is None:
        from coderag.llm import get_llm_client
        _llm = get_llm_client()
    try:
        txt = _llm.generate(
            "You split a code-search question into 2-3 atomic sub-questions, each "
            "naming ONE concept/function/file to find. Output one per line, no numbering.",
            f"Question: {query}", max_tokens=120).text
        subs = [ln.strip("-* \t") for ln in txt.splitlines() if ln.strip()]
        return subs[:3] or [query]
    except Exception:
        return [query]


def decomposed_files(query):
    subs = decompose(query)
    lists = [[c for c, _ in index.bm25.search(query, s.bm25_top_n)]]   # keep the original
    qvec = embedder.encode_query(query)
    lists.append([c for c, _ in index.vector_store.search(qvec, s.dense_top_n)])
    for sub in subs:
        sv = embedder.encode_query(sub)
        lists.append([c for c, _ in index.vector_store.search(sv, s.dense_top_n)])
        lists.append([c for c, _ in index.bm25.search(sub, s.bm25_top_n)])
    return files_from_ids(rrf(*lists, k=s.rrf_k))


# --- run all configs, collect per-question recall@5 / RR -------------------
def evaluate(fn, subset=None):
    rec, rr = [], []
    for q in questions:
        if subset and q.type not in subset:
            continue
        if not q.relevant_files:
            continue
        files = fn(q.question)
        rec.append(recall_at_k(files, q.relevant_files, 5))
        rr.append(reciprocal_rank(files, q.relevant_files))
    return rec, rr


def line(name, rec, rr, base_rec=None):
    r = bootstrap_ci(rec)
    m = bootstrap_ci(rr)
    sig = ""
    if base_rec is not None and len(base_rec) == len(rec):
        pb = paired_bootstrap(rec, base_rec)
        sig = f"  (Δrecall {pb['mean_diff']:+.3f}, p={pb['p_value']:.3f})"
    print(f"  {name:22s} recall@5 {r['mean']:.3f} [{r['lo']:.2f}-{r['hi']:.2f}]  "
          f"MRR {m['mean']:.3f} [{m['lo']:.2f}-{m['hi']:.2f}]  n={r['n']}{sig}")


print(f"\n=== {INDEX} | {len(questions)} in-scope questions | {len(_sym_df)} symbol tokens ===")
print("\n[ALL question types]")
base_rec, base_rr = evaluate(lambda q: fuse(q, 1.0, 1.0))
line("baseline hybrid", base_rec, base_rr)
ad_rec, ad_rr = evaluate(lambda q: fuse(q, *adaptive_weights(q)))
line("E3 adaptive fusion", ad_rec, ad_rr, base_rec)

print("\n[cross-file + multi-hop only — the decomposition target]")
multi = {"cross-file", "multi-hop"}
mbase_rec, mbase_rr = evaluate(lambda q: fuse(q, 1.0, 1.0), multi)
line("baseline hybrid", mbase_rec, mbase_rr)
made_rec, made_rr = evaluate(lambda q: fuse(q, *adaptive_weights(q)), multi)
line("E3 adaptive fusion", made_rec, made_rr, mbase_rec)
mdec_rec, mdec_rr = evaluate(decomposed_files, multi)
line("E4 decomposition", mdec_rec, mdec_rr, mbase_rec)

print(f"\n[E3 gate fired on {_fire[0]}/{_fire[1]} queries "
      f"({100*_fire[0]//max(_fire[1],1)}%) — v1 fired on 100%]")
