# Official CoIR full-10 benchmark — run on a free cloud notebook (Colab / Kaggle).
#
# $0 LLM API (pure local embedding on the instance's GPU).
# Reports the official mean nDCG@10 for both embedders across all 10 CoIR tasks.
#
# RAM NOTE (important): free Colab has only ~13 GB RAM. CoIR's `codesearchnet`
# (~1M docs) can exceed it. This script is memory-frugal — it runs each task (and
# each per-language split) one at a time and frees memory between them — but if free
# Colab still crashes on codesearchnet, either:
#   (a) set SKIP_HEAVY = True below  -> a robust 8-task run that fits ~13 GB, or
#   (b) use Kaggle (free, ~30 GB RAM + GPU) or Colab Pro for the full 10.
#
#   1. Colab: Runtime > Change runtime type > T4 GPU.  (Kaggle: enable GPU.)
#   2. Paste this whole file into a cell and run.

SKIP_HEAVY = False   # True -> skip codesearchnet / -ccr (the ~1M-doc tasks); 8-task run

import subprocess, sys, gc
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "coir-eval", "sentence-transformers", "einops"], check=True)

import coir
from statistics import mean
from sentence_transformers import SentenceTransformer


class Model:
    """CoIR model interface = a sentence-transformers model + the query prefix the
    embedder needs (mirrors coderag.embed.Embedder)."""
    def __init__(self, name, query_prefix="", trust=False):
        self.m = SentenceTransformer(name, trust_remote_code=trust)
        self.qp = query_prefix

    def encode_queries(self, queries, batch_size=64, **kw):
        return self.m.encode([self.qp + q for q in queries], batch_size=batch_size,
                             normalize_embeddings=True, show_progress_bar=False)

    def encode_corpus(self, corpus, batch_size=64, **kw):
        texts = [((d.get("title") or "") + " " + (d.get("text") or "")).strip() for d in corpus]
        return self.m.encode(texts, batch_size=batch_size,
                             normalize_embeddings=True, show_progress_bar=False)


TASKS = ["codetrans-dl", "codetrans-contest", "cosqa", "stackoverflow-qa",
         "synthetic-text2sql", "codefeedback-st", "codefeedback-mt", "apps"]
if not SKIP_HEAVY:
    TASKS += ["codesearchnet", "codesearchnet-ccr"]   # each expands into 6 languages


def score_task(model, task):
    """Run a task one (per-language) split at a time, freeing memory between splits,
    so peak RAM stays ~one corpus + one embedding matrix — not all 6 at once."""
    data = coir.get_tasks(tasks=[task])          # codesearchnet -> 6 sub-tasks
    subnames = list(data.keys())
    sub_scores = []
    for sub in subnames:
        ev = coir.COIR(tasks={sub: data[sub]}, batch_size=64)
        res = ev.run(model, output_folder=f"/tmp/coir/{sub}")
        sub_scores.append(res[sub]["NDCG"]["NDCG@10"])
        data[sub] = None                          # free this split's corpus
        del ev, res
        gc.collect()
    return mean(sub_scores)


for label, name, qprefix, trust in [
    ("st-codesearch (default)",
     "flax-sentence-embeddings/st-codesearch-distilroberta-base", "", False),
    ("CodeRankEmbed (opt-in)",
     "nomic-ai/CodeRankEmbed", "Represent this query for searching relevant code: ", True),
]:
    model = Model(name, qprefix, trust)
    per_task = []
    for t in TASKS:
        try:
            s = score_task(model, t)
            per_task.append(s)
            print(f"  {label:26} {t:20} nDCG@10={s:.4f}", flush=True)
        except Exception as e:
            print(f"  {label:26} {t:20} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    if per_task:
        print(f">>> {label}: CoIR mean nDCG@10 = {mean(per_task):.4f} "
              f"over {len(per_task)}/{len(TASKS)} tasks\n", flush=True)
    del model
    gc.collect()
