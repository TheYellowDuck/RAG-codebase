# Official CoIR full-10 benchmark — run on a free GPU (Google Colab).
#
# Local MPS Macs thrash on CoIR's ~1M-doc splits; a cloud GPU finishes in ~20-40 min.
# $0 LLM API (pure local embedding on the GPU instance).
#
#   1. Open https://colab.research.google.com  ->  Runtime > Change runtime type > T4 GPU
#   2. Paste this whole file into a cell and run it.
#   3. It prints the official mean nDCG@10 for both embedders across all 10 CoIR tasks.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "coir-eval", "sentence-transformers", "einops"], check=True)

import coir
from statistics import mean
from sentence_transformers import SentenceTransformer


class Model:
    """CoIR model interface, wrapping a sentence-transformers model with the
    query prefix the embedder expects (asymmetric models need it)."""
    def __init__(self, name, query_prefix="", trust=False):
        self.m = SentenceTransformer(name, trust_remote_code=trust)
        self.qp = query_prefix

    def encode_queries(self, queries, batch_size=128, **kw):
        return self.m.encode([self.qp + q for q in queries], batch_size=batch_size,
                             normalize_embeddings=True, show_progress_bar=False)

    def encode_corpus(self, corpus, batch_size=128, **kw):
        texts = [((d.get("title") or "") + " " + (d.get("text") or "")).strip() for d in corpus]
        return self.m.encode(texts, batch_size=batch_size,
                             normalize_embeddings=True, show_progress_bar=False)


TASKS = ["codetrans-dl", "codetrans-contest", "cosqa", "stackoverflow-qa",
         "synthetic-text2sql", "codefeedback-st", "codefeedback-mt", "apps",
         "codesearchnet", "codesearchnet-ccr"]   # last two expand into 6 languages each

EMBEDDERS = [
    ("st-codesearch (default)",
     "flax-sentence-embeddings/st-codesearch-distilroberta-base", "", False),
    ("CodeRankEmbed (opt-in)",
     "nomic-ai/CodeRankEmbed", "Represent this query for searching relevant code: ", True),
]

for label, name, qprefix, trust in EMBEDDERS:
    model = Model(name, qprefix, trust)
    per_task = []
    for t in TASKS:
        ev = coir.COIR(tasks=coir.get_tasks(tasks=[t]), batch_size=128)
        res = ev.run(model, output_folder=f"/tmp/coir/{label}")
        # codesearchnet / -ccr expand into 6 per-language sub-tasks -> average them.
        score = mean(v["NDCG"]["NDCG@10"] for v in res.values())
        per_task.append((t, score))
        print(f"  {label:26} {t:20} nDCG@10={score:.4f}", flush=True)
    overall = mean(s for _, s in per_task)
    print(f">>> {label}: official CoIR-10 mean nDCG@10 = {overall:.4f}\n", flush=True)
