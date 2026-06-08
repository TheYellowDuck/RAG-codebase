"""The eval runner and the table (outline §6.4).

Loops the golden set, runs retrieval (+ optionally generation), computes every
metric, and prints/saves a markdown table. Run it after each change so
improvements are attributable. Supports ablation configs on a single index
(dense / hybrid / rerank / graph); chunking + context-header ablations are done
by building separate indexes (see CLI flags / README).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import Settings
from ..index import CodeIndex
from ..retrieve import Retriever, Reranker
from .metrics import (
    recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, retrieved_files,
    retrieved_symbols, recall_at_k_symbol, reciprocal_rank_symbol,
)
from .bootstrap import bootstrap_ci


@dataclass
class GoldenQuestion:
    id: str
    question: str
    type: str = ""
    relevant_files: list[str] = field(default_factory=list)
    relevant_symbols: list[str] = field(default_factory=list)
    reference_answer: str = ""
    holdout: bool = False


@dataclass
class RunConfig:
    name: str
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    expand_graph: bool = False


# Ablation ladder runnable on a single index (mirrors outline §6.4's table shape).
DEFAULT_CONFIGS: dict[str, RunConfig] = {
    "dense": RunConfig("dense", use_dense=True, use_bm25=False, use_rerank=False),
    "hybrid": RunConfig("hybrid", use_dense=True, use_bm25=True, use_rerank=False),
    "rerank": RunConfig("rerank", use_dense=True, use_bm25=True, use_rerank=True),
    "graph": RunConfig("graph", use_dense=True, use_bm25=True, use_rerank=True,
                       expand_graph=True),
}


def load_golden(path: str) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            d = json.loads(line)
            questions.append(GoldenQuestion(
                id=d["id"], question=d["question"], type=d.get("type", ""),
                relevant_files=d.get("relevant_files", []),
                relevant_symbols=d.get("relevant_symbols", []),
                reference_answer=d.get("reference_answer", ""),
                holdout=d.get("holdout", False),
            ))
    return questions


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_config(retriever: Retriever, settings: Settings,
               questions: list[GoldenQuestion], config: RunConfig, *,
               eval_k: int = 5, retrieve_k: int = 15, generate: bool = False) -> dict:
    """Run one config over all questions; return per-question rows + aggregate."""
    from ..generate import generate_answer  # local import: only needed with --generate
    from ..verify import faithfulness_score, parse_citations
    from .gen_metrics import answer_correctness, citation_precision_recall

    per_q: list[dict] = []
    client = None  # lazily created on first Claude call; reused across questions

    for q in questions:
        results = retriever.retrieve(
            q.question, k=retrieve_k,
            use_dense=config.use_dense, use_bm25=config.use_bm25,
            use_rerank=config.use_rerank, expand_graph=config.expand_graph,
        )
        files = retrieved_files(results)
        row = {
            "id": q.id, "type": q.type, "question": q.question,
            "retrieved_files": files[:eval_k],
        }
        # Retrieval metrics only apply when the question has answer-bearing files.
        # Out-of-scope questions (empty relevant_files) are scored on abstention in
        # generation instead — counting them as recall=0 would be misleading.
        if q.relevant_files:
            row["recall@k"] = recall_at_k(files, q.relevant_files, eval_k)
            row["precision@k"] = precision_at_k(files, q.relevant_files, eval_k)
            row["mrr"] = reciprocal_rank(files, q.relevant_files)
            row["ndcg@k"] = ndcg_at_k(files, q.relevant_files, eval_k)
        # Optional symbol-granularity (only when the question labels symbols, §6.2).
        if q.relevant_symbols:
            syms = retrieved_symbols(results)
            row["sym_recall@k"] = recall_at_k_symbol(syms, q.relevant_symbols, eval_k)
            row["sym_mrr"] = reciprocal_rank_symbol(syms, q.relevant_symbols)

        if generate:
            if client is None:
                from ..llm import get_llm_client
                client = get_llm_client()
            gen_results = results[: settings.final_k]
            graph_ctx = retriever.graph_context(gen_results) if settings.include_graph_context else None
            ans = generate_answer(q.question, gen_results, settings,
                                  repo=retriever.index.repo, graph_context=graph_ctx,
                                  client=client)
            faith = faithfulness_score(ans.answer, ans.sources, settings, client=client)
            row["answer"] = ans.answer
            row["faithfulness"] = faith["faithfulness"]
            row["n_unsupported"] = len(faith.get("unsupported", []))
            row["context_tokens"] = ans.context_tokens
            cited = parse_citations(ans.answer)
            cit = citation_precision_recall(ans.sources, cited, q.relevant_files)
            row["citation_precision"] = cit["citation_precision"]
            row["citation_recall"] = cit["citation_recall"]
            if q.reference_answer:
                corr = answer_correctness(q.question, ans.answer, q.reference_answer,
                                          settings, client=client)
                row["correctness"] = corr["score"]
                row["correctness_grade"] = corr["grade"]
        per_q.append(row)

    agg = _aggregate(per_q, eval_k, generate)
    return {"config": config.name, "aggregate": agg, "per_question": per_q}


def _aggregate(per_q: list[dict], eval_k: int, generate: bool) -> dict:
    def col(key):
        return [r[key] for r in per_q if key in r]

    agg = {
        "n": len(per_q),
        # Bootstrap CIs on the metrics interviewers scrutinize, so saturation /
        # noise is visible rather than hidden behind point estimates.
        f"recall@{eval_k}": bootstrap_ci(col("recall@k")),
        "mrr": bootstrap_ci(col("mrr")),
        f"ndcg@{eval_k}": bootstrap_ci(col("ndcg@k")),
        f"precision@{eval_k}": {"mean": _mean(col("precision@k"))},
    }
    if any("sym_recall@k" in r for r in per_q):
        agg[f"sym_recall@{eval_k}"] = bootstrap_ci(col("sym_recall@k"))
        agg["sym_mrr"] = {"mean": _mean(col("sym_mrr"))}
    if generate:
        agg["faithfulness"] = bootstrap_ci(col("faithfulness"))
        agg["citation_precision"] = {"mean": _mean(col("citation_precision"))}
        agg["citation_recall"] = {"mean": _mean(col("citation_recall"))}
        if col("context_tokens"):
            agg["context_tokens"] = {"mean": round(_mean(col("context_tokens")))}
        if col("correctness"):
            agg["correctness"] = bootstrap_ci(col("correctness"))
    return agg


def format_table(run_results: list[dict], eval_k: int, generate: bool) -> str:
    has_sym = any(f"sym_recall@{eval_k}" in rr["aggregate"] for rr in run_results)
    headers = ["Config", f"Recall@{eval_k}", "MRR", f"NDCG@{eval_k}", f"P@{eval_k}"]
    if has_sym:
        headers += [f"SymR@{eval_k}", "SymMRR"]
    if generate:
        headers += ["Faithfulness", "Cite-P", "Cite-R", "Correct"]
    rows = ["| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|"]
    if generate:
        headers += ["CtxTok"]
    for rr in run_results:
        a = rr["aggregate"]
        cells = [
            rr["config"],
            _fmt_ci(a[f"recall@{eval_k}"]),
            _fmt_ci(a["mrr"]),
            _fmt_ci(a[f"ndcg@{eval_k}"]),
            f"{a[f'precision@{eval_k}']['mean']:.3f}",
        ]
        if has_sym:
            cells += [
                _fmt_ci(a.get(f"sym_recall@{eval_k}", {})),
                f"{a.get('sym_mrr', {}).get('mean', 0):.3f}",
            ]
        if generate:
            cells += [
                _fmt_ci(a.get("faithfulness", {})),
                f"{a.get('citation_precision', {}).get('mean', 0):.3f}",
                f"{a.get('citation_recall', {}).get('mean', 0):.3f}",
                _fmt_ci(a["correctness"]) if "correctness" in a else "n/a",
                f"{a.get('context_tokens', {}).get('mean', 0)}",
            ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _fmt_ci(ci: dict) -> str:
    if not ci or "mean" not in ci:
        return "n/a"
    if "lo" in ci:
        return f"{ci['mean']:.3f} [{ci['lo']:.2f}–{ci['hi']:.2f}]"
    return f"{ci['mean']:.3f}"


def evaluate(index_dir: str, golden_path: str, *, config_names: Optional[list[str]] = None,
             eval_k: int = 5, retrieve_k: int = 15, generate: bool = False,
             include_holdout: bool = False, out_dir: str = "eval_runs") -> dict:
    """Top-level eval entry point. Returns the full result dict and saves artifacts."""
    index = CodeIndex.load(index_dir)
    settings = index.settings

    config_names = config_names or ["dense", "hybrid", "rerank", "graph"]
    configs = [DEFAULT_CONFIGS[n] for n in config_names if n in DEFAULT_CONFIGS]
    if not configs:
        raise ValueError(f"No valid configs in {config_names}; "
                         f"choose from {list(DEFAULT_CONFIGS)}")

    questions = load_golden(golden_path)
    dev = [q for q in questions if not q.holdout]
    held = [q for q in questions if q.holdout]
    eval_questions = questions if include_holdout else dev
    if not eval_questions:
        raise ValueError("No questions to evaluate (all marked holdout?).")

    # One reranker/embedder shared across configs.
    reranker = Reranker(settings.rerank_model) if any(c.use_rerank for c in configs) else None
    retriever = Retriever(index, settings, reranker=reranker)

    run_results = []
    for cfg in configs:
        print(f"Running config: {cfg.name} over {len(eval_questions)} questions "
              f"({'with' if generate else 'no'} generation)...")
        run_results.append(run_config(
            retriever, settings, eval_questions, cfg,
            eval_k=eval_k, retrieve_k=retrieve_k, generate=generate,
        ))

    table = format_table(run_results, eval_k, generate)
    print("\n" + table + "\n")
    if held and not include_holdout:
        print(f"({len(held)} holdout questions excluded — run with --holdout to include.)")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_path = os.path.join(out_dir, stamp)
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, "summary.md"), "w") as f:
        f.write(f"# Eval — {index.repo} @ {index.git_sha}\n\n")
        f.write(f"Index: `{index_dir}`  •  embed: `{settings.embed_model}`  •  "
                f"k={eval_k}  •  N={len(eval_questions)}\n\n")
        f.write(table + "\n")
    with open(os.path.join(run_path, "results.json"), "w") as f:
        json.dump(run_results, f, indent=2)
    print(f"Saved run to {run_path}/ (summary.md, results.json)")

    return {"table": table, "runs": run_results, "run_path": run_path}
