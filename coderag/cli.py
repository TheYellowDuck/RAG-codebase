"""Command-line interface for Code RAG.

  python -m coderag.cli index   <repo_path> [--out DIR] [--window-chunk] [--no-context-header]
  python -m coderag.cli query   "question"  [--index DIR] [-k N] [--dense-only] [--expand-graph] [--retrieve-only]
  python -m coderag.cli update  <repo_path> [--index DIR] [--git OLD NEW]
  python -m coderag.cli eval    [--index DIR] [--golden FILE] [--configs a,b] [-k N] [--generate] [--holdout]
  python -m coderag.cli stats   [--index DIR]
  python -m coderag.cli graph   [--index DIR] (--symbol NAME | --file PATH) [--depth N]
"""
from __future__ import annotations

import argparse
import sys

from .config import Settings


# --------------------------------------------------------------------------- #
def _cmd_index(args) -> int:
    from .index import CodeIndex
    settings = Settings.from_env(index_dir=args.out)
    if args.window_chunk:
        settings.window_chunk = True
    if args.no_context_header:
        settings.use_context_header = False
    if args.embed_model:
        settings.embed_model = args.embed_model
    if args.max_tokens:
        settings.max_chunk_tokens = args.max_tokens

    index = CodeIndex.build(args.repo_path, settings, progress=True)
    out = index.save(args.out)
    print(f"\nIndex saved to {out}")
    _print_stats(index.stats())
    return 0


def _cmd_query(args) -> int:
    from .index import CodeIndex
    from .retrieve import Retriever

    index = CodeIndex.load(args.index)
    settings = index.settings
    retriever = Retriever(index, settings)

    results = retriever.retrieve(
        args.question, k=args.k,
        use_dense=not args.no_dense, use_bm25=not args.no_bm25,
        use_rerank=not args.no_rerank, expand_graph=args.expand_graph,
    )

    print(f"\nRetrieved {len(results)} chunks for: {args.question}\n")
    for r in results:
        tag = f"  [{r.rank}] {r.citation}"
        if r.chunk.symbol_name:
            tag += f"  {r.chunk.symbol_name}"
        if r.source == "graph":
            tag += f"  (graph: {r.relation})"
        else:
            tag += f"  (score {r.score:.3f})"
        print(tag)

    if args.retrieve_only:
        return 0

    # Generation + verification need an LLM provider key (Claude by default).
    try:
        from .llm import get_llm_client
        client = get_llm_client()
    except RuntimeError as e:
        print(f"\n[skipping generation] {e}")
        return 0

    from .generate import generate_answer
    graph_ctx = retriever.graph_context(results) if settings.include_graph_context else None
    print(f"\n{'=' * 70}\nAnswer (via {client.provider}/{client.gen_model}):\n")
    try:
        ans = generate_answer(args.question, results, settings, repo=index.repo,
                              graph_context=graph_ctx, stream=True, client=client)
    except Exception as e:  # network/model errors shouldn't dump a traceback
        print(f"\n\nGeneration failed: {type(e).__name__}: {e}")
        print(f"  provider={client.provider}  model={client.gen_model}")
        print("  Check the model name (CODERAG_GEN_MODEL) and that the provider/endpoint "
              "is reachable\n  (for a local server, confirm it's running at CODERAG_LLM_BASE_URL). "
              "Retrieval above still worked.")
        return 1

    if not args.no_verify:
        from .verify import verify_answer
        print("\n" + "=" * 70 + "\nVerification:")
        report = verify_answer(ans.answer, ans.sources, settings, client=client)
        # Structural check is free and always available — show it first.
        s = report["structural"]
        print(f"  citations: {s['valid_citations']} valid"
              + (f", {s['invalid_citations']} INVALID (no such source!)"
                 if s["invalid_citations"] else ""))
        if s["uncited_claim_sentences"]:
            print(f"  uncited claim sentences: {len(s['uncited_claim_sentences'])}")
        f = report.get("faithfulness")
        if f:
            print(f"  faithfulness: {f['faithfulness']:.2f} "
                  f"({f['n_supported']}/{f['n_claims']} claims supported)")
            for u in f.get("unsupported", []):
                print(f"    ⚠ low confidence: {u}")
        elif "faithfulness_error" in report:
            print(f"  faithfulness: unavailable — judge call failed "
                  f"({report['faithfulness_error']})")
            print("    (use a stronger judge via CODERAG_JUDGE_MODEL; small local "
                  "models often emit invalid JSON here)")
    return 0


def _cmd_update(args) -> int:
    from .index import CodeIndex
    from . import incremental

    index = CodeIndex.load(args.index)
    repo = args.repo_path or index.repo_path
    if not repo:
        print("No repo path known; pass <repo_path>.", file=sys.stderr)
        return 1

    if args.git:
        old, new = args.git
        summary = incremental.git_incremental_update(index, repo, old, new)
    else:
        summary = incremental.incremental_update(index, repo)

    print(f"Incremental update: {summary.describe()}")
    if summary.changed:
        index.save(args.index)
        print(f"Saved updated index to {args.index}")
    else:
        print("Index already up to date.")
    return 0


def _cmd_eval(args) -> int:
    from .eval.run import evaluate
    configs = args.configs.split(",") if args.configs else None
    evaluate(
        args.index, args.golden, config_names=configs, eval_k=args.k,
        generate=args.generate, include_holdout=args.holdout, out_dir=args.out,
    )
    return 0


def _cmd_stats(args) -> int:
    from .index import CodeIndex
    index = CodeIndex.load(args.index)
    _print_stats(index.stats())
    return 0


def _cmd_graph(args) -> int:
    from .index import CodeIndex
    index = CodeIndex.load(args.index)
    graph = index.graph

    targets = []
    if args.symbol:
        targets = [n for n in graph.nodes.values()
                   if n.qualified_name == args.symbol or n.simple_name == args.symbol]
    elif args.file:
        targets = [n for n in graph.nodes.values() if n.file_path == args.file]

    if not targets:
        print("No matching symbol/file in the graph.")
        return 1

    for node in targets[:10]:
        print(f"\n{node.qualified_name} ({node.kind}) — {node.citation}")
        neighbors = graph.neighbors(node.chunk_id, depth=args.depth)
        if not neighbors:
            print("  (no connections)")
        for nb in neighbors:
            print(f"  {nb.relation:>12}  {nb.node.qualified_name}  ({nb.node.citation})")
    return 0


# --------------------------------------------------------------------------- #
def _print_stats(stats: dict) -> None:
    print(f"\nrepo: {stats['repo']}  @ {stats['git_sha'][:12]}")
    print(f"files: {stats['files']}   chunks: {stats['chunks']}")
    print(f"chunks by type: {stats['chunks_by_type']}")
    print(f"chunks by language: {stats['chunks_by_language']}")
    if stats["window_fallback_files"]:
        print(f"window-fallback files (parse failed): {stats['window_fallback_files']}")
    g = stats["graph"]
    print(f"code graph: {g['nodes']} nodes, {g['edges']} edges {g['by_type']}")
    print(f"embed model: {stats['embed_model']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coderag", description="Code-aware RAG over a repo.")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="Index a repository.")
    pi.add_argument("repo_path")
    pi.add_argument("--out", default=Settings().index_dir, help="index output dir")
    pi.add_argument("--window-chunk", action="store_true",
                    help="baseline: pure line-window chunking (no AST)")
    pi.add_argument("--no-context-header", action="store_true",
                    help="ablation: embed code only, no context header")
    pi.add_argument("--embed-model", help="override embedding model id")
    pi.add_argument("--max-tokens", type=int, help="max tokens per chunk")
    pi.set_defaults(func=_cmd_index)

    pq = sub.add_parser("query", help="Ask a question about the indexed repo.")
    pq.add_argument("question")
    pq.add_argument("--index", default=Settings().index_dir)
    pq.add_argument("-k", type=int, default=None, help="final chunks to retrieve")
    pq.add_argument("--no-dense", action="store_true")
    pq.add_argument("--dense-only", dest="no_bm25", action="store_true")
    pq.add_argument("--no-bm25", dest="no_bm25", action="store_true")
    pq.add_argument("--no-rerank", action="store_true")
    pq.add_argument("--expand-graph", action="store_true",
                    help="pull code-graph neighbors into context")
    pq.add_argument("--retrieve-only", action="store_true", help="skip generation")
    pq.add_argument("--no-verify", action="store_true", help="skip faithfulness check")
    pq.set_defaults(func=_cmd_query)

    pu = sub.add_parser("update", help="Incrementally reindex changed files.")
    pu.add_argument("repo_path", nargs="?", default=None)
    pu.add_argument("--index", default=Settings().index_dir)
    pu.add_argument("--git", nargs=2, metavar=("OLD_SHA", "NEW_SHA"),
                    help="use git diff between two shas")
    pu.set_defaults(func=_cmd_update)

    pe = sub.add_parser("eval", help="Run the eval harness and print the table.")
    pe.add_argument("--index", default=Settings().index_dir)
    pe.add_argument("--golden", default="data/golden_questions.jsonl")
    pe.add_argument("--configs", default=None,
                    help="comma list of: dense,hybrid,rerank,graph")
    pe.add_argument("-k", type=int, default=5, help="eval k (file granularity)")
    pe.add_argument("--generate", action="store_true",
                    help="also generate answers + faithfulness/correctness (uses Claude)")
    pe.add_argument("--holdout", action="store_true", help="include holdout questions")
    pe.add_argument("--out", default="eval_runs", help="dir for run artifacts")
    pe.set_defaults(func=_cmd_eval)

    ps = sub.add_parser("stats", help="Print index coverage stats.")
    ps.add_argument("--index", default=Settings().index_dir)
    ps.set_defaults(func=_cmd_stats)

    pg = sub.add_parser("graph", help="Inspect code-graph neighbors of a symbol/file.")
    pg.add_argument("--index", default=Settings().index_dir)
    pg.add_argument("--symbol", help="qualified or simple symbol name")
    pg.add_argument("--file", help="file path (relative to repo root)")
    pg.add_argument("--depth", type=int, default=1)
    pg.set_defaults(func=_cmd_graph)

    return p


def main(argv=None) -> int:
    from .config import load_dotenv
    load_dotenv()  # pick up ANTHROPIC_API_KEY / CODERAG_* from a local .env
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
