"""Command-line interface for Code RAG.

  python -m coderag.cli index        <repo_path> [--out DIR] [--window-chunk] [--no-context-header]
  python -m coderag.cli query        "question"  [--index DIR] [-k N] [--dense-only] [--expand-graph] [--retrieve-only] [--model NAME]
  python -m coderag.cli chat         [--index DIR] [-k N] [--model NAME]  # interactive REPL
  python -m coderag.cli update       <repo_path> [--index DIR] [--git OLD NEW]
  python -m coderag.cli eval         [--index DIR] [--golden FILE] [--configs a,b] [-k N] [--generate] [--holdout]
  python -m coderag.cli bench        <data.jsonl> [--suite coderag|codesearchnet] [--mode dense|hybrid] [-k N]
  python -m coderag.cli stats        [--index DIR]
  python -m coderag.cli graph        [--index DIR] (--symbol NAME | --file PATH) [--depth N]
  python -m coderag.cli graph-export [--index DIR] [--symbol NAME] [--format html|dot|mermaid] [--out FILE]
  python -m coderag.cli graph-serve  [--index DIR] [--symbol NAME] [--port N]   # live editing + reset
  python -m coderag.cli graph-rebuild [--index DIR]                              # remake graph from records (no re-embed)
  python -m coderag.cli graph-import <edited.json> [--index DIR]                 # apply a static export's edits
  python -m coderag.cli graph-edit   [--index DIR] (--add-edge SRC DST | --remove-edge SRC DST) [--type T]
"""
from __future__ import annotations

import argparse
import sys

from .config import Settings


def _prompt_for_key():
    """Interactively ask for the missing provider key (hidden input), set it for this
    session, and optionally append it to .env. Returns a client, or None if declined."""
    import getpass
    import os
    from .config import LLMConfig
    from .llm import get_llm_client
    var = "OPENAI_API_KEY" if LLMConfig.from_env().provider == "openai" else "ANTHROPIC_API_KEY"
    try:
        key = getpass.getpass(f"\nEnter {var} (leave blank for retrieval-only): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not key:
        return None
    os.environ[var] = key
    try:
        client = get_llm_client()
    except RuntimeError as e:
        print(f"[still can't initialize provider: {e}]")
        return None
    try:
        if input("Save this key to .env for next time? [y/N] ").strip().lower() == "y":
            with open(".env", "a", encoding="utf-8") as f:
                f.write(f"\n{var}={key}\n")
            print("[saved to .env]")
    except (EOFError, KeyboardInterrupt):
        print()
    return client


def _resolve_client(args, interactive: bool = False):
    """Get an LLM client. Applies --model if given; on a missing key, prompts for one
    when run interactively (a TTY), else prints a clear message and returns None."""
    import os
    if getattr(args, "model", None):
        os.environ["CODERAG_GEN_MODEL"] = args.model
    from .llm import get_llm_client
    try:
        return get_llm_client()
    except RuntimeError as e:
        if interactive and sys.stdin.isatty():
            return _prompt_for_key()
        print(f"\n[skipping generation] {e}")
        return None


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

    index = CodeIndex.build(args.repo_path, settings, progress=True,
                            install_grammars=args.install_grammars)
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
        use_hyde=args.hyde, graph_rerank=args.graph_rerank,
        llm_rerank=args.llm_rerank or args.accurate,
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

    # Generation needs an LLM provider; --model overrides the env model, and on a
    # terminal we offer to enter a key if none is set.
    client = _resolve_client(args, interactive=True)
    if client is None:
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


_CHAT_HELP = """commands:
  <question>            ask a question (retrieve + cited answer)
  /retrieve <question>  retrieval only (no LLM call)
  /graph <symbol>       show a symbol's graph neighbors
  /sources              toggle showing retrieved chunks before the answer
  /stats                index stats
  /help                 this help
  /quit                 exit"""


def _chat_answer(question, retriever, settings, index, client, args,
                 show_sources, retrieve_only) -> None:
    results = retriever.retrieve(
        question, k=args.k, use_rerank=not args.no_rerank,
        expand_graph=args.expand_graph)
    if show_sources or retrieve_only or client is None:
        for r in results:
            tag = f"  [{r.rank}] {r.citation}"
            if r.chunk.symbol_name:
                tag += f"  {r.chunk.symbol_name}"
            tag += (f"  (graph: {r.relation})" if r.source == "graph"
                    else f"  (score {r.score:.3f})")
            print(tag)
    if retrieve_only or client is None:
        print()
        return

    from .generate import generate_answer
    from .verify import verify_answer
    graph_ctx = retriever.graph_context(results) if settings.include_graph_context else None
    print()
    try:
        ans = generate_answer(question, results, settings, repo=index.repo,
                              graph_context=graph_ctx, stream=True, client=client)
    except Exception as e:
        print(f"\n[generation failed: {type(e).__name__}: {e}]")
        return
    report = verify_answer(ans.answer, ans.sources, settings, client=client)
    s = report["structural"]
    note = f"\n[citations {s['valid_citations']} valid"
    if s["invalid_citations"]:
        note += f", {s['invalid_citations']} INVALID"
    f = report.get("faithfulness")
    if f:
        note += f"; faithfulness {f['faithfulness']:.2f} ({f['n_supported']}/{f['n_claims']})"
    print(note + "]\n")


def _chat_graph(index, name) -> None:
    g = index.graph
    nodes = [n for n in g.nodes.values()
             if name and (n.qualified_name == name or n.simple_name == name)]
    if not nodes:
        print(f"  no symbol matching '{name}'\n")
        return
    for node in nodes[:3]:
        print(f"  {node.qualified_name} ({node.citation})")
        for nb in g.neighbors(node.chunk_id, depth=1)[:12]:
            print(f"    {nb.relation:>12} → {nb.node.qualified_name} ({nb.node.citation})")
    print()


def _cmd_chat(args) -> int:
    """Interactive REPL over an index (load once, ask many)."""
    from .index import CodeIndex
    from .retrieve import Retriever

    index = CodeIndex.load(args.index)
    settings = index.settings
    retriever = Retriever(index, settings)

    client = _resolve_client(args, interactive=True)
    if client is None:
        print("[retrieval-only mode — answers disabled]")

    st = index.stats()
    banner = f"coderag chat — {index.repo}: {st['chunks']} chunks / {st['files']} files"
    if client:
        banner += f"  ·  answering via {client.provider}/{client.gen_model}"
    print(banner)
    print("Ask a question, or /help for commands. Ctrl-D or /quit to exit.\n")

    show_sources = True
    while True:
        try:
            line = input("coderag› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/quit", "/exit", "quit", "exit"):
            break
        if low in ("/help", "help", "?"):
            print(_CHAT_HELP + "\n")
            continue
        if low == "/stats":
            _print_stats(index.stats())
            continue
        if low == "/sources":
            show_sources = not show_sources
            print(f"[show retrieved sources: {'on' if show_sources else 'off'}]\n")
            continue
        if low.startswith("/graph"):
            _chat_graph(index, line[len("/graph"):].strip())
            continue
        retrieve_only = low.startswith("/retrieve")
        question = line[len("/retrieve"):].strip() if retrieve_only else line
        if question:
            _chat_answer(question, retriever, settings, index, client, args,
                         show_sources, retrieve_only)
    print("bye")
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


def _cmd_bench(args) -> int:
    """Run an *external* retrieval benchmark (validates generalization off our own
    golden set). 'coderag' = CodeRAG-Bench-style (full dense+BM25 retriever);
    'codesearchnet' = CSN docstring->code (embedder only)."""
    if args.suite == "codesearchnet":
        from .eval.codesearchnet import evaluate_codesearchnet
        res = evaluate_codesearchnet(args.data, k=args.k, limit=args.limit)
    else:
        from .eval.coderag_bench import evaluate_coderag_bench
        res = evaluate_coderag_bench(args.data, k=args.k, mode=args.mode, limit=args.limit)
    if not res.get("n"):
        print(f"No examples loaded from {args.data} (check the format).")
        return 1
    label = f"{args.suite}" + (f"/{res['mode']}" if "mode" in res else "")
    metrics = "  ".join(f"{key}={val:.3f}" for key, val in res.items()
                        if isinstance(val, float))
    print(f"[{label}]  n={res['n']}  {metrics}")
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


def _cmd_graph_export(args) -> int:
    """Render the code graph (focused subgraph or degree-capped overview) to an
    interactive editable HTML page, Graphviz DOT, or Mermaid."""
    import os
    from .index import CodeIndex
    from .graph import viz

    index = CodeIndex.load(args.index)
    renderers = {"html": viz.to_html, "dot": viz.to_dot, "mermaid": viz.to_mermaid}
    kwargs = dict(focus=args.symbol, depth=args.depth, max_nodes=args.max_nodes)
    if args.format == "html":
        # focused subgraphs are connected -> layered tree reads well; fragmented
        # overviews look cleaner as a force layout.
        kwargs["layout"] = args.layout or ("hierarchical" if args.symbol else "force")
    text = renderers[args.format](index.graph, **kwargs)
    out = args.out or f"coderag_graph.{ 'mmd' if args.format=='mermaid' else args.format }"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    n_nodes = len(index.graph.nodes)
    scope = f"around '{args.symbol}' (depth {args.depth})" if args.symbol \
        else f"top {args.max_nodes} of {n_nodes} nodes by degree"
    print(f"Wrote {args.format} graph ({scope}) to {os.path.abspath(out)}")
    if args.format == "html":
        print("Open it in a browser — drag/zoom, and the toolbar adds/edits/deletes "
              "edges; 'download edited JSON' saves your changes.")
    return 0


def _cmd_graph_rebuild(args) -> int:
    """Remake the code graph from stored parse records (no re-embedding)."""
    import os
    from .index import CodeIndex
    index = CodeIndex.load(args.index)
    before = index.graph.stats()
    stats = index.rebuild_graph()
    index.graph.save(os.path.join(args.index, "graph.json"))
    print("Rebuilt the graph from stored records (no re-chunking/embedding).")
    print(f"  nodes: {before['nodes']} → {stats['nodes']}")
    print(f"  edges: {before['edges']} → {stats['edges']}  {stats['by_type']}")
    print("  (to pick up source changes on disk, use `update` or `index` instead.)")
    return 0


def _cmd_graph_import(args) -> int:
    """Apply a static HTML export's downloaded JSON back to the index graph."""
    import json
    import os
    from .index import CodeIndex
    from .graph import viz

    index = CodeIndex.load(args.index)
    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)
    before = index.graph.stats()["edges"]
    diff = viz.apply_subgraph_edits(index.graph, data)
    index.graph.save(os.path.join(args.index, "graph.json"))
    print(f"Applied edits from {args.file}: +{diff['added']} edges, -{diff['removed']}.")
    print(f"Graph edges: {before} → {index.graph.stats()['edges']}.")
    return 0


def _cmd_graph_serve(args) -> int:
    """Serve the interactive graph with LIVE, persistent edits + a reset button."""
    from .index import CodeIndex
    from .graph.server import serve_graph
    index = CodeIndex.load(args.index)
    layout = args.layout or ("hierarchical" if args.symbol else "force")
    serve_graph(index, args.index, host=args.host, port=args.port,
                focus=args.symbol, depth=args.depth, max_nodes=args.max_nodes,
                open_browser=not args.no_open, layout=layout)
    return 0


def _cmd_graph_edit(args) -> int:
    """Edit the persisted code graph: add or remove an edge between two symbols."""
    import os
    from .index import CodeIndex

    index = CodeIndex.load(args.index)
    g = index.graph

    def resolve(name):
        ids = g.find_ids(name)
        if len(ids) != 1:
            print(f"  '{name}' resolves to {len(ids)} symbols — be specific "
                  f"(use the qualified name).", file=sys.stderr)
            return None
        return ids[0]

    pair = args.add_edge or args.remove_edge
    src, dst = resolve(pair[0]), resolve(pair[1])
    if not (src and dst):
        return 1
    if args.add_edge:
        g.add_edge(src, dst, args.type)
        print(f"Added edge: {pair[0]} --{args.type}--> {pair[1]}")
    else:
        removed = g.remove_edge(src, dst)
        print(f"Removed {removed} edge(s): {pair[0]} -/-> {pair[1]}")
    g.save(os.path.join(args.index, "graph.json"))   # persist just the graph
    print(f"Saved graph. Now: {g.stats()['edges']} edges.")
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
    pi.add_argument("--install-grammars", action="store_true",
                    help="auto-install tree-sitter grammars for languages found in "
                         "the repo (AST + code graph for all of them)")
    pi.set_defaults(func=_cmd_index)

    pq = sub.add_parser("query", help="Ask a question about the indexed repo.")
    pq.add_argument("question")
    pq.add_argument("--index", default=Settings().index_dir)
    pq.add_argument("-k", type=int, default=None, help="final chunks to retrieve")
    pq.add_argument("--no-dense", action="store_true")
    pq.add_argument("--dense-only", dest="no_bm25", action="store_true")
    pq.add_argument("--no-bm25", dest="no_bm25", action="store_true")
    pq.add_argument("--no-rerank", action="store_true")
    pq.add_argument("--hyde", action="store_true",
                    help="HyDE: draft a hypothetical snippet with the LLM and embed "
                         "it for dense search (needs a provider/key)")
    pq.add_argument("--expand-graph", action="store_true",
                    help="pull code-graph neighbors into context")
    pq.add_argument("--graph-rerank", action="store_true",
                    help="re-rank the fused pool by PageRank-connectivity (helps at scale)")
    pq.add_argument("--llm-rerank", action="store_true",
                    help="listwise LLM reranking of the fused pool (+recall, costs 1 LLM call/query)")
    pq.add_argument("--accurate", action="store_true",
                    help="high-accuracy preset: enables the listwise LLM reranker (the validated "
                         "recall+correctness lever); costs 1 extra LLM call/query")
    pq.add_argument("--retrieve-only", action="store_true", help="skip generation")
    pq.add_argument("--no-verify", action="store_true", help="skip faithfulness check")
    pq.add_argument("--model", default=None,
                    help="generation model for this run (overrides CODERAG_GEN_MODEL)")
    pq.set_defaults(func=_cmd_query)

    pchat = sub.add_parser("chat", help="Interactive REPL over the index (ask many).")
    pchat.add_argument("--index", default=Settings().index_dir)
    pchat.add_argument("-k", type=int, default=None, help="final chunks to retrieve")
    pchat.add_argument("--no-rerank", action="store_true")
    pchat.add_argument("--expand-graph", action="store_true")
    pchat.add_argument("--model", default=None,
                       help="generation model for this run (overrides CODERAG_GEN_MODEL)")
    pchat.set_defaults(func=_cmd_chat)

    pge = sub.add_parser("graph-export", help="Render the code graph (HTML/DOT/Mermaid).")
    pge.add_argument("--index", default=Settings().index_dir)
    pge.add_argument("--symbol", help="focus the subgraph on this symbol (else overview)")
    pge.add_argument("--depth", type=int, default=2, help="BFS depth around the focus symbol")
    pge.add_argument("--format", choices=["html", "dot", "mermaid"], default="html")
    pge.add_argument("--layout", choices=["hierarchical", "force"], default=None,
                     help="HTML layout (default: force for overview, hierarchical for --symbol)")
    pge.add_argument("--max-nodes", type=int, default=90, help="cap for legibility")
    pge.add_argument("--out", help="output file (default coderag_graph.<ext>)")
    pge.set_defaults(func=_cmd_graph_export)

    psv = sub.add_parser("graph-serve", help="Serve the graph with LIVE editing + reset.")
    psv.add_argument("--index", default=Settings().index_dir)
    psv.add_argument("--symbol", help="focus the subgraph on this symbol (else overview)")
    psv.add_argument("--depth", type=int, default=2)
    psv.add_argument("--layout", choices=["hierarchical", "force"], default=None,
                     help="default: force for overview, hierarchical for --symbol")
    psv.add_argument("--max-nodes", type=int, default=90)
    psv.add_argument("--host", default="127.0.0.1")
    psv.add_argument("--port", type=int, default=8000)
    psv.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    psv.set_defaults(func=_cmd_graph_serve)

    prb = sub.add_parser("graph-rebuild", help="Remake the graph from stored records (no re-embed).")
    prb.add_argument("--index", default=Settings().index_dir)
    prb.set_defaults(func=_cmd_graph_rebuild)

    pim = sub.add_parser("graph-import", help="Apply a graph-export's downloaded JSON back to the index.")
    pim.add_argument("file", help="edited JSON downloaded from the graph-export HTML page")
    pim.add_argument("--index", default=Settings().index_dir)
    pim.set_defaults(func=_cmd_graph_import)

    ped = sub.add_parser("graph-edit", help="Add/remove a code-graph edge between symbols.")
    ped.add_argument("--index", default=Settings().index_dir)
    ped.add_argument("--add-edge", nargs=2, metavar=("SRC", "DST"),
                     help="add edge SRC -> DST (by symbol name)")
    ped.add_argument("--remove-edge", nargs=2, metavar=("SRC", "DST"),
                     help="remove edge(s) SRC -> DST")
    ped.add_argument("--type", default="calls", choices=["calls", "imports", "contains"],
                     help="edge type for --add-edge")
    ped.set_defaults(func=_cmd_graph_edit)

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

    pb = sub.add_parser("bench", help="Run an external retrieval benchmark (off our golden set).")
    pb.add_argument("data", help="path to the benchmark JSONL (download from the dataset release)")
    pb.add_argument("--suite", choices=["coderag", "codesearchnet"], default="coderag",
                    help="coderag = full dense+BM25 retriever; codesearchnet = embedder only")
    pb.add_argument("--mode", choices=["dense", "hybrid", "rerank"], default="hybrid",
                    help="coderag suite: dense (embedder), hybrid (dense+BM25+RRF), "
                         "or rerank (hybrid + cross-encoder)")
    pb.add_argument("-k", type=int, default=10, help="cutoff for recall@k / nDCG@k")
    pb.add_argument("--limit", type=int, default=None, help="cap examples (quick check)")
    pb.set_defaults(func=_cmd_bench)

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
