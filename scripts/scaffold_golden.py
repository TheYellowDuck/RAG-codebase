#!/usr/bin/env python3
"""Scaffold golden questions from a built index — labels verified by construction.

The hard part of a golden set is labeling `relevant_files` correctly. Instead of
hand-guessing, this derives questions from the index + code graph so the labels are
structurally guaranteed (real call edges / symbol definitions), weighted toward
cross-file. See coderag/eval/scaffold.py for the logic.

  # generate 55 questions (symbol-named templates), append to the golden set:
  python scripts/scaffold_golden.py --index .coderag_index_code --n 55 \
      --start-id 46 >> data/golden_questions.jsonl

  # ...with an LLM pass that rewrites them to be identifier-free (de-confounds BM25).
  # Works with any configured provider — including a FREE local Ollama:
  python scripts/scaffold_golden.py --index .coderag_index_code --n 55 --paraphrase

The symbol-named form advantages lexical/BM25 (the question contains the exact
identifiers). --paraphrase removes that confound for a more rigorous eval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from coderag.index import CodeIndex  # noqa: E402
from coderag.eval.scaffold import build_questions, paraphrase_questions  # noqa: E402

MARKER = "// --- Scaffolded from the code graph"


def _symbols_in(lines) -> set[str]:
    seen: set[str] = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        for s in json.loads(line).get("relevant_symbols", []):
            seen.add(s.split("#")[0])
    return seen


def _existing_symbols(path: str) -> set[str]:
    try:
        with open(path) as f:
            return _symbols_in(f)
    except OSError:
        return set()


def _core_lines(path: str) -> list[str]:
    """The hand-written core: everything before the scaffold marker (or the whole
    file if there's no marker yet)."""
    with open(path) as f:
        lines = f.read().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(MARKER):
            return lines[:i]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=".coderag_index_code")
    ap.add_argument("--n", type=int, default=55)
    ap.add_argument("--start-id", type=int, default=46)
    ap.add_argument("--exclude", default="data/golden_questions.jsonl",
                    help="existing golden set whose symbols to avoid duplicating")
    ap.add_argument("--holdout-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--paraphrase", action="store_true",
                    help="LLM pass: rewrite questions to be identifier/file-free "
                         "(removes the BM25 confound). Needs a configured provider.")
    ap.add_argument("--update", metavar="GOLDEN.jsonl",
                    help="rewrite this golden file IN PLACE and ATOMICALLY: keep its "
                         "hand-written core (before the scaffold marker), regenerate "
                         "the scaffolded block. Writes only on success — safe if "
                         "--paraphrase fails. Replaces the fragile shell splice.")
    args = ap.parse_args()

    from coderag.config import load_dotenv
    load_dotenv()  # honor a local .env (like the CLI does) so the key can live there

    idx = CodeIndex.load(args.index)

    # Exclude the hand-written core's symbols so regeneration reproduces the same
    # scaffolded block (and never duplicates a hand-written question).
    if args.update:
        core = _core_lines(args.update)
        exclude = _symbols_in(core)
    else:
        exclude = _existing_symbols(args.exclude)

    questions = build_questions(
        idx, n=args.n, start_id=args.start_id, exclude_symbols=exclude,
        holdout_every=args.holdout_every, seed=args.seed,
    )
    if args.paraphrase:
        # Resolve the client and paraphrase BEFORE touching the file, so a missing
        # key / provider error aborts without writing anything.
        from coderag.llm import get_llm_client
        questions = paraphrase_questions(questions, get_llm_client(), progress=True)

    rendered = [json.dumps(q) for q in questions]
    if not args.update:
        print("\n".join(rendered))
        return 0

    marker = MARKER + (" (paraphrased, identifier-free)" if args.paraphrase
                       else " (verified labels; see scripts/scaffold_golden.py)") + " ---"
    out = "\n".join(core + [marker] + rendered) + "\n"
    tmp = args.update + ".tmp"
    with open(tmp, "w") as f:        # atomic: write temp, then rename over target
        f.write(out)
    os.replace(tmp, args.update)
    print(f"updated {args.update}: {len(core)} core lines + {len(rendered)} scaffolded "
          f"questions{' (paraphrased)' if args.paraphrase else ''}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
