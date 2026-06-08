#!/usr/bin/env python3
"""Fetch HumanEval and convert it to a CodeRAG-Bench-style retrieval file.

HumanEval (openai/human-eval, MIT) is 164 programming problems, each with a
`prompt` (signature + NL docstring) and a `canonical_solution` (the body). We
treat prompt->canonical_solution as a retrieval task — exactly the
"programming-solutions" flavor of CodeRAG-Bench — so `coderag bench` can produce a
real, reproducible number without bundling third-party data in the repo.

Usage:
    python scripts/fetch_humaneval_bench.py            # -> data/humaneval_bench.jsonl
    coderag bench data/humaneval_bench.jsonl --suite coderag --mode hybrid -k 10
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/humaneval_bench.jsonl"


def main() -> int:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"downloading {URL} ...")
    raw = gzip.decompress(urllib.request.urlopen(URL).read()).decode("utf-8")
    n = 0
    with open(OUT, "w", encoding="utf-8") as o:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            o.write(json.dumps({"id": d["task_id"],
                                "query": d["prompt"],
                                "code": d["canonical_solution"]}) + "\n")
            n += 1
    print(f"wrote {n} records -> {OUT}")
    print("now run: coderag bench %s --suite coderag --mode hybrid -k 10" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
