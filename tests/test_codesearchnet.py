"""CodeSearchNet adapter — loader + retrieval metrics on a synthetic sample."""
import json

from coderag.eval.codesearchnet import load_codesearchnet, evaluate_codesearchnet


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_loader_tolerates_field_names(tmp_path):
    p = tmp_path / "csn.jsonl"
    _write(p, [
        {"query": "add numbers", "code": "def add(a, b): return a + b"},
        {"func_documentation_string": "reverse string",
         "func_code_string": "def rev(s): return s[::-1]"},
        {"nope": 1},  # skipped (no query/code)
    ])
    ex = load_codesearchnet(str(p))
    assert len(ex) == 2
    assert ex[0]["query"] == "add numbers" and "def add" in ex[0]["code"]


def test_evaluate_recall_and_mrr(tmp_path, embedder):
    # each query's tokens overlap its own code most -> stub embedder ranks it first
    rows = [
        {"query": "compute factorial", "code": "def factorial(n): compute factorial recursively"},
        {"query": "reverse a string", "code": "def reverse(s): reverse a string chars"},
        {"query": "sort a list", "code": "def sort(x): sort a list ascending order"},
    ]
    p = tmp_path / "csn.jsonl"
    _write(p, rows)
    out = evaluate_codesearchnet(str(p), embedder=embedder, k=3)
    assert out["n"] == 3
    assert out["recall@3"] == 1.0       # gold snippet always in top-3
    assert out["mrr"] > 0.5
