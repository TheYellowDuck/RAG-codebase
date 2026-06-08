import math

from coderag.eval.metrics import (
    recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, retrieved_files,
    retrieved_symbols, recall_at_k_symbol, reciprocal_rank_symbol, symbol_match,
)
from coderag.eval.bootstrap import bootstrap_ci, paired_bootstrap


def test_recall_precision():
    retrieved = ["a.py", "b.py", "c.py", "d.py"]
    relevant = {"b.py", "d.py"}
    assert recall_at_k(retrieved, relevant, 2) == 0.5     # only b in top-2
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 0.5  # 1 of 2 relevant
    # precision divides by k, not len(top): one relevant file, k=5 -> 1/5 (never >1)
    assert precision_at_k(["b.py"], relevant, 5) == 0.2
    assert recall_at_k([], relevant, 5) == 0.0
    assert recall_at_k(retrieved, set(), 5) == 0.0        # no relevant -> 0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "b.py"], {"b.py"}) == 1 / 3
    assert reciprocal_rank(["b.py", "y"], {"b.py"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"b.py"}) == 0.0


def test_ndcg_perfect_and_partial():
    relevant = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], relevant, 3) == 1.0     # both at top
    # one relevant at rank 1 only -> dcg=1, idcg=1+1/log2(3)
    val = ndcg_at_k(["a", "x", "y"], relevant, 3)
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert abs(val - expected) < 1e-9


def test_retrieved_files_dedup_preserves_order():
    class R:
        def __init__(self, fp):
            self.file_path = fp
    files = retrieved_files([R("a.py"), R("a.py"), R("b.py")])
    assert files == ["a.py", "b.py"]


def test_symbol_match_suffix_and_simple():
    assert symbol_match("APIRouter.add_api_route", "add_api_route")
    assert symbol_match("add_api_route", "APIRouter.add_api_route")
    assert symbol_match("a.foo", "b.foo")     # same simple name
    assert not symbol_match("foo", "bar")


def test_symbol_recall_and_mrr():
    syms = ["mod.helper", "Widget.run", "Widget.scale"]
    assert recall_at_k_symbol(syms, ["helper"], 1) == 1.0
    assert recall_at_k_symbol(syms, ["scale"], 2) == 0.0   # scale at rank 3
    assert reciprocal_rank_symbol(syms, ["run"]) == 0.5


def test_retrieved_symbols_strips_window_suffix():
    class R:
        def __init__(self, s):
            self.symbol_name = s
    out = retrieved_symbols([R("foo#1"), R("foo#2"), R("bar")])
    assert out == ["foo", "bar"]


def test_paired_bootstrap():
    # a strictly beats b on every question -> significant, diff CI excludes 0
    r = paired_bootstrap([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
    assert r["mean_diff"] == 1.0 and r["lo"] > 0 and r["significant"]
    # identical -> no difference, not significant
    r2 = paired_bootstrap([1, 0, 1, 0, 1], [1, 0, 1, 0, 1])
    assert r2["mean_diff"] == 0.0 and not r2["significant"]


def test_bootstrap_ci_bounds():
    ci = bootstrap_ci([1, 1, 1, 1], n_resamples=200)
    assert ci["mean"] == 1.0 and ci["lo"] == 1.0 and ci["hi"] == 1.0
    ci2 = bootstrap_ci([0, 1] * 20, n_resamples=500)
    assert 0.0 <= ci2["lo"] <= ci2["mean"] <= ci2["hi"] <= 1.0
    assert bootstrap_ci([])["mean"] == 0.0
