from coderag.retrieve import rrf


def test_rrf_rewards_agreement():
    # 'b' appears (highish) in BOTH lists; 'a' and 'c' each appear in only one.
    # RRF's whole point: agreement across retrievers beats a single strong hit.
    dense = ["a", "b", "x", "y"]
    lexical = ["c", "b", "z", "w"]
    fused = rrf(dense, lexical, k=60)
    assert fused[0] == "b"
    assert fused.index("b") < fused.index("a")
    assert fused.index("b") < fused.index("c")


def test_rrf_single_list_preserves_order():
    assert rrf(["x", "y", "z"]) == ["x", "y", "z"]


def test_rrf_k_constant_effect():
    # Larger k flattens the contribution differences but ordering by agreement holds.
    fused = rrf(["a", "b"], ["b", "a"], k=1)
    assert set(fused) == {"a", "b"}
