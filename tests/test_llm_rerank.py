"""Listwise LLM reranking (_llm_rerank): reorder the fused pool by LLM judgment,
fail open on any error. Verified with a stub LLM — no key, no network."""
from coderag.retrieve.retriever import Retriever


class _C:
    def __init__(self, i):
        self.id = i
        self.file_path = f"{i}.py"
        self.start_line = 1
        self.code = f"def {i}(): pass"


def _retriever(llm):
    r = Retriever.__new__(Retriever)      # bypass heavy __init__
    r._llm = llm
    return r


def test_llm_rerank_applies_llm_order():
    class LLM:
        def generate(self, system, user, max_tokens=80):
            class R:
                text = "3, 1, 5"          # pick candidates c, a, e
            return R()
    cands = [_C(x) for x in "abcde"]
    out = [c.id for c in _retriever(LLM())._llm_rerank("q", cands, k=3, pool=5)]
    assert out == ["c", "a", "e"]


def test_llm_rerank_fills_remaining_from_fused_order():
    class LLM:
        def generate(self, system, user, max_tokens=80):
            class R:
                text = "2"                 # only one pick; rest fill from fused order
            return R()
    cands = [_C(x) for x in "abcde"]
    out = [c.id for c in _retriever(LLM())._llm_rerank("q", cands, k=3, pool=5)]
    assert out[0] == "b" and len(out) == 3 and set(out) <= set("abcde")


def test_llm_rerank_fails_open_on_error():
    class LLM:
        def generate(self, *a, **k):
            raise RuntimeError("no provider key")
    cands = [_C(x) for x in "abcde"]
    out = [c.id for c in _retriever(LLM())._llm_rerank("q", cands, k=2, pool=5)]
    assert out == ["a", "b"]              # fused top-k, unchanged
