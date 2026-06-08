from coderag.config import Settings
from coderag.schema import make_chunk
from coderag.retrieve import RetrievedChunk
from coderag.generate import assemble_context
from coderag.generate.generator import _compact_code, _gate_by_score


def _chunk(path, start, end, code=None):
    code = code if code is not None else f"def f_{path}_{start}(): return {start}"
    return make_chunk(
        repo="r", file_path=path, language="python", symbol_name=f"{path}:{start}",
        symbol_type="function", start_line=start, end_line=end, code=code,
        context_header="h", git_sha="s",
    )


def _rc(path, start, end, score=1.0, code=None):
    return RetrievedChunk(_chunk(path, start, end, code), score, 1)


def test_assembly_numbers_and_renders_sources():
    block, sources = assemble_context([_rc("a.py", 1, 10), _rc("b.py", 1, 5)], Settings())
    assert [s.n for s in sources] == [1, 2]
    assert "[1] a.py:1-10" in block
    assert "[2] b.py:1-5" in block


def test_assembly_dedupes_contained_span():
    src = [_rc("a.py", 1, 20), _rc("a.py", 5, 10), _rc("b.py", 1, 5)]  # middle contained
    _, sources = assemble_context(src, Settings())
    assert [s.file_path for s in sources] == ["a.py", "b.py"]


def test_assembly_respects_token_budget():
    big = _rc("a.py", 1, 50, code="line " * 500)
    small = _rc("b.py", 1, 2, code="zzz")
    _, sources = assemble_context([big, small], Settings(context_token_budget=1))
    assert len(sources) == 1  # first always added; second exceeds budget


def test_content_dedup_drops_identical_code():
    # Same code in two different files -> dedup keeps one.
    a = _rc("a.py", 1, 3, code="def helper(): return 1")
    b = _rc("b.py", 9, 11, code="def helper(): return 1")
    _, on = assemble_context([a, b], Settings(dedup_sources=True))
    _, off = assemble_context([a, b], Settings(dedup_sources=False))
    assert len(on) == 1
    assert len(off) == 2


def test_merge_adjacent_same_file():
    a = _rc("a.py", 1, 10, code="block one")
    b = _rc("a.py", 11, 20, code="block two")  # contiguous (gap 1)
    _, merged = assemble_context([a, b], Settings(merge_adjacent_sources=True, dedup_sources=False))
    assert len(merged) == 1
    assert merged[0].start_line == 1 and merged[0].end_line == 20
    assert "block one" in merged[0].code and "block two" in merged[0].code


def test_compact_code_collapses_blank_lines():
    assert _compact_code("a\n\n\n\nb\n   \nc\n") == "a\n\nb\n\nc"


def test_gate_drops_negative_rerank_tail_but_keeps_min():
    results = [_rc("a.py", 1, 2, score=3.0), _rc("b.py", 1, 2, score=-1.0),
              _rc("c.py", 1, 2, score=-2.0), _rc("d.py", 1, 2, score=-3.0)]
    # min_sources=1: keep rank0 (i<1) + any >=0 -> just the first
    assert len(_gate_by_score(results, min_sources=1)) == 1
    # min_sources=3: always keep at least 3 even if negative
    assert len(_gate_by_score(results, min_sources=3)) == 3