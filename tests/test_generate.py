from coderag.schema import make_chunk
from coderag.retrieve import RetrievedChunk
from coderag.generate import assemble_context


def _chunk(path, start, end, code="x"):
    return make_chunk(
        repo="r", file_path=path, language="python", symbol_name=f"{path}:{start}",
        symbol_type="function", start_line=start, end_line=end, code=code,
        context_header="h", git_sha="s",
    )


def test_assembly_numbers_and_renders_sources():
    a = RetrievedChunk(_chunk("a.py", 1, 10), 1.0, 1)
    b = RetrievedChunk(_chunk("b.py", 1, 5), 0.9, 2)
    block, sources = assemble_context([a, b], token_budget=100_000)
    assert [s.n for s in sources] == [1, 2]
    assert "[1] a.py:1-10" in block
    assert "[2] b.py:1-5" in block


def test_assembly_dedupes_contained_span():
    outer = RetrievedChunk(_chunk("a.py", 1, 20), 1.0, 1)
    inner = RetrievedChunk(_chunk("a.py", 5, 10), 0.9, 2)  # contained in outer
    other = RetrievedChunk(_chunk("b.py", 1, 5), 0.8, 3)
    _, sources = assemble_context([outer, inner, other], token_budget=100_000)
    files = [s.file_path for s in sources]
    assert files == ["a.py", "b.py"]  # inner dropped as overlapping


def test_assembly_respects_token_budget():
    big = RetrievedChunk(_chunk("a.py", 1, 50, code="line " * 500), 1.0, 1)
    small = RetrievedChunk(_chunk("b.py", 1, 2, code="x"), 0.9, 2)
    _, sources = assemble_context([big, small], token_budget=1)
    assert len(sources) == 1  # first always added; second exceeds budget
