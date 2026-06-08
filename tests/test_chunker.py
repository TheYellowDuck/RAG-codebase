import pytest

from coderag.ingest.chunker import chunk_file

PY = '''\
import os
from pkg.b import helper

CONST = 1


def top_level():
    """Top level fn."""
    return helper()


class Widget:
    """A widget."""

    def run(self, x):
        return self.scale(x)

    def scale(self, x):
        return x * 2
'''


def _by_symbol(parse):
    return {c.symbol_name: c for c in parse.chunks}


def test_python_ast_chunking_emits_expected_symbols(make_fileinfo):
    fi = make_fileinfo("pkg/a.py", PY)
    parse = chunk_file(fi, "repo", "sha")
    by = _by_symbol(parse)
    assert parse.parsed is True
    assert "top_level" in by and by["top_level"].symbol_type == "function"
    assert "Widget" in by and by["Widget"].symbol_type == "class"
    assert "Widget.run" in by and by["Widget.run"].symbol_type == "method"
    assert "Widget.scale" in by
    assert "pkg/a.py" in by and by["pkg/a.py"].symbol_type == "module"


def test_class_summary_lists_method_signatures(make_fileinfo):
    fi = make_fileinfo("pkg/a.py", PY)
    by = _by_symbol(chunk_file(fi, "repo", "sha"))
    summary = by["Widget"].code
    assert "class Widget" in summary
    assert "def run" in summary and "def scale" in summary
    assert "return x * 2" not in summary  # bodies are not in the summary


def test_method_context_header_has_class_and_signature(make_fileinfo):
    fi = make_fileinfo("pkg/a.py", PY)
    by = _by_symbol(chunk_file(fi, "repo", "sha"))
    header = by["Widget.run"].context_header
    assert "File: pkg/a.py" in header
    assert "Class: Widget" in header
    assert "def run" in header


def test_embed_text_includes_header_by_default(make_fileinfo):
    fi = make_fileinfo("pkg/a.py", PY)
    by = _by_symbol(chunk_file(fi, "repo", "sha"))
    c = by["top_level"]
    assert c.embed_text.startswith("File: pkg/a.py")
    assert c.code in c.embed_text


def test_no_context_header_embeds_code_only(make_fileinfo):
    fi = make_fileinfo("pkg/a.py", PY)
    by = _by_symbol(chunk_file(fi, "repo", "sha", use_context_header=False))
    assert by["top_level"].embed_text == by["top_level"].code


def test_calls_and_imports_recorded(make_fileinfo):
    fi = make_fileinfo("pkg/a.py", PY)
    parse = chunk_file(fi, "repo", "sha")
    callees = {name for _, name in parse.calls}
    assert "helper" in callees   # top_level() calls helper()
    assert "scale" in callees    # run() calls self.scale()
    imported = {name for _, name in parse.imports}
    assert "helper" in imported  # from pkg.b import helper


def test_decorated_function_includes_decorator_in_span(make_fileinfo):
    src = "@deco\ndef decorated():\n    return 1\n"
    fi = make_fileinfo("d.py", src)
    by = _by_symbol(chunk_file(fi, "repo", "sha"))
    assert "decorated" in by
    assert by["decorated"].code.startswith("@deco")


def test_oversized_function_is_windowed(make_fileinfo):
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    src = f"def big():\n{body}\n    return 0\n"
    fi = make_fileinfo("big.py", src)
    parse = chunk_file(fi, "repo", "sha", max_tokens=20)
    windowed = [c for c in parse.chunks if c.symbol_name and "#" in c.symbol_name]
    assert len(windowed) >= 2
    # signature is carried in every window's header
    assert all("def big" in c.context_header for c in windowed)


def test_window_chunk_mode_produces_only_window_chunks(make_fileinfo):
    fi = make_fileinfo("pkg/a.py", PY)
    parse = chunk_file(fi, "repo", "sha", window_chunk=True, window_lines=5)
    assert parse.parsed is False
    assert {c.symbol_type for c in parse.chunks} == {"window"}


def test_unparsable_python_falls_back_to_window(make_fileinfo):
    fi = make_fileinfo("broken.py", "def (((:\n  ???\n")
    parse = chunk_file(fi, "repo", "sha")
    assert parse.chunks  # nothing silently dropped


def test_unknown_language_falls_back_to_window(make_fileinfo):
    fi = make_fileinfo("notes.txt", "hello\nworld\n", language="text")
    parse = chunk_file(fi, "repo", "sha")
    assert parse.parsed is False
    assert parse.chunks


def test_generic_chunker_on_python_grammar(make_fileinfo):
    # Force the pattern-based (generic) path against the real Python grammar —
    # this is how every non-precise language is chunked.
    parse = chunk_file(make_fileinfo("g.py", PY), "repo", "sha", force_generic=True)
    by = _by_symbol(parse)
    assert "top_level" in by and by["top_level"].symbol_type == "function"
    assert "Widget" in by and by["Widget"].symbol_type == "class"
    assert "Widget.run" in by and by["Widget.run"].symbol_type == "method"
    callees = {name for _, name in parse.calls}
    assert {"helper", "scale"} & callees   # calls extracted generically


def test_go_generic_chunking_if_pack_available(make_fileinfo):
    pytest.importorskip("tree_sitter_language_pack")
    src = (
        'package main\n\nimport "fmt"\n\n'
        'func Hello() string {\n  return greet()\n}\n\n'
        'func greet() string {\n  return fmt.Sprintf("hi")\n}\n'
    )
    parse = chunk_file(make_fileinfo("main.go", src, language="go"), "repo", "sha")
    names = {c.symbol_name for c in parse.chunks}
    assert parse.parsed is True
    assert "Hello" in names and "greet" in names
    callees = {name for _, name in parse.calls}
    assert "greet" in callees


def test_ruby_generic_chunking_if_pack_available(make_fileinfo):
    pytest.importorskip("tree_sitter_language_pack")
    src = (
        "class Widget\n  def run(x)\n    scale(x)\n  end\n\n"
        "  def scale(x)\n    x * 2\n  end\nend\n"
    )
    parse = chunk_file(make_fileinfo("w.rb", src, language="ruby"), "repo", "sha")
    names = {c.symbol_name for c in parse.chunks}
    assert "Widget" in names
    assert any(n and n.endswith("run") for n in names)


def test_javascript_ast_chunking_if_grammar_available(make_fileinfo):
    # Skip only if NO JavaScript grammar is available from any source (the
    # dedicated tree-sitter-javascript package OR the language pack) — consistent
    # with how chunk_file actually resolves grammars.
    from coderag.ingest.languages import get_parser
    if get_parser("javascript") is None:
        pytest.skip("no JavaScript grammar installed")
    src = (
        "import {helper} from './b';\n\n"
        "function topLevel() {\n  return helper();\n}\n\n"
        "class Widget {\n  run(x) {\n    return this.scale(x);\n  }\n"
        "  scale(x) {\n    return x * 2;\n  }\n}\n"
    )
    fi = make_fileinfo("a.js", src, language="javascript")
    parse = chunk_file(fi, "repo", "sha")
    names = {c.symbol_name for c in parse.chunks}
    assert parse.parsed is True
    assert "topLevel" in names
    assert "Widget" in names
    assert "Widget.run" in names
