from coderag.ingest.chunker import chunk_file
from coderag.graph import CodeGraph

A = '''\
from pkg.b import helper


def top_level():
    return helper()


class Widget:
    def run(self, x):
        return self.scale(x)

    def scale(self, x):
        return x * 2
'''

B = '''\
def helper():
    return 42
'''


def _build(make_fileinfo):
    pa = chunk_file(make_fileinfo("pkg/a.py", A), "repo", "sha")
    pb = chunk_file(make_fileinfo("pkg/b.py", B), "repo", "sha")
    return CodeGraph.build([pa, pb]), pa, pb


def _node(parse, symbol):
    sym = next(s for s in parse.symbols if s.qualified_name == symbol)
    return sym.chunk_id


def test_call_edge_resolves_crossfile(make_fileinfo):
    g, pa, pb = _build(make_fileinfo)
    top = _node(pa, "top_level")
    rels = {(n.relation, n.node.qualified_name) for n in g.neighbors(top)}
    assert ("calls", "helper") in rels


def test_call_edge_within_class(make_fileinfo):
    g, pa, _ = _build(make_fileinfo)
    run = _node(pa, "Widget.run")
    rels = {(n.relation, n.node.qualified_name) for n in g.neighbors(run)}
    assert ("calls", "Widget.scale") in rels
    assert ("contained_by", "Widget") in rels


def test_contains_edge(make_fileinfo):
    g, pa, _ = _build(make_fileinfo)
    widget = _node(pa, "Widget")
    contained = {n.node.qualified_name for n in g.neighbors(widget)
                 if n.relation == "contains"}
    assert {"Widget.run", "Widget.scale"} <= contained


def test_import_edge(make_fileinfo):
    g, pa, _ = _build(make_fileinfo)
    module = _node(pa, "pkg/a.py")
    imported = {n.node.qualified_name for n in g.neighbors(module)
                if n.relation == "imports"}
    assert "helper" in imported


def test_remove_file_drops_nodes_and_edges(make_fileinfo):
    g, pa, _ = _build(make_fileinfo)
    before = g.stats()["nodes"]
    g.remove_file("pkg/b.py")
    assert g.stats()["nodes"] < before
    # top_level's call to helper is now dangling -> gone
    top = _node(pa, "top_level")
    assert all(n.node.qualified_name != "helper" for n in g.neighbors(top))


def test_save_load_roundtrip(make_fileinfo, tmp_path):
    g, pa, _ = _build(make_fileinfo)
    path = str(tmp_path / "graph.json")
    g.save(path)
    g2 = CodeGraph.load(path)
    assert g2.stats()["nodes"] == g.stats()["nodes"]
    assert g2.stats()["edges"] == g.stats()["edges"]


def test_ambiguous_name_not_linked(make_fileinfo):
    # Two distinct symbols share a simple name -> resolver must not guess.
    src = (
        "def foo():\n    return shared()\n\n"
        "def shared():\n    return 1\n"
    )
    src2 = "def shared():\n    return 2\n"
    pa = chunk_file(make_fileinfo("x.py", src), "repo", "sha")
    pb = chunk_file(make_fileinfo("y.py", src2), "repo", "sha")
    g = CodeGraph.build([pa, pb])
    foo = next(s.chunk_id for s in pa.symbols if s.qualified_name == "foo")
    # 'shared' is ambiguous across files -> no calls edge created
    assert all(n.relation != "calls" for n in g.neighbors(foo))
