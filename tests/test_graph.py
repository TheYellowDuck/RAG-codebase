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


def test_same_file_call_resolves(make_fileinfo):
    # 'shared' is defined in both files; the caller's OWN-file definition should win
    # (Python scoping) — richer edge-recall without guessing across files.
    src = "def foo():\n    return shared()\n\ndef shared():\n    return 1\n"
    src2 = "def shared():\n    return 2\n"
    pa = chunk_file(make_fileinfo("x.py", src), "repo", "sha")
    pb = chunk_file(make_fileinfo("y.py", src2), "repo", "sha")
    g = CodeGraph.build([pa, pb])
    foo = next(s.chunk_id for s in pa.symbols if s.qualified_name == "foo")
    calls = [n for n in g.neighbors(foo) if n.relation == "calls"]
    assert any(n.node.file_path == "x.py" and n.node.simple_name == "shared"
               for n in calls)


def test_resolve_import_disambiguation():
    # 'helper' is ambiguous (defined in x.py and y.py). The resolver should:
    #  (1) prefer the caller's own file, (2) else the file the caller imports it from,
    #  (3) else refuse to guess.
    from coderag.graph.code_graph import CodeGraph
    by_q, by_s = {}, {"helper": ["cidX", "cidY"]}
    nf = {"cidX": "x.py", "cidY": "y.py"}
    # caller in z.py imports helper from y.py -> resolve to y.py's helper
    assert CodeGraph._resolve("helper", by_q, by_s, prefer_file="z.py",
                              node_file=nf, import_files={"y.py"}) == "cidY"
    # caller is x.py itself -> own-file wins over the import hint
    assert CodeGraph._resolve("helper", by_q, by_s, prefer_file="x.py",
                              node_file=nf, import_files={"y.py"}) == "cidX"
    # ambiguous, no own-file match, no import info -> don't guess
    assert CodeGraph._resolve("helper", by_q, by_s, prefer_file="z.py", node_file=nf) is None


def test_personalized_pagerank_favors_connected(make_fileinfo):
    # foo->shared connected; lonely is isolated. PPR seeded at foo must rank the
    # connected `shared` above the disconnected `lonely`.
    pa = chunk_file(make_fileinfo("x.py", "def foo():\n    return shared()\n\n"
                                  "def shared():\n    return 1\n"), "r", "s")
    pb = chunk_file(make_fileinfo("y.py", "def lonely():\n    return 0\n"), "r", "s")
    g = CodeGraph.build([pa, pb])
    foo = next(s.chunk_id for s in pa.symbols if s.qualified_name == "foo")
    shared = next(s.chunk_id for s in pa.symbols if s.simple_name == "shared")
    lonely = next(s.chunk_id for s in pb.symbols if s.simple_name == "lonely")
    ppr = g.personalized_pagerank([foo])
    assert ppr[shared] > ppr.get(lonely, 0.0)


def test_cross_file_ambiguous_not_linked(make_fileinfo):
    # Caller has NO local definition and the name is ambiguous across two other
    # files -> resolver must not guess.
    pz = chunk_file(make_fileinfo("z.py", "def bar():\n    return shared()\n"), "repo", "sha")
    px = chunk_file(make_fileinfo("x.py", "def shared():\n    return 1\n"), "repo", "sha")
    py = chunk_file(make_fileinfo("y.py", "def shared():\n    return 2\n"), "repo", "sha")
    g = CodeGraph.build([pz, px, py])
    bar = next(s.chunk_id for s in pz.symbols if s.qualified_name == "bar")
    assert all(n.relation != "calls" for n in g.neighbors(bar))
