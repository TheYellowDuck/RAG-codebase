"""Graph visualizer (DOT/Mermaid/HTML export) + editor (add/remove edge)."""
from coderag.ingest.chunker import chunk_file
from coderag.graph import CodeGraph
from coderag.graph import viz


def _graph(make_fileinfo):
    pa = chunk_file(make_fileinfo(
        "x.py", "def foo():\n    return shared()\n\ndef shared():\n    return 1\n"), "r", "s")
    pb = chunk_file(make_fileinfo("y.py", "def helper():\n    return foo()\n"), "r", "s")
    return CodeGraph.build([pa, pb])


def test_to_dot_and_mermaid(make_fileinfo):
    g = _graph(make_fileinfo)
    dot = viz.to_dot(g)
    assert dot.startswith("digraph") and "->" in dot
    mer = viz.to_mermaid(g)
    assert mer.startswith("graph LR") and "-->" in mer


def test_to_html_is_standalone_and_editable(make_fileinfo):
    html = viz.to_html(_graph(make_fileinfo))      # default = hierarchical (clean)
    assert "vis-network" in html           # the rendering lib (CDN)
    assert "manipulation" in html          # the in-browser edit toolbar
    assert "DataSet" in html and "download JSON" in html
    assert "hierarchical" in html and "centralGravity" in html   # tree + tamed force
    assert "__NODES__" not in html and "__LAYOUT__" not in html  # all tokens filled
    # force layout is selectable and starts unfrozen
    assert "layoutForce" in html
    assert "'force'" in viz.to_html(_graph(make_fileinfo), layout="force")


def test_subgraph_focus_includes_neighbors(make_fileinfo):
    nodes, edges = viz.subgraph(_graph(make_fileinfo), focus="foo", depth=1)
    names = {n.simple_name for n in nodes}
    assert "foo" in names
    assert "shared" in names               # foo -> shared, 1 hop


def test_rebuild_graph_from_records(sample_repo, embedder):
    from coderag.index import CodeIndex
    from coderag.config import Settings
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    orig_edges = idx.graph.stats()["edges"]
    orig_nodes = idx.graph.stats()["nodes"]
    assert orig_edges > 0
    # wipe the graph entirely, then remake it from the stored parse records
    idx.graph.out_edges.clear(); idx.graph.in_edges.clear(); idx.graph.nodes.clear()
    assert idx.graph.stats()["edges"] == 0
    stats = idx.rebuild_graph()
    assert stats["edges"] == orig_edges and stats["nodes"] == orig_nodes


def test_apply_subgraph_edits_add_and_remove(make_fileinfo):
    g = _graph(make_fileinfo)
    foo, = g.find_ids("foo")
    shared, = g.find_ids("shared")
    helper, = g.find_ids("helper")
    # Edited export over {foo, shared, helper}: drop foo->shared, add foo->helper.
    data = {"nodes": [{"id": foo}, {"id": shared}, {"id": helper}],
            "edges": [{"from": foo, "to": helper, "label": "calls"}]}  # shared edge omitted
    diff = viz.apply_subgraph_edits(g, data)
    assert diff["added"] == 1 and diff["removed"] >= 1
    out = {(d, t) for d, t in g.out_edges.get(foo, [])}
    assert (helper, "calls") in out and (shared, "calls") not in out


def test_apply_subgraph_edits_leaves_outside_edges_untouched(make_fileinfo):
    g = _graph(make_fileinfo)              # has a helper -> foo edge outside an empty view
    before = g.stats()["edges"]
    viz.apply_subgraph_edits(g, {"nodes": [], "edges": []})   # empty view touches nothing
    assert g.stats()["edges"] == before


def test_graph_edit_add_remove_roundtrip(make_fileinfo):
    g = _graph(make_fileinfo)
    foo, = g.find_ids("foo")
    shared, = g.find_ids("shared")
    before = g.stats()["edges"]
    removed = g.remove_edge(foo, shared)
    assert removed >= 1 and g.stats()["edges"] == before - removed
    g.add_edge(foo, shared, "calls")
    assert g.stats()["edges"] == before     # back to where we started
    assert g.find_ids("nonexistent_symbol") == []
