"""Code-graph visualization + export.

Turn the (possibly huge) code graph into something a human can look at:
  - a focused subgraph around a symbol (BFS to `depth`), or a degree-capped
    overview of the whole graph (so a 40k-node graph stays legible);
  - rendered as Graphviz DOT, Mermaid, or a standalone **interactive, editable**
    HTML page (vis-network) — drag/zoom, and the manipulation toolbar lets you
    add/edit/delete edges in-browser and download the edited graph as JSON.
"""
from __future__ import annotations

import json
from collections import defaultdict

# Soft pastel palette: light fills (black text stays readable), lighter edges.
_EDGE_COLOR = {"calls": "#5e9bc4", "imports": "#5cab7d", "contains": "#b6bcc4"}
_KIND_COLOR = {"class": "#f6bd60", "method": "#b5a7e6", "function": "#8ecae6",
               "module": "#cdd2d8"}


def subgraph(graph, focus: "str | None" = None, depth: int = 2,
             max_nodes: int = 150):
    """Return (nodes, edges) to draw. With `focus` (symbol name) → BFS neighborhood;
    otherwise the highest-degree nodes (a legible overview of a large graph)."""
    if focus:
        seeds = [n.chunk_id for n in graph.nodes.values()
                 if n.qualified_name == focus or n.simple_name == focus]
        keep = set(seeds)
        for sid in seeds:
            for nb in graph.neighbors(sid, depth=depth):
                keep.add(nb.chunk_id)
    else:
        degree: dict = defaultdict(int)
        for src, edges in graph.out_edges.items():
            for dst, _ in edges:
                degree[src] += 1
                degree[dst] += 1
        keep = set(sorted(degree, key=degree.get, reverse=True)[:max_nodes])
        if not keep:
            keep = set(list(graph.nodes)[:max_nodes])
    keep &= set(graph.nodes)
    if len(keep) > max_nodes:
        keep = set(list(keep)[:max_nodes])

    nodes = [graph.nodes[cid] for cid in keep]
    edges = []
    for src in keep:
        for dst, etype in graph.out_edges.get(src, []):
            if dst in keep:
                edges.append((src, dst, etype))
    return nodes, edges


def to_dot(graph, focus=None, depth=2, max_nodes=150) -> str:
    nodes, edges = subgraph(graph, focus, depth, max_nodes)
    lines = ["digraph code {", '  rankdir=LR;', '  node [shape=box, style=rounded];']
    for n in nodes:
        color = _KIND_COLOR.get(n.kind, "#333333")
        label = (n.simple_name or n.qualified_name).replace('"', "'")
        lines.append(f'  "{n.chunk_id}" [label="{label}", color="{color}"];')
    for src, dst, etype in edges:
        lines.append(f'  "{src}" -> "{dst}" [color="{_EDGE_COLOR.get(etype, "#333")}", '
                     f'label="{etype}"];')
    lines.append("}")
    return "\n".join(lines)


def to_mermaid(graph, focus=None, depth=2, max_nodes=150) -> str:
    nodes, edges = subgraph(graph, focus, depth, max_nodes)
    idx = {n.chunk_id: f"n{i}" for i, n in enumerate(nodes)}
    lines = ["graph LR"]
    for n in nodes:
        label = (n.simple_name or n.qualified_name).replace('"', "'")
        lines.append(f'  {idx[n.chunk_id]}["{label}"]')
    for src, dst, etype in edges:
        if src in idx and dst in idx:
            lines.append(f"  {idx[src]} -->|{etype}| {idx[dst]}")
    return "\n".join(lines)


# Template uses __TOKEN__ placeholders (filled by str.replace) so the JS keeps its
# normal single braces — much less error-prone than escaping every brace for .format.
_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>coderag graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>html,body{margin:0;height:100%;font-family:sans-serif}#net{height:92vh;border-bottom:1px solid #ddd}
#bar{padding:8px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}button{padding:4px 10px}</style></head>
<body><div id="bar"><b>coderag graph</b><span>__TITLE__</span>
<span style="color:#5e9bc4">calls</span> <span style="color:#5cab7d">imports</span> <span style="color:#9aa1a9">contains</span>
<input id="q" placeholder="search nodes…" oninput="search()" style="padding:4px;width:150px">
<button onclick="layoutTree()">tree (clean)</button>
<button onclick="layoutForce()">force</button>
<button onclick="dl()">download JSON</button>
<span style="color:#888">· scratch view — edits aren't saved; download then <code>coderag graph-import &lt;file&gt;</code>, or use <code>graph-serve</code>.</span></div>
<div id="net"></div>
<script>
const nodes=new vis.DataSet(__NODES__);
const edges=new vis.DataSet(__EDGES__);
const el=document.getElementById('net');
// Original colors captured ONCE from pristine data — bulletproof search restore.
const ORIG={}; nodes.get().forEach(n=>{ORIG[n.id]=n.color;});
const BASE={nodes:{shape:'box',shapeProperties:{borderRadius:7},margin:10,borderWidth:1,
    font:{size:14,face:'Helvetica, Arial, sans-serif',color:'#222'},
    color:{border:'rgba(0,0,0,0.18)'},
    shadow:{enabled:true,size:8,x:1,y:2,color:'rgba(0,0,0,0.13)'},
    widthConstraint:{maximum:200}},
  edges:{arrows:{to:{scaleFactor:0.6}},color:{opacity:0.75},font:{size:11,color:'#888',strokeWidth:3}},
  manipulation:{enable:true}};
// TREE = deterministic layered layout (static). FORCE = ForceAtlas2 (spreads well).
const TREE={physics:false,edges:{smooth:{type:'cubicBezier',forceDirection:'horizontal',roundness:0.55}},
  layout:{hierarchical:{enabled:true,direction:'LR',sortMethod:'directed',
    levelSeparation:220,nodeSpacing:130,treeSpacing:210,
    blockShifting:true,edgeMinimization:true,parentCentralization:true}}};
const FORCE={layout:{hierarchical:false},edges:{smooth:{type:'continuous'}},
  physics:{solver:'forceAtlas2Based',minVelocity:0.75,
    stabilization:{enabled:true,iterations:600,fit:true},
    forceAtlas2Based:{gravitationalConstant:-260,centralGravity:0.015,
      springLength:230,springConstant:0.08,damping:0.5,avoidOverlap:1}}};
let net=null;
// Recreate the network on each switch — vis keeps stale positions if you only
// setOptions, which makes layouts non-deterministic ("reorganizes" / force balls).
function render(opts){
  if(net) net.destroy();
  net=new vis.Network(el,{nodes,edges},
    Object.assign({},BASE,opts,{edges:Object.assign({},BASE.edges,opts.edges)}));
  net.on('stabilizationIterationsDone',()=>{net.setOptions({physics:false});net.fit();});
}
function layoutTree(){render(TREE);}
function layoutForce(){render(FORCE);}
// search: matches keep their OWN color (with a dark outline), the rest dim to gray.
function search(){const q=document.getElementById('q').value.toLowerCase().trim();
  nodes.update(nodes.get().map(n=>{const m=q&&(n.label||'').toLowerCase().includes(q);
    return {id:n.id,borderWidth:m?3:1,
      color:!q?ORIG[n.id]:(m?{background:ORIG[n.id],border:'#333'}:'#e6e6e6')};}));
  if(!q) return;
  const hits=nodes.get().filter(n=>(n.label||'').toLowerCase().includes(q)).map(n=>n.id);
  if(hits.length===1) net.focus(hits[0],{scale:2.4,animation:{duration:400}});
  else if(hits.length) net.fit({nodes:hits,animation:{duration:400}});}
render('__LAYOUT__'==='force'?FORCE:TREE);   // default
function dl(){const d={nodes:nodes.get(),edges:edges.get()};
  const b=new Blob([JSON.stringify(d,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='graph_edited.json';a.click();}
</script></body></html>"""


def apply_subgraph_edits(graph, data: dict) -> dict:
    """Apply an HTML-export's downloaded JSON back to the graph.

    The export is a *subgraph view*, so we reconcile edges only WITHIN its node set
    (edges among other nodes are untouched): add edges present in `data` but missing,
    remove edges missing from `data` but present. Returns {added, removed}."""
    node_ids = {n["id"] for n in data.get("nodes", [])} & set(graph.nodes)
    desired = {(e["from"], e["to"], e.get("label") or "calls")
               for e in data.get("edges", [])
               if e.get("from") in node_ids and e.get("to") in node_ids}
    current = {(s, d, t) for s in node_ids
               for d, t in graph.out_edges.get(s, []) if d in node_ids}
    to_add, to_remove = desired - current, current - desired
    for s, d, t in to_add:
        graph.add_edge(s, d, t)
    for s, d, t in to_remove:
        graph.remove_edge(s, d, t)
    return {"added": len(to_add), "removed": len(to_remove)}


def to_html(graph, focus=None, depth=2, max_nodes=150, layout="hierarchical") -> str:
    """layout: 'hierarchical' (clean layered, static — default) or 'force'."""
    nodes, edges = subgraph(graph, focus, depth, max_nodes)
    vnodes = [{"id": n.chunk_id,
               "label": n.simple_name or n.qualified_name,
               "title": f"{n.qualified_name}\n{n.citation}",
               "color": _KIND_COLOR.get(n.kind, "#333333")} for n in nodes]
    vedges = [{"from": s, "to": d, "label": t, "arrows": "to",
               "color": {"color": _EDGE_COLOR.get(t, "#333")}} for s, d, t in edges]
    title = f"focus={focus} depth={depth}" if focus else f"top {len(nodes)} nodes by degree"
    return (_HTML.replace("__NODES__", json.dumps(vnodes))
                 .replace("__EDGES__", json.dumps(vedges))
                 .replace("__TITLE__", title)
                 .replace("__LAYOUT__", "force" if layout == "force" else "hierarchical"))
