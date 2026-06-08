"""Live, editable graph server (`graph-serve`).

A tiny localhost HTTP server (stdlib — no deps) that serves the vis-network editor
and makes edits **persistent and live**: adding/deleting an edge in the browser POSTs
to the server, which mutates the graph and saves `graph.json` immediately. A **Reset**
button reverts to the graph as it was when the server started.

Edits are restricted to *edges between existing nodes* (same surface as `graph-edit`);
node ids are chunk ids.
"""
from __future__ import annotations

import json
import os
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import viz

# Uses __TOKEN__ placeholders (str.replace) so the JS keeps normal single braces.
_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>coderag graph (live)</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>html,body{margin:0;height:100%;font-family:sans-serif}#net{height:92vh;border-bottom:1px solid #ddd}
#bar{padding:8px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}button{padding:4px 10px}</style></head>
<body><div id="bar"><b>coderag graph — live edit</b><span>__TITLE__</span>
<span style="color:#5e9bc4">calls</span> <span style="color:#5cab7d">imports</span> <span style="color:#9aa1a9">contains</span>
<input id="q" placeholder="search nodes…" oninput="search()" style="padding:4px;width:150px">
<button onclick="layoutTree()">tree (clean)</button>
<button onclick="layoutForce()">force</button>
<button onclick="reset()">⟲ reset</button>
<span id="msg" style="color:#888">drag to add an edge · select + Delete to remove · edits save live</span></div>
<div id="net"></div>
<script>
const nodes=new vis.DataSet([]), edges=new vis.DataSet([]);
let _layout='__LAYOUT__', net=null;
const ORIG={};                       // original colors, rebuilt from pristine data each load
const el=document.getElementById('net');
function msg(t){document.getElementById('msg').textContent=t;}
const TREE={physics:false,edges:{smooth:{type:'cubicBezier',forceDirection:'horizontal',roundness:0.55}},
  layout:{hierarchical:{enabled:true,direction:'LR',sortMethod:'directed',
    levelSeparation:220,nodeSpacing:130,treeSpacing:210,
    blockShifting:true,edgeMinimization:true,parentCentralization:true}}};
const FORCE={layout:{hierarchical:false},edges:{smooth:{type:'continuous'}},
  physics:{solver:'forceAtlas2Based',minVelocity:0.75,
    stabilization:{enabled:true,iterations:600,fit:true},
    forceAtlas2Based:{gravitationalConstant:-260,centralGravity:0.015,
      springLength:230,springConstant:0.08,damping:0.5,avoidOverlap:1}}};
function post(p,b){return fetch(p,{method:'POST',body:JSON.stringify(b||{})}).then(r=>r.json());}
const MANIP={enable:true,
  addEdge:(d,cb)=>{if(d.from===d.to)return cb(null);
    post('/api/edge',{from:d.from,to:d.to,type:'calls'}).then(r=>{
      d.id=d.from+'|'+d.to+'|calls';d.label='calls';d.color={color:'#5e9bc4'};
      cb(d);msg('added edge ('+r.edges+' total)');});},
  deleteEdge:(d,cb)=>{Promise.all(d.edges.map(id=>{const e=edges.get(id);
      return post('/api/edge/delete',{from:e.from,to:e.to});})).then(()=>{cb(d);msg('deleted');});}};
const BASE={nodes:{shape:'box',shapeProperties:{borderRadius:7},margin:10,borderWidth:1,
    font:{size:14,face:'Helvetica, Arial, sans-serif',color:'#222'},
    color:{border:'rgba(0,0,0,0.18)'},
    shadow:{enabled:true,size:8,x:1,y:2,color:'rgba(0,0,0,0.13)'},widthConstraint:{maximum:200}},
  edges:{arrows:{to:{scaleFactor:0.6}},color:{opacity:0.75},font:{size:11,color:'#888',strokeWidth:3}},
  manipulation:MANIP};
// Recreate the network on each layout switch — setOptions keeps stale positions,
// making layouts non-deterministic ("reorganizes") and force never spread cleanly.
function render(){const o=_layout==='force'?FORCE:TREE;
  if(net) net.destroy();
  net=new vis.Network(el,{nodes,edges},Object.assign({},BASE,o,{edges:Object.assign({},BASE.edges,o.edges)}));
  net.on('stabilizationIterationsDone',()=>{net.setOptions({physics:false});net.fit();});}
function layoutTree(){_layout='tree';render();}
function layoutForce(){_layout='force';render();}
function show(d){nodes.clear();edges.clear();nodes.add(d.nodes);edges.add(d.edges);
  for(const k in ORIG)delete ORIG[k]; d.nodes.forEach(n=>{ORIG[n.id]=n.color;}); render();}
function load(){fetch('/api/graph').then(r=>r.json()).then(show);}
function reset(){post('/api/reset').then(d=>{show(d);msg('reset to original');});}
function search(){const q=document.getElementById('q').value.toLowerCase().trim();
  nodes.update(nodes.get().map(n=>{const m=q&&(n.label||'').toLowerCase().includes(q);
    return {id:n.id,borderWidth:m?3:1,
      color:!q?ORIG[n.id]:(m?{background:ORIG[n.id],border:'#333'}:'#e6e6e6')};}));
  if(!q) return;
  const hits=nodes.get().filter(n=>(n.label||'').toLowerCase().includes(q)).map(n=>n.id);
  if(hits.length===1) net.focus(hits[0],{scale:2.4,animation:{duration:400}});
  else if(hits.length) net.fit({nodes:hits,animation:{duration:400}});}
load();
</script></body></html>"""


def serve_graph(index, index_dir, host="127.0.0.1", port=8000, focus=None,
                depth=2, max_nodes=150, open_browser=True, layout="hierarchical"):
    g = index.graph
    graph_path = os.path.join(index_dir, "graph.json")
    # snapshot edges for reset (edge edits don't touch nodes)
    snap_out = {k: list(v) for k, v in g.out_edges.items()}
    snap_in = {k: list(v) for k, v in g.in_edges.items()}

    def payload() -> dict:
        nodes, edges = viz.subgraph(g, focus, depth, max_nodes)
        return {
            "nodes": [{"id": n.chunk_id, "label": n.simple_name or n.qualified_name,
                       "title": f"{n.qualified_name}\n{n.citation}",
                       "color": viz._KIND_COLOR.get(n.kind, "#333")} for n in nodes],
            "edges": [{"id": f"{s}|{d}|{t}", "from": s, "to": d, "label": t,
                       "arrows": "to", "color": {"color": viz._EDGE_COLOR.get(t, "#333")}}
                      for s, d, t in edges],
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):       # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json_body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/":
                title = f"focus={focus} depth={depth}" if focus else f"top {max_nodes} by degree"
                html = (_HTML.replace("__TITLE__", title)
                        .replace("__LAYOUT__", "force" if layout == "force" else "hierarchical"))
                self._send(200, html, "text/html")
            elif self.path == "/api/graph":
                self._send(200, json.dumps(payload()))
            else:
                self._send(404, "{}")

        def do_POST(self):
            if self.path == "/api/edge":
                d = self._json_body()
                g.add_edge(d["from"], d["to"], d.get("type", "calls"))
                g.save(graph_path)
                self._send(200, json.dumps({"ok": True, "edges": g.stats()["edges"]}))
            elif self.path == "/api/edge/delete":
                d = self._json_body()
                removed = g.remove_edge(d["from"], d["to"])
                g.save(graph_path)
                self._send(200, json.dumps({"ok": True, "removed": removed}))
            elif self.path == "/api/reset":
                g.out_edges = defaultdict(list, {k: list(v) for k, v in snap_out.items()})
                g.in_edges = defaultdict(list, {k: list(v) for k, v in snap_in.items()})
                g.save(graph_path)
                self._send(200, json.dumps(payload()))
            else:
                self._send(404, "{}")

    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Serving editable graph at {url}")
    print(f"  edits save live to {graph_path}; Ctrl-C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        srv.server_close()
