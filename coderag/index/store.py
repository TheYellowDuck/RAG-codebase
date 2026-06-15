"""CodeIndex — the central object tying chunks, vectors, BM25, and the graph.

Build path (§2.4): discover → chunk → embed `embed_text` → add to the dense and
BM25 indexes (both keyed by chunk.id so fusion is trivial) → build the code
graph. A manifest of {file_path: (content_sha, git_sha, language, n_lines,
chunk_ids)} powers incremental reindex (§7) and coverage sanity-checks.

Per-file graph records (symbols/calls/imports) are persisted so the graph can be
rebuilt on incremental updates without re-chunking unchanged files — only changed
files get re-embedded.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

from ..config import Settings
from ..schema import Chunk
from ..embed import Embedder
from ..ingest import FileInfo, chunk_file, discover_files, get_git_sha
from ..ingest.chunker import FileParse, SymbolRecord
from ..graph import CodeGraph
from .vector_store import make_vector_store, load_vector_store
from .bm25_index import BM25Index

INDEX_VERSION = 1


class CodeIndex:
    def __init__(self, settings: Settings, embedder: Optional[Embedder] = None):
        self.settings = settings
        self.embedder = embedder or Embedder.from_settings(settings)
        self.repo = ""
        self.repo_path = ""
        self.git_sha = "nogit"
        self.chunks: dict[str, Chunk] = {}
        self.manifest: dict[str, dict] = {}
        self.graph_records: dict[str, dict] = {}  # file_path -> {symbols, calls, imports}
        self.vector_store = make_vector_store(settings)
        self.bm25 = BM25Index()
        self.graph = CodeGraph()

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, repo_path: str, settings: Settings,
              embedder: Optional[Embedder] = None, progress: bool = True,
              install_grammars: bool = False) -> "CodeIndex":
        idx = cls(settings, embedder)
        idx.repo_path = os.path.abspath(repo_path)
        idx.repo = os.path.basename(idx.repo_path.rstrip("/")) or "repo"
        idx.git_sha = get_git_sha(idx.repo_path) or "nogit"

        files = discover_files(idx.repo_path)
        if progress:
            print(f"Discovered {len(files)} candidate files in {idx.repo}")

        # Make grammars available for the languages actually present (so AST +
        # graph cover all of them); installs on demand only when asked.
        if not settings.window_chunk:
            from ..ingest.grammars import ensure_grammars, auto_install_enabled
            ensure_grammars({fi.language for fi in files},
                            auto_install=auto_install_enabled(install_grammars),
                            progress=progress)

        parses: list[FileParse] = []
        for fi in files:
            parse = chunk_file(
                fi, idx.repo, idx.git_sha,
                use_context_header=settings.use_context_header,
                window_chunk=settings.window_chunk,
                window_lines=settings.window_lines,
                max_tokens=settings.max_chunk_tokens,
            )
            parses.append(parse)
            idx._register_parse(fi, parse)

        if progress:
            print(f"Produced {len(idx.chunks)} chunks. Embedding...")
        idx._embed_and_index(list(idx.chunks.values()), progress=progress)

        idx.graph = CodeGraph.build(parses)
        if progress:
            g = idx.graph.stats()
            print(f"Code graph: {g['nodes']} nodes, {g['edges']} edges {g['by_type']}")
        return idx

    def _register_parse(self, fi: FileInfo, parse: FileParse) -> None:
        chunk_ids = []
        for chunk in parse.chunks:
            self.chunks[chunk.id] = chunk
            chunk_ids.append(chunk.id)
        self.manifest[fi.file_path] = {
            "content_sha": fi.content_sha,
            "git_sha": self.git_sha,
            "language": fi.language,
            "n_lines": fi.n_lines,
            "chunk_ids": chunk_ids,
            "parsed": parse.parsed,
        }
        self.graph_records[fi.file_path] = {
            "symbols": [asdict(s) for s in parse.symbols],
            "calls": [list(c) for c in parse.calls],
            "imports": [list(i) for i in parse.imports],
        }

    def _embed_and_index(self, chunks: list[Chunk], progress: bool = False) -> None:
        if not chunks:
            return
        texts = [c.embed_text for c in chunks]
        ids = [c.id for c in chunks]
        vectors = self.embedder.encode(texts, show_progress=progress)
        self.vector_store.add(ids, vectors)
        self.bm25.add(ids, texts)

    # ------------------------------------------------------------------ #
    # Incremental mutation (used by incremental.py, §7)
    # ------------------------------------------------------------------ #
    def remove_files(self, file_paths: list[str]) -> None:
        doomed_ids: set[str] = set()
        for fp in file_paths:
            entry = self.manifest.pop(fp, None)
            if entry:
                doomed_ids.update(entry["chunk_ids"])
            self.graph_records.pop(fp, None)
            self.graph.remove_file(fp)
        for cid in doomed_ids:
            self.chunks.pop(cid, None)
        if doomed_ids:
            self.vector_store.remove(doomed_ids)
            self.bm25.remove(doomed_ids)

    def add_files(self, file_infos: list[FileInfo]) -> None:
        """(Re)chunk, embed, and index a set of files; refresh their graph."""
        new_chunks: list[Chunk] = []
        for fi in file_infos:
            parse = chunk_file(
                fi, self.repo, self.git_sha,
                use_context_header=self.settings.use_context_header,
                window_chunk=self.settings.window_chunk,
                window_lines=self.settings.window_lines,
                max_tokens=self.settings.max_chunk_tokens,
            )
            self._register_parse(fi, parse)
            new_chunks.extend(parse.chunks)
        self._embed_and_index(new_chunks)
        # Cross-file edges depend on the full name index, so rebuild from records.
        self.rebuild_graph()

    def rebuild_graph(self) -> dict:
        """Remake the code graph from the stored per-file parse records
        (symbols/calls/imports) — NO re-chunking or re-embedding. Use this to apply
        graph/resolver changes (e.g. import-aware resolution) to an existing index,
        or to regenerate the graph after manual edits got messy. Does NOT pick up
        source changes on disk — use `update`/`index` for that. Returns graph stats."""
        self.graph = CodeGraph.build(self._all_parses())
        return self.graph.stats()

    def _all_parses(self) -> list[FileParse]:
        parses: list[FileParse] = []
        for rec in self.graph_records.values():
            fp = FileParse()
            fp.symbols = [SymbolRecord(**s) for s in rec["symbols"]]
            fp.calls = [tuple(c) for c in rec["calls"]]
            fp.imports = [tuple(i) for i in rec["imports"]]
            parses.append(fp)
        return parses

    # ------------------------------------------------------------------ #
    # Access helpers
    # ------------------------------------------------------------------ #
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        by_lang: dict[str, int] = {}
        for c in self.chunks.values():
            by_type[c.symbol_type] = by_type.get(c.symbol_type, 0) + 1
            by_lang[c.language] = by_lang.get(c.language, 0) + 1
        unparsed = [fp for fp, m in self.manifest.items() if not m.get("parsed", True)]
        return {
            "repo": self.repo,
            "git_sha": self.git_sha,
            "files": len(self.manifest),
            "chunks": len(self.chunks),
            "chunks_by_type": by_type,
            "chunks_by_language": by_lang,
            "window_fallback_files": len(unparsed),
            "graph": self.graph.stats(),
            "embed_model": self.settings.embed_model,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, dir_path: Optional[str] = None) -> str:
        dir_path = dir_path or self.settings.index_dir
        os.makedirs(dir_path, exist_ok=True)

        with open(os.path.join(dir_path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for chunk in self.chunks.values():
                f.write(json.dumps(chunk.to_dict()) + "\n")

        with open(os.path.join(dir_path, "manifest.json"), "w") as f:
            json.dump(self.manifest, f)
        with open(os.path.join(dir_path, "graph_records.json"), "w") as f:
            json.dump(self.graph_records, f)
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            json.dump({
                "version": INDEX_VERSION,
                "repo": self.repo,
                "repo_path": self.repo_path,
                "git_sha": self.git_sha,
                "embed_model": self.settings.embed_model,
                "settings": self.settings.to_dict(),
            }, f)

        self.vector_store.save(dir_path)
        self.bm25.save(dir_path)
        self.graph.save(os.path.join(dir_path, "graph.json"))
        return dir_path

    @classmethod
    def load(cls, dir_path: str, embedder: Optional[Embedder] = None) -> "CodeIndex":
        meta_path = os.path.join(dir_path, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"No index found at {dir_path}. Build one first with "
                f"`python -m coderag.cli index <repo_path>`."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        settings = Settings(**meta["settings"])
        idx = cls(settings, embedder)
        idx.repo = meta["repo"]
        idx.repo_path = meta.get("repo_path", "")
        idx.git_sha = meta.get("git_sha", "nogit")

        with open(os.path.join(dir_path, "chunks.jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = Chunk.from_dict(json.loads(line))
                    idx.chunks[chunk.id] = chunk
        with open(os.path.join(dir_path, "manifest.json")) as f:
            idx.manifest = json.load(f)
        gr_path = os.path.join(dir_path, "graph_records.json")
        if os.path.isfile(gr_path):
            with open(gr_path) as f:
                idx.graph_records = json.load(f)

        idx.vector_store = load_vector_store(settings, dir_path)
        idx.bm25 = BM25Index.load(dir_path)
        idx.graph = CodeGraph.load(os.path.join(dir_path, "graph.json"))
        return idx
