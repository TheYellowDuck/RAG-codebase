"""Ingestion: code-aware chunking (outline §2).

This is the core technical contribution. Prose RAG splits on character/token
windows; that shreds code — half a function retrieves as noise. We chunk on AST
boundaries with tree-sitter: methods become their own chunks, classes get a
summary chunk (signature + docstring + method signatures), oversized functions
are windowed with the signature carried in the context header, and a module-level
chunk captures imports + top-level code. Files we can't parse (no grammar, or a
syntax error) fall back to line-window chunking so nothing is silently dropped.

Two classification modes (see languages.py):
  - PRECISE  (18 mainstream languages — Python, JS, TS, Go, Rust, Ruby, Java, C,
    C++, C#, PHP, Kotlin, Scala, Swift, Lua, Bash, Perl, Objective-C): exact
    node-type sets, best quality.
  - GENERIC  (any other grammar): nodes are classified by type-name *patterns*
    (`*function*`, `*class*/struct/impl/trait/...`, `*call*`, `*import*`), so any
    tree-sitter language yields symbol-level chunks + a code graph with no
    per-language hand-coding. The window fallback guarantees this is never worse
    than line chunking.

Alongside chunks we emit symbol/call/import records so the code graph (graph/)
can link definitions to their callers, callees, and imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..schema import Chunk, chunk_id, make_chunk
from ..tokenization import token_len
from .discovery import FileInfo
from .languages import LanguageSpec, get_parser, get_spec


# --------------------------------------------------------------------------- #
# Records consumed by the code graph
# --------------------------------------------------------------------------- #
@dataclass
class SymbolRecord:
    chunk_id: str
    qualified_name: str
    simple_name: str
    kind: str               # function | method | class | module
    file_path: str
    start_line: int
    end_line: int


@dataclass
class FileParse:
    chunks: list[Chunk] = field(default_factory=list)
    symbols: list[SymbolRecord] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    imports: list[tuple[str, str]] = field(default_factory=list)
    parsed: bool = True     # False if we fell back to window chunking


# --------------------------------------------------------------------------- #
# Generic classification patterns (used when spec.generic is True)
# --------------------------------------------------------------------------- #
_FUNC_RE = re.compile(r"function|method|constructor|subroutine|(?:\bfn\b)")
_CLASS_RE = re.compile(
    r"class|struct|interface|trait|impl|enum|namespace|module|protocol|object|record|mixin|union")
_CALL_RE = re.compile(r"call|invocation")
_IMPORT_RE = re.compile(r"import|include|require|use_declaration|using|package_clause|preproc_include")
_DEF_MARK = re.compile(r"definition|declaration|item|specifier")
_KNOWN_FUNC = {"method", "singleton_method", "method_definition", "function_definition",
               "function_declaration", "func_literal"}
_KNOWN_CLASS = {"class", "module", "object", "interface", "struct", "enum", "trait",
                "impl", "mod_item", "namespace"}
_NAME_TYPES = {"identifier", "type_identifier", "field_identifier", "property_identifier",
               "constant", "name", "scoped_identifier", "simple_identifier",
               "constant_identifier", "word"}
_BODY_TYPES = {"block", "statement_block", "compound_statement", "declaration_list",
               "field_declaration_list", "class_body", "enum_body", "interface_body",
               "enum_variant_list", "body"}


def _generic_is_class(t: str) -> bool:
    if t in _KNOWN_CLASS:
        return True
    return bool(_CLASS_RE.search(t)) and bool(_DEF_MARK.search(t)) and "type" not in t


def _generic_is_func(t: str) -> bool:
    if t in _KNOWN_CLASS:
        return False
    if t in _KNOWN_FUNC:
        return True
    return (bool(_FUNC_RE.search(t)) and bool(_DEF_MARK.search(t))
            and "type" not in t and "call" not in t)


def _kind(node, spec: LanguageSpec) -> Optional[str]:
    t = node.type
    if spec.generic:
        if _generic_is_class(t):
            return "class"
        if _generic_is_func(t):
            return "func"
        return None
    if t in spec.class_types:
        return "class"
    if t in spec.func_types:
        return "func"
    return None


# --------------------------------------------------------------------------- #
# AST helpers (spec-aware: precise fields, generic fallbacks)
# --------------------------------------------------------------------------- #
def _text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _line(node) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _collect(node, types: set[str], out: list) -> None:
    for child in node.children:
        if child.type in types:
            out.append(child)
        _collect(child, types, out)


def _collect_pred(node, pred: Callable[[str], bool], out: list) -> None:
    for child in node.children:
        if pred(child.type):
            out.append(child)
        _collect_pred(child, pred, out)


def _unwrap_definition(child, spec: LanguageSpec):
    """Return (def_node, span_node, kind) or None, handling decorator wrappers."""
    if spec.decorated_wrapper and child.type == spec.decorated_wrapper:
        for inner in child.children:
            k = _kind(inner, spec)
            if k:
                return inner, child, k
        return None
    k = _kind(child, spec)
    if k:
        return child, child, k
    return None


def _body(node, spec: LanguageSpec):
    if not spec.generic and spec.body_field:
        b = node.child_by_field_name(spec.body_field)
        if b is not None:
            return b
    b = node.child_by_field_name("body")
    if b is not None:
        return b
    for c in node.children:
        if c.type in _BODY_TYPES or c.type.endswith(("block", "body")):
            return c
    return None


def _node_name(node, source: bytes, spec: LanguageSpec) -> str:
    """Symbol name: try the spec's name field, then scan for a name-ish child.

    Shared by precise and generic modes. The scan fallback is what lets precise
    specs work for grammars whose name isn't a direct field (C/C++ declarators,
    Kotlin/Rust-impl with no `name` field)."""
    nm = node.child_by_field_name(spec.name_field or "name")
    if nm is not None:
        return _text(nm, source)
    for c in node.children:                 # direct name-ish child
        if c.type in _NAME_TYPES:
            return _text(c, source)
    for c in node.children:                 # one level down (e.g. C declarators)
        for g in c.children:
            if g.type in _NAME_TYPES:
                return _text(g, source)
    return "<anonymous>"


def _signature_line(node, source: bytes, spec: LanguageSpec) -> str:
    body = _body(node, spec)
    if body is not None:
        sig = source[node.start_byte:body.start_byte].decode("utf-8", "replace")
    else:
        # A zero-width / error node can decode to "" → splitlines() == [] → [0]
        # would IndexError out of chunk_file (whose try only wraps parser.parse),
        # crashing the whole file's indexing. Fall back to no signature instead.
        lines = _text(node, source).splitlines()
        sig = lines[0] if lines else ""
    return " ".join(sig.split())


def _docstring(node, source: bytes, spec: LanguageSpec) -> Optional[str]:
    if not spec.supports_docstring:
        return None
    body = _body(node, spec)
    if body is None:
        return None
    for child in body.named_children:
        if child.type == "expression_statement" and child.named_children:
            inner = child.named_children[0]
            if inner.type == "string":
                raw = _text(inner, source).strip().strip('"\'')
                first = raw.splitlines()[0].strip() if raw else ""
                return first or None
        break
    return None


_CALLEE_FIELDS = ("function", "callee", "method", "name", "constructor")
_NAME_FIELDS = ("name", "property", "field", "attribute")


def _callee_name(call_node, source: bytes) -> Optional[str]:
    """Rightmost identifier of a call target: foo() -> 'foo', a.b.c() -> 'c'.

    Language-agnostic: locate the callee expression via any of the common fields
    (function/callee/method/name), then take its trailing name. Works across
    Python's `function`, Ruby's `method`, Java's `name`, Go/Rust member access,
    etc., so precise specs only need to list which node types are calls."""
    callee = None
    for f in _CALLEE_FIELDS:
        n = call_node.child_by_field_name(f)
        if n is not None:
            callee = n
            break
    if callee is None:
        callee = call_node.named_children[0] if call_node.named_children else None
    if callee is None:
        return None
    if callee.type in _NAME_TYPES:
        return _text(callee, source)
    for f in _NAME_FIELDS:                   # member/attribute access -> trailing name
        a = callee.child_by_field_name(f)
        if a is not None and a.type in _NAME_TYPES:
            return _text(a, source)
    ids: list = []                            # last resort: deepest identifier
    _collect(callee, _NAME_TYPES, ids)
    return _text(ids[-1], source) if ids else None


# --------------------------------------------------------------------------- #
# Context header (§2.3)
# --------------------------------------------------------------------------- #
def _header(file_path: str, *, class_name: Optional[str] = None,
            signature: Optional[str] = None, doc: Optional[str] = None,
            label: Optional[str] = None) -> str:
    lines = [f"File: {file_path}"]
    if class_name:
        lines.append(f"Class: {class_name}")
    if label:
        lines.append(label)
    if signature:
        lines.append(signature)
    if doc:
        lines.append(f'"""{doc}"""')
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Chunk emitters
# --------------------------------------------------------------------------- #
def _emit_function(parse, def_node, span_node, qualified, kind, fi, spec, repo,
                   git_sha, use_header, max_tokens):
    code = _text(span_node, source=fi.source)
    start, end = _line(span_node)
    signature = _signature_line(def_node, fi.source, spec)
    doc = _docstring(def_node, fi.source, spec)
    class_name = qualified.rsplit(".", 1)[0] if kind == "method" else None
    header = _header(fi.file_path, class_name=class_name, signature=signature, doc=doc)

    if token_len(code) <= max_tokens:
        chunk = make_chunk(
            repo=repo, file_path=fi.file_path, language=fi.language,
            symbol_name=qualified, symbol_type=kind, start_line=start, end_line=end,
            code=code, context_header=header, git_sha=git_sha,
            use_context_header=use_header,
        )
        parse.chunks.append(chunk)
        symbol_chunk_id = chunk.id
    else:
        windows = _window_lines(code, start, max_tokens)
        symbol_chunk_id = None
        for i, (w_start, w_end, w_code) in enumerate(windows):
            w_header = _header(
                fi.file_path, class_name=class_name, signature=signature, doc=doc,
                label=f"(part {i + 1}/{len(windows)} of {qualified})",
            )
            chunk = make_chunk(
                repo=repo, file_path=fi.file_path, language=fi.language,
                symbol_name=f"{qualified}#{i + 1}", symbol_type=kind,
                start_line=w_start, end_line=w_end, code=w_code,
                context_header=w_header, git_sha=git_sha, use_context_header=use_header,
            )
            parse.chunks.append(chunk)
            if symbol_chunk_id is None:
                symbol_chunk_id = chunk.id

    parse.symbols.append(SymbolRecord(
        chunk_id=symbol_chunk_id, qualified_name=qualified,
        simple_name=qualified.rsplit(".", 1)[-1], kind=kind,
        file_path=fi.file_path, start_line=start, end_line=end,
    ))
    body = _body(def_node, spec) or def_node
    if spec.generic:
        def is_call(t):
            return bool(_CALL_RE.search(t)) and not _DEF_MARK.search(t)
    else:
        call_set = set(spec.call_types)
        def is_call(t):
            return t in call_set
    call_nodes: list = []
    if is_call(body.type):          # expression-bodied fn: the body *is* the call
        call_nodes.append(body)
    _collect_pred(body, is_call, call_nodes)
    for call in call_nodes:
        name = _callee_name(call, fi.source)
        if name:
            parse.calls.append((symbol_chunk_id, name))


def _emit_class_summary(parse, class_node, span_node, qualified, fi, spec, repo,
                        git_sha, use_header):
    signature = _signature_line(class_node, fi.source, spec)
    doc = _docstring(class_node, fi.source, spec)
    start, end = _line(span_node)

    method_sigs: list[str] = []
    body = _body(class_node, spec)
    if body is not None:
        for child in body.children:
            unwrapped = _unwrap_definition(child, spec)
            if unwrapped and unwrapped[2] == "func":
                method_sigs.append(_signature_line(unwrapped[0], fi.source, spec))

    parts = [signature]
    if doc:
        parts.append(f'    """{doc}"""')
    parts.extend(f"    {s}" for s in method_sigs)
    code = "\n".join(parts)
    header = _header(fi.file_path, label=f"Class summary: {qualified}",
                     signature=signature, doc=doc)

    chunk = make_chunk(
        repo=repo, file_path=fi.file_path, language=fi.language,
        symbol_name=qualified, symbol_type="class", start_line=start, end_line=end,
        code=code, context_header=header, git_sha=git_sha, use_context_header=use_header,
    )
    parse.chunks.append(chunk)
    parse.symbols.append(SymbolRecord(
        chunk_id=chunk.id, qualified_name=qualified, simple_name=qualified.rsplit(".", 1)[-1],
        kind="class", file_path=fi.file_path, start_line=start, end_line=end,
    ))


def _emit_module(parse, root, fi, spec, repo, git_sha, use_header, max_tokens):
    top_nodes = [c for c in root.children if _unwrap_definition(c, spec) is None]
    import_nodes: list = []
    if spec.generic:
        _collect_pred(root, lambda t: bool(_IMPORT_RE.search(t)), import_nodes)
    else:
        _collect(root, set(spec.import_types), import_nodes)

    module_chunk_id = chunk_id(fi.file_path, 1, 1)
    if top_nodes:
        snippet = "\n".join(_text(n, fi.source) for n in top_nodes).strip()
        if snippet:
            if token_len(snippet) > max_tokens:
                snippet = snippet[: max_tokens * 4]
            start = top_nodes[0].start_point[0] + 1
            end = top_nodes[-1].end_point[0] + 1
            header = _header(fi.file_path, label=f"Module: {fi.file_path}")
            chunk = make_chunk(
                repo=repo, file_path=fi.file_path, language=fi.language,
                symbol_name=fi.file_path, symbol_type="module",
                start_line=start, end_line=end, code=snippet,
                context_header=header, git_sha=git_sha, use_context_header=use_header,
            )
            parse.chunks.append(chunk)
            module_chunk_id = chunk.id
            parse.symbols.append(SymbolRecord(
                chunk_id=chunk.id, qualified_name=fi.file_path,
                simple_name=fi.file_path.rsplit("/", 1)[-1], kind="module",
                file_path=fi.file_path, start_line=start, end_line=end,
            ))

    for imp in import_nodes:
        for name in _imported_names(imp, fi.source):
            parse.imports.append((module_chunk_id, name))


def _imported_names(imp_node, source: bytes) -> list[str]:
    names: list[str] = []
    nodes: list = []
    _collect(imp_node, {"dotted_name", "identifier", "aliased_import", "string",
                        "scoped_identifier", "qualified_name", "namespace_name"}, nodes)
    for n in nodes:
        if n.type == "aliased_import":
            inner = n.child_by_field_name("name")
            if inner is not None:
                names.append(_text(inner, source))
        elif n.type == "string":
            raw = _text(n, source).strip().strip('"\'`')
            raw = raw.lstrip("./")
            if raw:
                names.append(raw.rsplit("/", 1)[-1])
        else:
            names.append(_text(n, source))
    return [n for n in names if n and n not in ("import", "from", "as", "use")]


# --------------------------------------------------------------------------- #
# Line-window chunking (baseline + fallback path)
# --------------------------------------------------------------------------- #
def _window_lines(code: str, start_line: int, max_tokens: int) -> list[tuple[int, int, str]]:
    lines = code.split("\n")
    windows: list[tuple[int, int, str]] = []
    cur: list[str] = []
    cur_start = start_line
    for offset, line in enumerate(lines):
        cur.append(line)
        if token_len("\n".join(cur)) >= max_tokens:
            windows.append((cur_start, start_line + offset, "\n".join(cur)))
            cur = []
            cur_start = start_line + offset + 1
    if cur:
        windows.append((cur_start, start_line + len(lines) - 1, "\n".join(cur)))
    return windows or [(start_line, start_line + len(lines) - 1, code)]


def _window_chunk_file(fi: FileInfo, repo: str, git_sha: str, use_header: bool,
                       window_lines: int) -> FileParse:
    parse = FileParse(parsed=False)
    text = fi.source.decode("utf-8", "replace")
    lines = text.split("\n")
    for i in range(0, len(lines), window_lines):
        block = lines[i:i + window_lines]
        code = "\n".join(block)
        if not code.strip():
            continue
        start, end = i + 1, i + len(block)
        header = _header(fi.file_path, label=f"Lines {start}-{end} of {fi.file_path}")
        chunk = make_chunk(
            repo=repo, file_path=fi.file_path, language=fi.language,
            symbol_name=None, symbol_type="window", start_line=start, end_line=end,
            code=code, context_header=header, git_sha=git_sha, use_context_header=use_header,
        )
        parse.chunks.append(chunk)
    return parse


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def chunk_file(fi: FileInfo, repo: str, git_sha: str, *,
               use_context_header: bool = True, window_chunk: bool = False,
               window_lines: int = 50, max_tokens: int = 800,
               force_generic: bool = False) -> FileParse:
    """Chunk one file. Supported languages are AST-chunked (precise or generic);
    anything without a grammar, or a parse failure, falls back to line windows.

    `force_generic` forces the pattern-based path even for languages with a
    precise spec — used by the tests to exercise the generic classifier.
    """
    if window_chunk:
        return _window_chunk_file(fi, repo, git_sha, use_context_header, window_lines)

    parser = get_parser(fi.language)
    if parser is None:
        return _window_chunk_file(fi, repo, git_sha, use_context_header, window_lines)
    spec = get_spec(fi.language)
    if force_generic:
        spec = LanguageSpec(language=fi.language, generic=True,
                            supports_docstring=spec.supports_docstring)

    try:
        tree = parser.parse(fi.source)
    except Exception:
        return _window_chunk_file(fi, repo, git_sha, use_context_header, window_lines)

    root = tree.root_node
    if root.has_error and not root.children:
        return _window_chunk_file(fi, repo, git_sha, use_context_header, window_lines)

    parse = FileParse(parsed=True)

    def walk(node, class_path: list[str]):
        for child in node.children:
            unwrapped = _unwrap_definition(child, spec)
            if unwrapped is None:
                continue
            def_node, span_node, kind = unwrapped
            name = _node_name(def_node, fi.source, spec)
            if kind == "class":
                qualified = ".".join(class_path + [name])
                _emit_class_summary(parse, def_node, span_node, qualified, fi, spec,
                                    repo, git_sha, use_context_header)
                walk(_body(def_node, spec) or def_node, class_path + [name])
            else:
                qualified = ".".join(class_path + [name]) if class_path else name
                fn_kind = "method" if class_path else "function"
                _emit_function(parse, def_node, span_node, qualified, fn_kind, fi, spec,
                               repo, git_sha, use_context_header, max_tokens)

    walk(root, [])
    _emit_module(parse, root, fi, spec, repo, git_sha, use_context_header, max_tokens)

    if not parse.chunks:  # markup/data/empty file — still index something
        return _window_chunk_file(fi, repo, git_sha, use_context_header, window_lines)
    return parse
