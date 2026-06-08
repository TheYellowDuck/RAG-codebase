"""Language specs + grammar loading for AST chunking (outline §2.1).

Goal: support *any* language tree-sitter can parse, as precisely as possible.

Grammar loading sources parsers from (in order) a dedicated grammar module, then
`tree-sitter-language-pack` (~165 grammars), then a `tree_sitter_<lang>` module.
Missing grammar → the chunker falls back to line-window chunking.

Two kinds of spec:
  - PRECISE specs — exact node-type sets per language, derived empirically by
    parsing real samples (not guessed). Highest quality; no pattern false hits.
  - A GENERIC spec — for any language without a precise spec, the chunker
    classifies nodes by type-name patterns (see chunker.py). Still produces
    symbol chunks + a graph, just less precisely.

A spec only declares *which node types* are functions / classes / calls / imports.
Name, body, signature, and callee extraction are shared robust routines in
chunker.py, so adding a precise language is just listing node types — the kind of
thing you confirm by parsing a sample and reading the AST.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _fs(*items) -> frozenset:
    return frozenset(items)


@dataclass(frozen=True)
class LanguageSpec:
    language: str
    generic: bool = False
    func_types: frozenset = frozenset()    # function/method definition node types
    class_types: frozenset = frozenset()   # class-like (container) node types
    call_types: frozenset = frozenset()    # call-expression node types
    import_types: frozenset = frozenset()  # import/include statement node types
    name_field: str = "name"               # field tried first for a symbol's name
    body_field: str = "body"               # field tried first for a def's body
    decorated_wrapper: Optional[str] = None  # node that wraps a decorated def
    supports_docstring: bool = False       # Python-style first-statement docstrings


# --------------------------------------------------------------------------- #
# Precise specs (node types verified by parsing samples with each grammar)
# --------------------------------------------------------------------------- #
PYTHON = LanguageSpec(
    "python",
    func_types=_fs("function_definition"),
    class_types=_fs("class_definition"),
    call_types=_fs("call"),
    import_types=_fs("import_statement", "import_from_statement"),
    decorated_wrapper="decorated_definition",
    supports_docstring=True,
)

JAVASCRIPT = LanguageSpec(
    "javascript",
    func_types=_fs("function_declaration", "generator_function_declaration",
                   "method_definition", "function_expression"),
    class_types=_fs("class_declaration"),
    call_types=_fs("call_expression", "new_expression"),
    import_types=_fs("import_statement"),
)

TYPESCRIPT = LanguageSpec(
    "typescript",
    func_types=_fs("function_declaration", "generator_function_declaration",
                   "method_definition", "function_expression",
                   "function_signature", "method_signature"),
    class_types=_fs("class_declaration", "abstract_class_declaration",
                    "interface_declaration", "enum_declaration"),
    call_types=_fs("call_expression", "new_expression"),
    import_types=_fs("import_statement"),
)

GO = LanguageSpec(
    "go",
    func_types=_fs("function_declaration", "method_declaration"),
    call_types=_fs("call_expression"),
    import_types=_fs("import_declaration"),
)

RUST = LanguageSpec(
    "rust",
    func_types=_fs("function_item", "function_signature_item"),
    class_types=_fs("struct_item", "enum_item", "union_item", "trait_item",
                    "impl_item", "mod_item"),
    call_types=_fs("call_expression"),
    import_types=_fs("use_declaration"),
)

RUBY = LanguageSpec(
    "ruby",
    func_types=_fs("method", "singleton_method"),
    class_types=_fs("class", "module"),
    call_types=_fs("call"),
)

JAVA = LanguageSpec(
    "java",
    func_types=_fs("method_declaration", "constructor_declaration"),
    class_types=_fs("class_declaration", "interface_declaration",
                    "enum_declaration", "record_declaration"),
    call_types=_fs("method_invocation", "object_creation_expression"),
    import_types=_fs("import_declaration"),
)

C = LanguageSpec(
    "c",
    func_types=_fs("function_definition"),
    call_types=_fs("call_expression"),
    import_types=_fs("preproc_include"),
)

CPP = LanguageSpec(
    "cpp",
    func_types=_fs("function_definition"),
    class_types=_fs("class_specifier", "struct_specifier", "namespace_definition"),
    call_types=_fs("call_expression"),
    import_types=_fs("preproc_include", "using_declaration"),
)

CSHARP = LanguageSpec(
    "csharp",
    func_types=_fs("method_declaration", "constructor_declaration",
                   "local_function_statement"),
    class_types=_fs("class_declaration", "interface_declaration", "struct_declaration",
                    "enum_declaration", "record_declaration", "namespace_declaration"),
    call_types=_fs("invocation_expression", "object_creation_expression"),
    import_types=_fs("using_directive"),
)

PHP = LanguageSpec(
    "php",
    func_types=_fs("function_definition", "method_declaration"),
    class_types=_fs("class_declaration", "interface_declaration",
                    "trait_declaration", "enum_declaration"),
    call_types=_fs("function_call_expression", "member_call_expression",
                   "scoped_call_expression", "object_creation_expression"),
    import_types=_fs("namespace_use_declaration"),
)

KOTLIN = LanguageSpec(
    "kotlin",
    func_types=_fs("function_declaration"),
    class_types=_fs("class_declaration", "object_declaration"),
    call_types=_fs("call_expression"),
    import_types=_fs("import_header"),
)

SCALA = LanguageSpec(
    "scala",
    func_types=_fs("function_definition"),
    class_types=_fs("class_definition", "object_definition", "trait_definition"),
    call_types=_fs("call_expression"),
    import_types=_fs("import_declaration"),
)

SWIFT = LanguageSpec(
    "swift",
    func_types=_fs("function_declaration"),
    class_types=_fs("class_declaration", "protocol_declaration"),
    call_types=_fs("call_expression"),
    import_types=_fs("import_declaration"),
)

LUA = LanguageSpec(
    "lua",
    func_types=_fs("function_declaration", "function_definition"),
    call_types=_fs("function_call"),
)

BASH = LanguageSpec(
    "bash",
    func_types=_fs("function_definition"),
)

PERL = LanguageSpec(
    "perl",
    func_types=_fs("subroutine_declaration_statement"),
    import_types=_fs("use_statement"),
)

OBJC = LanguageSpec(
    "objc",
    func_types=_fs("function_definition", "method_declaration"),
    class_types=_fs("class_interface", "class_implementation"),
    call_types=_fs("call_expression", "message_expression"),
    import_types=_fs("preproc_include"),
)

PRECISE: dict[str, LanguageSpec] = {
    s.language: s for s in [
        PYTHON, JAVASCRIPT, TYPESCRIPT, GO, RUST, RUBY, JAVA, C, CPP, CSHARP,
        PHP, KOTLIN, SCALA, SWIFT, LUA, BASH, PERL, OBJC,
    ]
}
PRECISE["tsx"] = TYPESCRIPT  # .tsx uses the TypeScript spec

# Dedicated grammar modules to try before the language pack (keeps the lean
# Python-only / js-ts installs working without pulling the whole pack).
_DEDICATED = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
}

# Map an internal language name to the tree-sitter-language-pack name when they
# differ. Unlisted names pass through unchanged.
_PACK_ALIAS = {
    "objc": "objc", "shell": "bash",
}


# --------------------------------------------------------------------------- #
# Extension → language (comprehensive; pack-compatible names)
# --------------------------------------------------------------------------- #
EXT_TO_LANG = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby", ".rake": "ruby",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala", ".sc": "scala",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".m": "objc",
    ".lua": "lua",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".pl": "perl", ".pm": "perl",
    # Generic AST path (no precise spec yet) — still chunked structurally:
    ".r": "r", ".R": "r",
    ".jl": "julia",
    ".hs": "haskell",
    ".ml": "ocaml", ".mli": "ocaml",
    ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".hrl": "erlang",
    ".elm": "elm",
    ".clj": "clojure", ".cljs": "clojure", ".cljc": "clojure",
    ".dart": "dart",
    ".groovy": "groovy", ".gradle": "groovy",
    ".ps1": "powershell", ".psm1": "powershell",
    ".sql": "sql",
    ".vue": "vue", ".svelte": "svelte",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss",
    ".tf": "hcl", ".hcl": "hcl",
    ".proto": "proto",
    ".zig": "zig",
    ".nix": "nix",
}


def get_spec(language: str) -> LanguageSpec:
    """Return the precise spec for a language, or a generic spec for any other."""
    spec = PRECISE.get(language)
    if spec is not None:
        return spec
    return LanguageSpec(language=language, generic=True)


def is_precise(language: str) -> bool:
    return language in PRECISE


# Parsers are cached per language; None means "no grammar available" → fallback.
_PARSERS: dict[str, object] = {}


def _build_parser(language_obj):
    from tree_sitter import Parser
    try:
        return Parser(language_obj)              # tree-sitter >= 0.22
    except TypeError:
        parser = Parser()
        if hasattr(parser, "set_language"):
            parser.set_language(language_obj)     # tree-sitter < 0.22
        else:
            parser.language = language_obj
        return parser


def _load_dedicated(language: str):
    info = _DEDICATED.get(language)
    if not info:
        return None
    module_name, func_name = info
    try:
        import importlib
        from tree_sitter import Language
        module = importlib.import_module(module_name)
        capsule = getattr(module, func_name)()
        try:
            lang_obj = Language(capsule)
        except TypeError:
            lang_obj = Language(capsule, language)
        return _build_parser(lang_obj)
    except Exception:
        return None


def _load_pack(language: str):
    # Use get_language (returns a real tree_sitter.Language) and build the parser
    # with OUR installed tree-sitter, so every grammar yields the same Node API.
    # (The pack's own get_parser returns a parser bound to a vendored, ABI-
    # incompatible core whose Node API differs — avoid it.)
    try:
        from tree_sitter_language_pack import get_language as pack_get_language
        name = _PACK_ALIAS.get(language, language)
        return _build_parser(pack_get_language(name))
    except Exception:
        return None


def _load_module(language: str):
    try:
        import importlib
        from tree_sitter import Language
        module = importlib.import_module(f"tree_sitter_{language}")
        capsule = module.language()
        try:
            lang_obj = Language(capsule)
        except TypeError:
            lang_obj = Language(capsule, language)
        return _build_parser(lang_obj)
    except Exception:
        return None


def get_parser(language: str):
    """Return a tree-sitter Parser for `language`, or None if no grammar is
    available (the chunker then falls back to window chunking)."""
    if language in _PARSERS:
        return _PARSERS[language]
    parser = _load_dedicated(language) or _load_pack(language) or _load_module(language)
    _PARSERS[language] = parser
    return parser


def reset_parser_cache() -> None:
    """Forget cached parsers (incl. cached failures) so get_parser retries — call
    after installing a grammar at runtime."""
    _PARSERS.clear()
