"""Ingestion: file discovery (outline §1).

Before parsing, decide what's even a candidate — this quietly determines index
quality. We walk the repo, respect .gitignore, skip binaries/lockfiles/vendored
deps/minified/oversized files, detect language, and keep a manifest for
incremental reindex and coverage sanity-checks.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from .languages import EXT_TO_LANG

try:
    import pathspec
except ImportError:  # pragma: no cover - dependency guard
    pathspec = None

# Directories that almost never contain first-party source worth indexing.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".venv", "venv", "env", "node_modules", "vendor",
    "dist", "build", ".next", ".nuxt", "target", ".idea", ".vscode",
    ".gradle", ".terraform", "site-packages", ".eggs", "bower_components",
}

# Lockfiles / generated manifests — high volume, low signal.
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Pipfile.lock", "Cargo.lock", "go.sum", "composer.lock", "Gemfile.lock",
}

# Language detection by extension lives in languages.EXT_TO_LANG (comprehensive,
# tree-sitter-language-pack-compatible names). 18 mainstream languages get precise
# AST specs; every other recognized language gets the generic AST path; the rest
# fall back to line-window chunking.

DEFAULT_MAX_BYTES = 1_000_000   # 1 MB single-file cap (usually generated)
DEFAULT_MAX_LINES = 5_000       # >5k lines is usually generated


@dataclass
class FileInfo:
    abs_path: str
    file_path: str        # relative to repo root — the citation anchor
    language: str
    n_lines: int
    content_sha: str
    source: bytes         # raw bytes (tree-sitter is byte-oriented)


def _content_sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def get_git_sha(repo_path: str) -> Optional[str]:
    """Current commit sha, or None if not a git repo. Used for staleness (§7)."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _load_gitignore(repo_path: str):
    if pathspec is None:
        return None
    patterns: list[str] = []
    gi = os.path.join(repo_path, ".gitignore")
    if os.path.isfile(gi):
        try:
            with open(gi, encoding="utf-8", errors="replace") as f:
                patterns = f.read().splitlines()
        except OSError:
            patterns = []
    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _detect_language(name: str, source: bytes) -> Optional[str]:
    _, ext = os.path.splitext(name)
    if ext:
        return EXT_TO_LANG.get(ext) or EXT_TO_LANG.get(ext.lower())
    # Extensionless: sniff a shebang for scripts.
    head = source[:128].decode("utf-8", "replace")
    if head.startswith("#!"):
        first = head.splitlines()[0]
        if "python" in first:
            return "python"
        if any(s in first for s in ("bash", "/sh", "zsh")):
            return "bash"
        if "node" in first:
            return "javascript"
        if "ruby" in first:
            return "ruby"
        if "perl" in first:
            return "perl"
    return None


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _is_minified(name: str) -> bool:
    return name.endswith((".min.js", ".min.css", "-min.js", "bundle.js"))


def discover_files(
    repo_path: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_lines: int = DEFAULT_MAX_LINES,
    include_languages: Optional[set[str]] = None,
) -> list[FileInfo]:
    """Walk the repo and return the indexable candidate files.

    A quiet failure mode is indexing test fixtures / generated protobufs / huge
    data files, which drowns the real signal. The skip lists above guard against
    the common cases; spot-check coverage with the `stats` CLI command.
    """
    repo_path = os.path.abspath(repo_path)
    spec = _load_gitignore(repo_path)
    results: list[FileInfo] = []

    for root, dirs, files in os.walk(repo_path):
        # Prune skip dirs in-place so os.walk doesn't descend into them.
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]

        for name in files:
            if name in SKIP_FILES or _is_minified(name):
                continue
            abs_path = os.path.join(root, name)
            rel = os.path.relpath(abs_path, repo_path)

            if spec is not None and spec.match_file(rel):
                continue

            try:
                if os.path.getsize(abs_path) > max_bytes:
                    continue
                with open(abs_path, "rb") as f:
                    data = f.read()
            except OSError:
                continue

            if _looks_binary(data):
                continue

            language = _detect_language(name, data)
            if language is None:
                continue
            if include_languages is not None and language not in include_languages:
                continue

            n_lines = data.count(b"\n") + 1
            if n_lines > max_lines:
                continue

            results.append(FileInfo(
                abs_path=abs_path,
                file_path=rel.replace(os.sep, "/"),
                language=language,
                n_lines=n_lines,
                content_sha=_content_sha(data),
                source=data,
            ))

    results.sort(key=lambda fi: fi.file_path)
    return results
