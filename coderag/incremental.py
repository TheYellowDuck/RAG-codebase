"""Incremental indexing (outline §7).

Rebuilding the whole index on every commit is the obvious-but-wrong approach.
Instead, reindex only what changed:

  1. Diff current files against the manifest (content-hash) or two git shas.
  2. Delete chunks for modified/deleted files (file_path is a first-class field).
  3. Re-chunk, re-embed, re-insert added/modified files; refresh the graph.
  4. Store the new git_sha.

This turns "reindex 150k LOC" into "reindex the 3 files that changed."
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

from .ingest import FileInfo, discover_files, get_git_sha
from .index import CodeIndex


@dataclass
class UpdateSummary:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    def describe(self) -> str:
        return (f"+{len(self.added)} added, ~{len(self.modified)} modified, "
                f"-{len(self.deleted)} deleted")


def incremental_update(index: CodeIndex, repo_path: str) -> UpdateSummary:
    """Content-hash diff against the manifest. Works with or without git."""
    current = {fi.file_path: fi for fi in discover_files(repo_path)}
    old = index.manifest

    added = [p for p in current if p not in old]
    deleted = [p for p in old if p not in current]
    modified = [p for p in current
                if p in old and current[p].content_sha != old[p].get("content_sha")]

    summary = UpdateSummary(added=added, modified=modified, deleted=deleted)
    if summary.changed == 0:
        return summary

    index.git_sha = get_git_sha(repo_path) or index.git_sha
    index.remove_files(modified + deleted)
    index.add_files([current[p] for p in (added + modified)])
    return summary


def git_incremental_update(index: CodeIndex, repo_path: str,
                           old_sha: str, new_sha: str) -> UpdateSummary:
    """Use `git diff --name-status old..new` to find changes (outline §7 path)."""
    out = subprocess.run(
        ["git", "-C", repo_path, "diff", "--name-status", f"{old_sha}..{new_sha}"],
        capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git diff failed: {out.stderr.strip()}")

    added, modified, deleted = [], [], []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:       # rename: old→new
            deleted.append(parts[1])
            added.append(parts[2])
        elif status.startswith("A"):
            added.append(parts[1])
        elif status.startswith("M"):
            modified.append(parts[1])
        elif status.startswith("D"):
            deleted.append(parts[1])

    # Apply discovery filters/language detection consistently to the change set.
    current = {fi.file_path: fi for fi in discover_files(repo_path)}
    to_remove = [p for p in (modified + deleted) if p in index.manifest]
    to_add = [current[p] for p in (added + modified) if p in current]

    summary = UpdateSummary(
        added=[p for p in added if p in current],
        modified=[p for p in modified if p in current],
        deleted=[p for p in deleted if p in index.manifest],
    )
    if summary.changed == 0:
        return summary

    index.git_sha = new_sha
    index.remove_files(to_remove)
    index.add_files(to_add)
    return summary
