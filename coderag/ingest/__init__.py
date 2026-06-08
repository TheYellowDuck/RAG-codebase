"""Ingestion: file discovery (§1) and code-aware chunking (§2)."""
from .discovery import FileInfo, discover_files, get_git_sha
from .chunker import FileParse, chunk_file

__all__ = ["FileInfo", "discover_files", "get_git_sha", "FileParse", "chunk_file"]
