"""On-demand tree-sitter grammar provisioning.

So the AST chunker + code graph work for *every* language found in a repo, not
just the ones whose grammar happens to be installed. When scanning, we collect
the languages present; any that lack a loadable parser can be enabled by
installing `tree-sitter-language-pack` (one package, ~165 grammars).

Auto-installing packages mid-scan is a side effect (network + writes to the
environment), so it's **opt-in**: `index --install-grammars`, or
CODERAG_AUTO_INSTALL_GRAMMARS=1. Without it, missing-grammar languages simply
window-chunk (and we print how to enable AST support).
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

from .languages import get_parser, reset_parser_cache

_PACK = "tree-sitter-language-pack"


def missing_grammars(languages) -> set[str]:
    """Languages present in the repo that currently have no loadable grammar."""
    return {lang for lang in set(languages) if get_parser(lang) is None}


def auto_install_enabled(flag: bool = False) -> bool:
    if flag:
        return True
    return os.environ.get("CODERAG_AUTO_INSTALL_GRAMMARS", "").lower() in ("1", "true", "yes")


def ensure_grammars(languages, *, auto_install: bool = False, progress: bool = True) -> dict:
    """Make grammars available for `languages`. Returns a small report.

    With auto_install, pip-installs the language pack (covering all of them) and
    re-checks. Without it, just reports which languages will window-chunk.
    """
    languages = set(languages)
    missing = missing_grammars(languages)
    if not missing:
        return {"installed": False, "gained": set(), "still_missing": set()}

    if not auto_install:
        if progress:
            print(f"[grammars] {len(missing)} language(s) have no grammar and will "
                  f"line-window-chunk: {sorted(missing)}")
            print("[grammars] re-run with --install-grammars (or "
                  "CODERAG_AUTO_INSTALL_GRAMMARS=1) for AST + graph on all of them.")
        return {"installed": False, "gained": set(), "still_missing": missing}

    if progress:
        print(f"[grammars] installing {_PACK} to enable AST for {sorted(missing)} ...")
    installed = _pip_install(_PACK)
    reset_parser_cache()
    still_missing = missing_grammars(languages)
    gained = missing - still_missing
    if progress:
        if gained:
            print(f"[grammars] AST + graph now enabled for: {sorted(gained)}")
        if still_missing:
            print(f"[grammars] no grammar available (still window-chunking): "
                  f"{sorted(still_missing)}")
    return {"installed": installed, "gained": gained, "still_missing": still_missing}


def _pip_install(package: str) -> bool:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package],
                       check=True)
        importlib.invalidate_caches()  # let the fresh package be importable now
        return True
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[grammars] install failed ({type(e).__name__}: {e}); "
              f"those languages will window-chunk.")
        return False
