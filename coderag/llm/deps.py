"""Auto-install a provider's SDK on demand (based on which key is set).

A client checks its key first, then calls `ensure_sdk(...)`. So we only ever
install the SDK for the provider the user actually configured (e.g. setting
`OPENAI_API_KEY` + `CODERAG_LLM_PROVIDER=openai` pulls in `openai`, not
`anthropic`). It's one small package and is required for that provider to work at
all, so this is on by default; disable with CODERAG_NO_AUTO_INSTALL=1 (then we
raise with the manual pip command instead). Mirrors ingest/grammars.py.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys


def auto_install_enabled() -> bool:
    return os.environ.get("CODERAG_NO_AUTO_INSTALL", "").lower() not in ("1", "true", "yes")


def ensure_sdk(import_name: str, pip_spec: str, *, progress: bool = True):
    """Import and return `import_name`, pip-installing `pip_spec` first if missing.

    Raises RuntimeError with the manual command if the package is absent and
    auto-install is disabled, or if the install fails.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError:
        pass

    if not auto_install_enabled():
        raise RuntimeError(
            f"The '{import_name}' package is not installed. Run: pip install {pip_spec}\n"
            f"(or unset CODERAG_NO_AUTO_INSTALL to let coderag install it for you)."
        )

    if progress:
        print(f"[llm] installing the '{pip_spec}' SDK (one-time) ...", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_spec], check=True)
        importlib.invalidate_caches()  # make the freshly installed package importable now
        return importlib.import_module(import_name)
    except Exception as e:  # pragma: no cover - environment/network dependent
        raise RuntimeError(
            f"Auto-install of '{pip_spec}' failed ({type(e).__name__}: {e}). "
            f"Install it manually: pip install {pip_spec}"
        ) from e
