"""Small resilience + logging helpers (production hardening).

Two distinct retry needs, handled in the right place each:

* **LLM API calls** already get exponential backoff from the Anthropic/OpenAI
  SDKs — so we don't reimplement it; we expose `timeout` + `max_retries` on the
  clients (see llm/) so operators can configure them instead of relying on
  undocumented defaults.
* **Local model loading** (sentence-transformers / HF download) has *no* retry —
  a flaky network fails the whole index/query. `with_retry` covers that gap.

Logging is via the stdlib `logging` module (level from CODERAG_LOG_LEVEL, default
WARNING) so retries/timeouts are observable without printing to users' output.
"""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

_LEVEL = os.environ.get("CODERAG_LOG_LEVEL", "WARNING").upper()


def get_logger(name: str = "coderag") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(h)
        log.setLevel(getattr(logging, _LEVEL, logging.WARNING))
        log.propagate = False
    return log


_log = get_logger("coderag.resilience")

# HTTP-ish status codes (or exceptions carrying them) that are worth retrying.
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "temporarily", "overloaded", "unavailable",
    "rate limit", "ratelimit", "connection", "reset by peer", "try again",
)


def is_transient(exc: BaseException) -> bool:
    """Heuristic: retry network/timeouts/5xx/429, fail fast on everything else
    (bad model name, auth error, programming bug). Duck-typed so we don't import
    the provider SDKs here."""
    code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(code, int) and code in _TRANSIENT_STATUS:
        return True
    text = (type(exc).__name__ + " " + str(exc)).lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


def with_retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 0.5,
               max_delay: float = 8.0,
               transient: Optional[Callable[[BaseException], bool]] = None,
               desc: str = "operation") -> T:
    """Call `fn` with exponential backoff + jitter on *transient* failures.

    Retries up to `attempts` times; re-raises immediately on a non-transient
    error, and re-raises the last error once attempts are exhausted. Each retry is
    logged at WARNING so it's visible in production logs."""
    transient = transient or is_transient
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as e:  # noqa: BLE001 - we re-raise unless transient
            if attempt >= attempts or not transient(e):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * (1 + random.random())
            _log.warning("%s failed (attempt %d/%d): %s: %s — retrying in %.2fs",
                         desc, attempt, attempts, type(e).__name__, e, delay)
            time.sleep(delay)
    raise RuntimeError(f"unreachable: {desc} retry loop exited")  # pragma: no cover
