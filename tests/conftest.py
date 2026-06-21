"""Shared pytest fixtures. A stub embedder keeps the suite torch-free and fast."""
import hashlib

import numpy as np
import pytest

from coderag.ingest.discovery import FileInfo
from coderag.tokenization import code_tokens


def stable_hash(tok: str) -> int:
    """Process-independent token hash. Python's builtin hash() is randomized per
    run (PYTHONHASHSEED), so using it for the dimension mapping made the stub
    embedder non-deterministic — occasional ranking ties flipped and flaked CI.
    A fixed digest keeps the embedder actually deterministic, as promised."""
    return int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")


class StubEmbedder:
    """Deterministic bag-of-code-tokens hashing embedder — no torch, no network.

    Same interface as coderag.embed.Embedder: encode / encode_query / dim.
    Token-overlap drives similarity, which is enough to exercise the pipeline.
    """
    model_name = "stub"
    dim = 64

    def encode(self, texts, batch_size=64, show_progress=False, is_query=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in code_tokens(t):
                out[i, stable_hash(tok) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out

    def encode_query(self, q):
        return self.encode([q])[0]


@pytest.fixture
def embedder():
    return StubEmbedder()


@pytest.fixture
def make_fileinfo():
    def _make(path, src, language="python"):
        data = src.encode("utf-8")
        return FileInfo(
            abs_path=path, file_path=path, language=language,
            n_lines=data.count(b"\n") + 1,
            content_sha=hashlib.sha1(data).hexdigest(), source=data,
        )
    return _make


SAMPLE_A = '''\
import os
from pkg.b import helper

GREETING = "hi"


def top_level():
    """Top level fn."""
    return helper()


class Widget:
    """A widget."""

    def run(self, x):
        return self.scale(x)

    def scale(self, x):
        return x * 2
'''

SAMPLE_B = '''\
def helper():
    return 42
'''


@pytest.fixture
def sample_repo(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text(SAMPLE_A)
    (pkg / "b.py").write_text(SAMPLE_B)
    return tmp_path
