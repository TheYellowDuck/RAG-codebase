"""Asymmetric-embedder prefix support (coderag/embed/embedder.py).

Modern retrieval embedders (e5/bge) need query vs passage prefixes or recall
craters. These verify the prefixes are inferred from the model name, applied to
the right side, and overridable — without loading any model (a fake captures the
text actually sent to encode)."""
import numpy as np

from coderag.config import Settings
from coderag.embed import Embedder
from coderag.embed.embedder import infer_prefixes


def test_infer_prefixes_by_model_name():
    assert infer_prefixes("intfloat/e5-base-v2") == ("query: ", "passage: ")
    assert infer_prefixes("BAAI/bge-base-en-v1.5")[0].startswith("Represent")
    # symmetric / unknown models get no prefix (no behavior change)
    assert infer_prefixes("flax-sentence-embeddings/st-codesearch-distilroberta-base") == ("", "")


class _FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **kw):
        self.seen = list(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


def test_prefix_applied_to_correct_side():
    e = Embedder("intfloat/e5-base-v2")
    e._model = _FakeModel()
    e.encode(["def foo(): ..."], is_query=False)
    assert e._model.seen[0].startswith("passage: ")
    e.encode_query("how do I foo")
    assert e._model.seen[0].startswith("query: ")


def test_no_prefix_for_symmetric_model():
    e = Embedder("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    e._model = _FakeModel()
    e.encode(["x = 1"], is_query=False)
    assert e._model.seen[0] == "x = 1"        # untouched


def test_from_settings_infers_and_overrides():
    e = Embedder.from_settings(Settings(embed_model="intfloat/e5-base-v2"))
    assert (e.query_prefix, e.doc_prefix) == ("query: ", "passage: ")
    e2 = Embedder.from_settings(
        Settings(embed_model="whatever", embed_query_prefix="Q: ", embed_doc_prefix="D: "))
    assert (e2.query_prefix, e2.doc_prefix) == ("Q: ", "D: ")
