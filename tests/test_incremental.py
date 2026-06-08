import os

from coderag.index import CodeIndex
from coderag.config import Settings
from coderag import incremental


def test_incremental_add_modify_delete(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    pkg = sample_repo / "pkg"

    (pkg / "a.py").write_text("def changed():\n    return 1\n")   # modify
    (pkg / "c.py").write_text("def added():\n    return 2\n")     # add
    os.remove(pkg / "b.py")                                       # delete

    idx.embedder = embedder
    summ = incremental.incremental_update(idx, str(sample_repo))

    assert summ.added == ["pkg/c.py"]
    assert "pkg/a.py" in summ.modified
    assert "pkg/b.py" in summ.deleted
    assert "pkg/b.py" not in idx.manifest
    assert "pkg/c.py" in idx.manifest
    # indexes stay consistent after mutation
    assert len(idx.vector_store) == len(idx.chunks)
    assert len(idx.bm25) == len(idx.chunks)
    # graph no longer references the deleted file
    assert all(n.file_path != "pkg/b.py" for n in idx.graph.nodes.values())


def test_incremental_noop_when_unchanged(sample_repo, embedder):
    idx = CodeIndex.build(str(sample_repo), Settings(), embedder=embedder, progress=False)
    idx.embedder = embedder
    summ = incremental.incremental_update(idx, str(sample_repo))
    assert summ.changed == 0
