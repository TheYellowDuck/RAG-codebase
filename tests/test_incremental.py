import os
import subprocess

from coderag.index import CodeIndex
from coderag.config import Settings
from coderag import incremental


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _commit_all(repo, msg) -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t.t", "-c", "user.name=t",
         "commit", "-q", "-m", msg, "--no-gpg-sign")
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


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


def test_git_incremental_update_add_modify_delete_rename(tmp_path, embedder):
    repo = tmp_path
    _git(repo, "init", "-q")
    pkg = repo / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("def a():\n    return 1\n")
    (pkg / "b.py").write_text("def b():\n    return 2\n")
    (pkg / "old.py").write_text("def old_one():\n    return 3\n")
    old_sha = _commit_all(repo, "init")

    idx = CodeIndex.build(str(repo), Settings(), embedder=embedder, progress=False)
    idx.embedder = embedder

    (pkg / "a.py").write_text("def a():\n    return 99\n")    # modify
    (pkg / "c.py").write_text("def c():\n    return 4\n")      # add
    os.remove(pkg / "b.py")                                    # delete
    os.rename(pkg / "old.py", pkg / "renamed.py")             # rename (identical content)
    new_sha = _commit_all(repo, "change")

    summ = incremental.git_incremental_update(idx, str(repo), old_sha, new_sha)

    # rename is handled as delete-old + add-new whether or not git reports it as 'R'
    assert "pkg/c.py" in summ.added and "pkg/renamed.py" in summ.added
    assert "pkg/a.py" in summ.modified
    assert "pkg/b.py" in summ.deleted and "pkg/old.py" in summ.deleted
    # manifest + git_sha reflect the change set
    assert "pkg/b.py" not in idx.manifest and "pkg/old.py" not in idx.manifest
    assert "pkg/c.py" in idx.manifest and "pkg/renamed.py" in idx.manifest
    assert idx.git_sha == new_sha
    # indexes stay consistent; the graph drops removed files
    assert len(idx.vector_store) == len(idx.chunks) == len(idx.bm25)
    assert all(n.file_path not in ("pkg/b.py", "pkg/old.py")
               for n in idx.graph.nodes.values())


def test_git_incremental_update_noop_on_same_sha(tmp_path, embedder):
    repo = tmp_path
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("def a():\n    return 1\n")
    sha = _commit_all(repo, "init")
    idx = CodeIndex.build(str(repo), Settings(), embedder=embedder, progress=False)
    idx.embedder = embedder
    summ = incremental.git_incremental_update(idx, str(repo), sha, sha)
    assert summ.changed == 0
