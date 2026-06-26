"""discover_files filtering (ingest/discovery.py) — the layer that quietly decides
index quality: gitignore, binary / minified / lockfile skips, skip-dirs, size and
line caps, and shebang language detection for extensionless scripts."""
import pytest

from coderag.ingest.discovery import discover_files


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def test_discover_files_skips_noise_and_detects_shebang(tmp_path):
    _write(tmp_path / "keep.py", "def f():\n    return 1\n")
    _write(tmp_path / "app.min.js", "var x=1;")               # minified → skip
    _write(tmp_path / "package-lock.json", "{}\n")            # lockfile → skip
    _write(tmp_path / "blob.bin", b"\x00\x01\x02binary")      # NUL byte → binary skip
    _write(tmp_path / "node_modules" / "dep.js", "x=1;\n")    # skip-dir → not descended
    _write(tmp_path / "data.unknownext", "no language here")  # unknown ext → skip
    _write(tmp_path / "runme", "#!/usr/bin/env python\nprint(1)\n")  # shebang → python

    found = {fi.file_path: fi for fi in discover_files(str(tmp_path))}

    assert found["keep.py"].language == "python"
    assert found["runme"].language == "python"             # extensionless, sniffed
    for dropped in ("app.min.js", "package-lock.json", "blob.bin",
                    "node_modules/dep.js", "data.unknownext"):
        assert dropped not in found


def test_discover_files_size_and_line_caps(tmp_path):
    _write(tmp_path / "ok.py", "x = 1\n")                    # 6 bytes, 2 lines
    _write(tmp_path / "manylines.py", "x = 1\n" * 50)        # 50 lines
    _write(tmp_path / "bigbytes.py", "y = '" + "a" * 500 + "'\n")   # ~500 bytes

    by_lines = {fi.file_path for fi in discover_files(str(tmp_path), max_lines=10)}
    assert "ok.py" in by_lines and "manylines.py" not in by_lines

    by_bytes = {fi.file_path for fi in discover_files(str(tmp_path), max_bytes=100)}
    assert "ok.py" in by_bytes and "bigbytes.py" not in by_bytes


def test_discover_files_respects_gitignore(tmp_path):
    pytest.importorskip("pathspec")   # gitignore matching needs the optional dep
    _write(tmp_path / ".gitignore", "ignored.py\nbuild_out/\n")
    _write(tmp_path / "ignored.py", "x = 1\n")
    _write(tmp_path / "build_out" / "gen.py", "y = 2\n")
    _write(tmp_path / "kept.py", "z = 3\n")

    found = {fi.file_path for fi in discover_files(str(tmp_path))}
    assert "kept.py" in found
    assert "ignored.py" not in found
    assert "build_out/gen.py" not in found


def test_discover_files_include_languages_filter(tmp_path):
    _write(tmp_path / "a.py", "x = 1\n")
    _write(tmp_path / "b.js", "var y = 2;\n")
    found = {fi.file_path for fi in discover_files(str(tmp_path),
                                                   include_languages={"python"})}
    assert "a.py" in found and "b.js" not in found
