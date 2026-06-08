"""On-demand grammar provisioning — detection + the opt-in gate (no real install)."""
from coderag.ingest.grammars import (
    missing_grammars, ensure_grammars, auto_install_enabled,
)

_UNKNOWN = "totally_unknown_lang_xyz"  # no tree-sitter grammar will ever load


def test_missing_grammars_detects_unavailable():
    # python grammar is installed; the bogus language is not.
    assert missing_grammars({"python"}) == set()
    assert missing_grammars({"python", _UNKNOWN}) == {_UNKNOWN}


def test_ensure_grammars_noop_when_all_present():
    report = ensure_grammars({"python"}, auto_install=False, progress=False)
    assert report["installed"] is False
    assert report["still_missing"] == set()


def test_ensure_grammars_reports_missing_without_installing():
    # auto_install=False must NOT attempt an install; just report the gap.
    report = ensure_grammars({"python", _UNKNOWN}, auto_install=False, progress=False)
    assert report["installed"] is False
    assert _UNKNOWN in report["still_missing"]


def test_auto_install_gate(monkeypatch):
    monkeypatch.delenv("CODERAG_AUTO_INSTALL_GRAMMARS", raising=False)
    assert auto_install_enabled(False) is False
    assert auto_install_enabled(True) is True
    monkeypatch.setenv("CODERAG_AUTO_INSTALL_GRAMMARS", "1")
    assert auto_install_enabled(False) is True
