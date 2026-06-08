"""Auto-install of the provider SDK (coderag/llm/deps.py).

We exercise the no-network branches: an already-importable package returns
without installing, and a missing package with auto-install disabled raises the
manual pip command. We avoid triggering a real `pip install` in CI.
"""
import pytest

from coderag.llm import deps


def test_ensure_sdk_returns_installed_module_without_installing(monkeypatch):
    called = {"install": False}
    monkeypatch.setattr(deps.subprocess, "run",
                        lambda *a, **k: called.__setitem__("install", True))
    mod = deps.ensure_sdk("json", "json")        # stdlib, always importable
    assert mod.__name__ == "json"
    assert called["install"] is False            # never tried to install


def test_ensure_sdk_respects_opt_out(monkeypatch):
    monkeypatch.setenv("CODERAG_NO_AUTO_INSTALL", "1")
    monkeypatch.setattr(deps.subprocess, "run",
                        lambda *a, **k: pytest.fail("should not install when opted out"))
    with pytest.raises(RuntimeError) as ei:
        deps.ensure_sdk("coderag_definitely_missing_pkg", "some-pip-spec")
    assert "pip install some-pip-spec" in str(ei.value)


def test_auto_install_enabled_default_and_opt_out(monkeypatch):
    monkeypatch.delenv("CODERAG_NO_AUTO_INSTALL", raising=False)
    assert deps.auto_install_enabled() is True
    monkeypatch.setenv("CODERAG_NO_AUTO_INSTALL", "yes")
    assert deps.auto_install_enabled() is False
