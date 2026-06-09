"""Resilience helpers + LLM timeout/retry wiring (coderag/resilience.py, config)."""
import types

import pytest

from coderag import resilience
from coderag.config import LLMConfig


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(resilience.time, "sleep", lambda _s: None)  # don't wait in tests


def test_with_retry_succeeds_first_try():
    assert resilience.with_retry(lambda: 7) == 7


def test_with_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("connection reset by peer")
        return "ok"

    assert resilience.with_retry(flaky, attempts=5) == "ok"
    assert calls["n"] == 3


def test_with_retry_gives_up_and_reraises_last():
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        resilience.with_retry(always, attempts=3)
    assert calls["n"] == 3      # tried exactly `attempts` times


def test_with_retry_does_not_retry_non_transient():
    calls = {"n": 0}

    def bug():
        calls["n"] += 1
        raise ValueError("bad model name")     # not transient -> fail fast

    with pytest.raises(ValueError):
        resilience.with_retry(bug, attempts=5)
    assert calls["n"] == 1                      # no retries


def test_is_transient_classification():
    assert resilience.is_transient(ConnectionError("connection reset"))
    assert resilience.is_transient(TimeoutError("timed out"))
    assert resilience.is_transient(RuntimeError("Service Unavailable"))

    class HttpErr(Exception):
        status_code = 503
    assert resilience.is_transient(HttpErr())

    assert not resilience.is_transient(ValueError("nope"))
    assert not resilience.is_transient(KeyError("missing"))


def test_llmconfig_reads_resilience_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("CODERAG_LLM_TIMEOUT", "42.5")
    monkeypatch.setenv("CODERAG_LLM_MAX_RETRIES", "5")
    cfg = LLMConfig.from_env()
    assert cfg.timeout == 42.5 and cfg.max_retries == 5


def test_llmconfig_resilience_defaults_to_none(monkeypatch):
    monkeypatch.delenv("CODERAG_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("CODERAG_LLM_MAX_RETRIES", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = LLMConfig.from_env()
    assert cfg.timeout is None and cfg.max_retries is None


def test_anthropic_client_passes_resilience_knobs(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kw):
            captured.update(kw)

    import coderag.llm.deps as deps
    monkeypatch.setattr(deps, "ensure_sdk",
                        lambda *a, **k: types.SimpleNamespace(Anthropic=FakeAnthropic))
    from coderag.llm.anthropic_client import AnthropicClient

    AnthropicClient(LLMConfig(timeout=30.0, max_retries=4))
    assert captured == {"timeout": 30.0, "max_retries": 4}

    captured.clear()
    AnthropicClient(LLMConfig())            # unset -> keep SDK defaults (no kwargs)
    assert captured == {}
