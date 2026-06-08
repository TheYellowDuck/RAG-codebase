"""Provider-agnostic LLM layer: config resolution, JSON parsing, and that
generation/faithfulness work through an injected client (no SDK/network)."""
import pytest

from coderag.config import LLMConfig
from coderag.llm.base import extract_json, Completion, LLMClient


# --- extract_json -----------------------------------------------------------
def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_embedded_in_prose():
    assert extract_json('Sure! {"grade": "correct"} done.') == {"grade": "correct"}


def test_extract_json_code_fence():
    assert extract_json('```json\n{"x": [1,2]}\n```') == {"x": [1, 2]}


# --- LLMConfig.from_env -----------------------------------------------------
def test_config_defaults_to_anthropic(monkeypatch):
    for k in ("CODERAG_LLM_PROVIDER", "OPENAI_API_KEY", "CODERAG_GEN_MODEL",
              "CODERAG_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    cfg = LLMConfig.from_env()
    assert cfg.provider == "anthropic"
    assert cfg.gen_model == "claude-opus-4-8"
    assert cfg.base_url is None


def test_config_auto_detects_openai(monkeypatch):
    monkeypatch.delenv("CODERAG_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CODERAG_GEN_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    cfg = LLMConfig.from_env()
    assert cfg.provider == "openai"
    assert cfg.gen_model == "gpt-4o"          # provider default
    assert cfg.judge_model == "gpt-4o-mini"


def test_config_explicit_provider_and_base_url(monkeypatch):
    monkeypatch.setenv("CODERAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-x")
    monkeypatch.setenv("CODERAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("CODERAG_GEN_MODEL", "anthropic/claude-3.5-sonnet")
    cfg = LLMConfig.from_env()
    assert cfg.provider == "openai"
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.gen_model == "anthropic/claude-3.5-sonnet"


def test_unknown_provider_raises():
    from coderag.llm import get_llm_client
    with pytest.raises(RuntimeError):
        get_llm_client(LLMConfig(provider="bogus"))


def test_adaptive_thinking_only_for_supported_models():
    # Sending adaptive thinking to Haiku 4.5 / Sonnet 4.5 / older returns a 400,
    # so the Anthropic adapter must gate it by model.
    from coderag.llm.anthropic_client import _supports_adaptive_thinking
    assert _supports_adaptive_thinking("claude-opus-4-8")
    assert _supports_adaptive_thinking("claude-sonnet-4-6")
    assert not _supports_adaptive_thinking("claude-haiku-4-5")
    assert not _supports_adaptive_thinking("claude-sonnet-4-5")


# --- generation + faithfulness through an injected fake client --------------
class FakeLLM(LLMClient):
    """Stub LLMClient — no SDK, no network. Returns canned text/JSON."""
    provider = "fake"

    def __init__(self, gen_text="", json_queue=None):
        super().__init__("fake-gen", "fake-judge")
        self._gen_text = gen_text
        self._json_queue = list(json_queue or [])

    def generate(self, system, user, *, max_tokens, stream=False):
        return Completion(self._gen_text, {"input_tokens": 1, "output_tokens": 1})

    def judge_json(self, prompt, schema, *, max_tokens=2048):
        return self._json_queue.pop(0) if self._json_queue else {}


def test_generate_answer_uses_injected_client():
    from coderag.config import Settings
    from coderag.schema import make_chunk
    from coderag.retrieve import RetrievedChunk
    from coderag.generate import generate_answer

    chunk = make_chunk(repo="r", file_path="a.py", language="python", symbol_name="f",
                       symbol_type="function", start_line=1, end_line=3, code="def f(): ...",
                       context_header="h", git_sha="s")
    res = [RetrievedChunk(chunk, 1.0, 1)]
    fake = FakeLLM(gen_text="It does X [1].")
    out = generate_answer("what?", res, Settings(), client=fake)
    assert out.answer == "It does X [1]."
    assert out.model == "fake-gen"
    assert len(out.sources) == 1


def test_faithfulness_through_fake_client():
    from coderag.config import Settings
    from coderag.generate import Source
    from coderag.verify import faithfulness_score

    sources = [Source(n=1, chunk_id="c1", file_path="a.py", start_line=1, end_line=3,
                      code="def f(): return 422")]
    # Single-call judge: one payload with claims + verdicts.
    fake = FakeLLM(json_queue=[
        {"claims": [
            {"text": "f returns 422", "sources": [1], "supported": True, "reason": "ok"},
            {"text": "g returns 200", "sources": [2], "supported": False, "reason": "no src 2"},
        ]},
    ])
    out = faithfulness_score("f returns 422 [1]. g returns 200 [2].", sources,
                             Settings(), client=fake)
    assert out["n_claims"] == 2
    assert out["n_supported"] == 1
    assert out["faithfulness"] == 0.5
