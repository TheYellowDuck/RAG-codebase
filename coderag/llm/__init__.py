"""Provider-agnostic LLM layer (generation + judging).

`get_llm_client()` resolves the provider from the environment (default Anthropic)
and returns a ready client. Bring your own key/provider:

    # Anthropic (default)
    export ANTHROPIC_API_KEY=sk-ant-...

    # OpenAI
    export CODERAG_LLM_PROVIDER=openai OPENAI_API_KEY=sk-...

    # Any OpenAI-compatible endpoint (OpenRouter / Together / Groq / Ollama / ...)
    export CODERAG_LLM_PROVIDER=openai \
           CODERAG_LLM_BASE_URL=https://openrouter.ai/api/v1 \
           OPENAI_API_KEY=sk-or-... CODERAG_GEN_MODEL=anthropic/claude-3.5-sonnet
"""
from __future__ import annotations

from typing import Optional

from ..config import LLMConfig
from .base import Completion, LLMClient, extract_json

__all__ = ["LLMClient", "Completion", "extract_json", "LLMConfig", "get_llm_client"]


def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Return an LLM client for the configured provider (raises with guidance if
    the provider's key/SDK is missing)."""
    config = config or LLMConfig.from_env()
    if config.provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(config)
    if config.provider in ("openai", "openai-compatible"):
        from .openai_client import OpenAIClient
        return OpenAIClient(config)
    raise RuntimeError(
        f"Unknown CODERAG_LLM_PROVIDER={config.provider!r}. "
        f"Use 'anthropic' (default) or 'openai'."
    )
