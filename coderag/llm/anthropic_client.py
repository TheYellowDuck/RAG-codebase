"""Anthropic (Claude) adapter — the default provider.

Uses the official `anthropic` SDK with adaptive thinking for generation and a
strict `output_config` JSON schema for judging (the most reliable structured
output path). Model defaults to claude-opus-4-8.
"""
from __future__ import annotations

import os
import sys

from .base import Completion, LLMClient, extract_json

# Adaptive thinking is only supported on these model families; sending it to
# others (e.g. Haiku 4.5, Sonnet 4.5) returns a 400. We enable it where valid and
# simply omit the parameter otherwise.
_ADAPTIVE_THINKING_MODELS = ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6")


def _supports_adaptive_thinking(model: str) -> bool:
    return any(m in model for m in _ADAPTIVE_THINKING_MODELS)


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, config):
        super().__init__(config.gen_model, config.judge_model)
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — the Anthropic provider needs it.\n"
                "Get a key at https://console.anthropic.com/ then `export "
                "ANTHROPIC_API_KEY=sk-ant-...`, or use a different provider with "
                "CODERAG_LLM_PROVIDER=openai."
            )
        from .deps import ensure_sdk
        anthropic = ensure_sdk("anthropic", "anthropic")
        # The SDK implements exponential backoff; we just bound it (defaults kept
        # when the env knobs are unset). See coderag/resilience.py.
        opts = {k: v for k, v in (("timeout", config.timeout),
                                  ("max_retries", config.max_retries)) if v is not None}
        self._client = anthropic.Anthropic(**opts)

    def generate(self, system: str, user: str, *, max_tokens: int,
                 stream: bool = False) -> Completion:
        kwargs = dict(
            model=self.gen_model, max_tokens=max_tokens,
            # Mark the system prefix cacheable. Harmless if it's below the model's
            # minimum cacheable size; pays off if the prefix is reused/large.
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if _supports_adaptive_thinking(self.gen_model):
            kwargs["thinking"] = {"type": "adaptive"}
        if stream:
            parts: list[str] = []
            with self._client.messages.stream(**kwargs) as s:
                for text in s.text_stream:
                    parts.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
                final = s.get_final_message()
            sys.stdout.write("\n")
            return Completion("".join(parts), _usage(final))
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text, _usage(resp))

    def judge_json(self, prompt: str, schema: dict, *, max_tokens: int = 2048) -> dict:
        # No thinking param: judging wants fast, deterministic structured output,
        # and omitting it works across all models (incl. Haiku 4.5). Structured
        # outputs (output_config) are supported on Opus 4.8 / Sonnet 4.6 / Haiku 4.5.
        resp = self._client.messages.create(
            model=self.judge_model, max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return extract_json(text)


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    return {"input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None)}
