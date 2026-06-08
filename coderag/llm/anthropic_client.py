"""Anthropic (Claude) adapter — the default provider.

Uses the official `anthropic` SDK with adaptive thinking for generation and a
strict `output_config` JSON schema for judging (the most reliable structured
output path). Model defaults to claude-opus-4-8.
"""
from __future__ import annotations

import os
import sys

from .base import Completion, LLMClient, extract_json


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
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - dependency guard
            raise RuntimeError("The 'anthropic' package is not installed. Run: "
                               "pip install anthropic") from e
        self._client = anthropic.Anthropic()

    def generate(self, system: str, user: str, *, max_tokens: int,
                 stream: bool = False) -> Completion:
        kwargs = dict(
            model=self.gen_model, max_tokens=max_tokens,
            thinking={"type": "adaptive"}, system=system,
            messages=[{"role": "user", "content": user}],
        )
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
        resp = self._client.messages.create(
            model=self.judge_model, max_tokens=max_tokens,
            thinking={"type": "disabled"},
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
