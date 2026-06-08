"""OpenAI-compatible adapter — OpenAI and anything that speaks its API.

With `CODERAG_LLM_BASE_URL` this reaches OpenRouter (→ Claude/Gemini/Llama/...),
Together, Groq, Azure, and local servers (Ollama, LM Studio, vLLM). That makes
"bring your own key/model — including a free/local one" a single env var.

Uses the official `openai` SDK. Judging uses `response_format=json_object` (widely
supported across compatible servers) plus a tolerant JSON parse, rather than
OpenAI-only strict json_schema.
"""
from __future__ import annotations

import os
import sys

from .base import Completion, LLMClient, extract_json


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, config):
        super().__init__(config.gen_model, config.judge_model)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set — the OpenAI provider needs it.\n"
                "Set OPENAI_API_KEY (for local OpenAI-compatible servers any value "
                "works), and optionally CODERAG_LLM_BASE_URL for a custom endpoint."
            )
        try:
            import openai
        except ImportError as e:  # pragma: no cover - dependency guard
            raise RuntimeError("The 'openai' package is not installed. Run: "
                               "pip install 'coderag[openai]'  (or: pip install openai)") from e
        kwargs = {}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = openai.OpenAI(**kwargs)  # reads OPENAI_API_KEY from env

    def generate(self, system: str, user: str, *, max_tokens: int,
                 stream: bool = False) -> Completion:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        if stream:
            parts: list[str] = []
            for chunk in self._client.chat.completions.create(
                    model=self.gen_model, max_tokens=max_tokens,
                    messages=messages, stream=True):
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    parts.append(delta)
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            sys.stdout.write("\n")
            return Completion("".join(parts), {})
        resp = self._client.chat.completions.create(
            model=self.gen_model, max_tokens=max_tokens, messages=messages)
        return Completion(resp.choices[0].message.content or "", _usage(resp))

    def judge_json(self, prompt: str, schema: dict, *, max_tokens: int = 2048) -> dict:
        resp = self._client.chat.completions.create(
            model=self.judge_model, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "user",
                       "content": prompt + "\n\nRespond with a single JSON object."}],
        )
        return extract_json(resp.choices[0].message.content or "")


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    return {"input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None)}
