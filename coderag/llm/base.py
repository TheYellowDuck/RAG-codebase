"""LLM provider abstraction.

Generation and judging go through a tiny interface so the system isn't tied to
one vendor — anyone can bring their own key/provider. Two methods cover all the
LLM work in this project:

  generate(system, user)  -> grounded answer text (streamed to stdout if asked)
  judge_json(prompt, schema) -> a parsed JSON object (claims, verdicts, grades)

Adapters live in sibling modules (anthropic_client.py, openai_client.py); each
uses its provider's *official* SDK. The default provider is Anthropic (Claude).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    usage: dict = field(default_factory=dict)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating prose/code fences around
    it. Used by judge_json so providers that don't honor strict JSON modes still
    work."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class LLMClient:
    """Provider-agnostic client, bound to a generation model and a judge model."""
    provider: str = "base"

    def __init__(self, gen_model: str, judge_model: str):
        self.gen_model = gen_model
        self.judge_model = judge_model

    def generate(self, system: str, user: str, *, max_tokens: int,
                 stream: bool = False) -> Completion:
        raise NotImplementedError

    def judge_json(self, prompt: str, schema: dict, *, max_tokens: int = 2048) -> dict:
        raise NotImplementedError
