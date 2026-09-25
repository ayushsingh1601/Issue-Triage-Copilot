"""Query rewriter: distills a raw issue into one clean retrieval query via an LLM."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from triage.jsonutil import extract_json
from triage.prompts.rewrite import build_rewrite_prompt


def default_rewrite_respond() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    return lambda prompt: llm.invoke(prompt).content


class QueryRewriter:
    def __init__(self, respond: Callable[[str], str] | None = None) -> None:
        self._respond = respond or default_rewrite_respond()

    def rewrite(self, query: str) -> str:
        raw = self._respond(build_rewrite_prompt(query))
        data = json.loads(extract_json(raw))
        return str(data["query"])
