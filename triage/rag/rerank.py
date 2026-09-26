"""LLM reranker: re-ranks a candidate pool in a single call."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from triage.agents.react import invoke_respond_sync
from triage.jsonutil import extract_json
from triage.prompts.rerank import build_rerank_prompt
from triage.rag.store import Match


def default_rerank_respond() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    return lambda prompt, config=None: llm.invoke(prompt, config=config).content


class Reranker:
    def __init__(self, respond: Callable[[str], str] | None = None) -> None:
        self._respond = respond or default_rerank_respond()

    def rerank(
        self,
        query: str,
        matches: list[Match],
        top_k: int,
        config: RunnableConfig = None,
    ) -> list[Match]:
        if len(matches) <= 1:
            return matches[:top_k]
        raw = invoke_respond_sync(self._respond, build_rerank_prompt(query, matches), config)
        data = json.loads(extract_json(raw))
        ordered = _order(data.get("order", []), len(matches))
        return [matches[index] for index in ordered[:top_k]]


def _order(indices: list[Any], n: int) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for index in indices:
        try:
            value = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= value < n and value not in seen:
            seen.add(value)
            ordered.append(value)
    ordered.extend(index for index in range(n) if index not in seen)
    return ordered
