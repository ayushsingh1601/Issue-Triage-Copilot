"""Vanilla RAG agent: one LLM call over retrieved context, no orchestration."""
from __future__ import annotations

import os
import re
from collections.abc import Callable

from triage.guardrails.schema import TriageDecision
from triage.prompts.vanilla import build_vanilla_prompt
from triage.rag.store import Match

FAST_MODEL = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")

RespondFn = Callable[[str], str]


def default_respond_fn() -> RespondFn:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=FAST_MODEL, temperature=0)

    def respond(prompt: str) -> str:
        return llm.invoke(prompt).content

    return respond


class VanillaAgent:
    def __init__(self, respond: RespondFn | None = None) -> None:
        self._respond = respond or default_respond_fn()

    def run(
        self,
        query: str,
        issue_id: str,
        issue_matches: list[Match],
        doc_matches: list[Match],
    ) -> TriageDecision:
        prompt = build_vanilla_prompt(query, issue_matches, doc_matches)
        raw = self._respond(prompt)
        return TriageDecision.model_validate_json(_extract_json(raw))


def _extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text
