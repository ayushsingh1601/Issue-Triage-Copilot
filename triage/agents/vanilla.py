"""Vanilla RAG agent: one LLM call over retrieved context, no orchestration."""
from __future__ import annotations

import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from triage.agents.react import invoke_respond
from triage.guardrails.schema import TriageDecision
from triage.jsonutil import extract_json
from triage.prompts.vanilla import build_vanilla_prompt
from triage.rag.store import Match

FAST_MODEL = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")


def default_respond_fn() -> Any:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=FAST_MODEL, temperature=0)

    class Respond:
        async def ainvoke(self, prompt: str, config: RunnableConfig = None) -> str:
            response = await llm.ainvoke(prompt, config=config)
            return response.content

    return Respond()


class VanillaAgent:
    def __init__(self, respond: Any | None = None) -> None:
        self._respond = respond or default_respond_fn()

    async def run(
        self,
        query: str,
        issue_id: str,
        issue_matches: list[Match],
        doc_matches: list[Match],
        config: RunnableConfig = None,
    ) -> TriageDecision:
        prompt = build_vanilla_prompt(query, issue_matches, doc_matches)
        raw = await invoke_respond(self._respond, prompt, config)
        return TriageDecision.model_validate_json(extract_json(raw))
