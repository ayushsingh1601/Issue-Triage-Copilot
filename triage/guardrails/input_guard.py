"""Input guardrail: prompt-injection detection and triage relevance check.

The injection check is a deterministic rule set (fast, no LLM). The relevance
check is a binary LLM verdict (injectable for tests). A rejected query stops the
graph before any agent or LLM work happens.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from triage.jsonutil import extract_json
from triage.prompts.guard import build_relevance_prompt

MIN_LENGTH = 10

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|earlier)\s+(instructions|messages|prompts)",
    r"disregard\s+(all\s+)?(previous|prior|earlier)\s+(instructions|messages|prompts)",
    r"reveal\s+(your|the)\s+system\s+prompt",
    r"(?:show|print|output|repeat|leak)\s+(your\s+)?(system\s+)?prompt",
    r"you\s+are\s+now\s+",
    r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+(an?\s+)?(developer|assistant|robot)",
    r"jailbreak",
    r"forget\s+(all\s+)?(previous|prior|earlier)",
    r"override\s+(your\s+)?(instructions|prompt)",
    r"developer\s+mode",
    r"do\s+anything\s+now",
]


class GuardResult(BaseModel):
    allowed: bool
    reason: str


def default_relevance_respond() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    return lambda prompt: llm.invoke(prompt).content


class InputGuard:
    def __init__(
        self,
        relevance_respond: Callable[[str], str] | None = None,
    ) -> None:
        self._relevance_respond = relevance_respond or default_relevance_respond()

    def guard(self, query: str) -> GuardResult:
        text = query.strip()
        if len(text) < MIN_LENGTH:
            return GuardResult(allowed=False, reason=f"query too short ({len(text)} chars)")
        reason = self.check_injection(text)
        if reason:
            return GuardResult(allowed=False, reason=reason)
        if not self.is_relevant(text):
            return GuardResult(allowed=False, reason="query is not a relevant triage issue")
        return GuardResult(allowed=True, reason="ok")

    def check_injection(self, query: str) -> str | None:
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return f"injection pattern matched: {pattern}"
        return None

    def is_relevant(self, query: str) -> bool:
        raw = self._relevance_respond(build_relevance_prompt(query))
        data = json.loads(extract_json(raw))
        return str(data.get("verdict", "")).strip().lower() == "yes"
