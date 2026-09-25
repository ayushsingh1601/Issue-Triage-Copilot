"""LLM-as-judge scorers for triage decisions."""
from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel
from triage.jsonutil import extract_json
from triage.prompts.judge import METRICS, build_judge_prompt


class JudgeScore(BaseModel):
    metric: str
    score: int
    reason: str


class Judge:
    def __init__(self, respond: Any | None = None) -> None:
        self._respond = respond or default_judge_respond()

    def score(self, metric: str, query: str, decision: str, context: str = "") -> JudgeScore:
        prompt = build_judge_prompt(metric, query, decision, context)
        raw = self._respond(prompt)
        data = json.loads(extract_json(raw))
        return JudgeScore(
            metric=metric,
            score=int(data["score"]),
            reason=str(data.get("reason", "")),
        )

    def evaluate(self, query: str, decision: str, context: str = "") -> dict[str, JudgeScore]:
        return {metric: self.score(metric, query, decision, context) for metric in METRICS}


def default_judge_respond() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    return lambda prompt: llm.invoke(prompt).content
