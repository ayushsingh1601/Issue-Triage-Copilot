"""LLM-as-judge scorers for triage decisions.

Each metric is scored by its own LLM call (one `respond` per metric, or a shared
default). Verdicts are binary: `yes` -> score 1, `no` -> score 0.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from triage.jsonutil import extract_json
from triage.prompts.judge import METRICS, build_judge_prompt


class JudgeScore(BaseModel):
    metric: str
    verdict: str
    score: int
    reason: str


class Judge:
    def __init__(
        self,
        respond: Callable[[str], str] | None = None,
        responds: dict[str, Callable[[str], str]] | None = None,
    ) -> None:
        self._default = respond
        self._responds = responds or {}

    def score(self, metric: str, query: str, decision: str, context: str = "") -> JudgeScore:
        prompt = build_judge_prompt(metric, query, decision, context)
        raw = self._respond_for(metric)(prompt)
        data = json.loads(extract_json(raw))
        verdict = str(data.get("verdict", "")).strip().lower()
        return JudgeScore(
            metric=metric,
            verdict=verdict,
            score=1 if verdict == "yes" else 0,
            reason=str(data.get("reason", "")),
        )

    def _respond_for(self, metric: str) -> Callable[[str], str]:
        respond = self._responds.get(metric)
        if respond is None:
            respond = self._default or default_judge_respond()
        return respond

    def evaluate(self, query: str, decision: str, context: str = "") -> dict[str, JudgeScore]:
        return {metric: self.score(metric, query, decision, context) for metric in METRICS}


def default_judge_respond() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    return lambda prompt: llm.invoke(prompt).content
