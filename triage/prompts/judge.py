"""Prompts for the LLM-as-judge (binary verdicts)."""
from __future__ import annotations

METRICS = ["answer_relevancy", "context_relevance", "groundedness"]

_SCORE_DESCRIPTIONS = {
    "answer_relevancy": "Does the decision directly address the new issue's problem?",
    "context_relevance": "Does the retrieved context actually support the decision?",
    "groundedness": "Is every claim supported by a citation present in the retrieved context?",
}


def build_judge_prompt(metric: str, query: str, decision: str, context: str = "") -> str:
    return f"""You are a strict evaluator of an issue-triage copilot.

METRIC: {metric}
{_SCORE_DESCRIPTIONS.get(metric, "")}

Answer with a binary verdict. Respond with JSON only:
{{"verdict": "yes" if the metric holds, "no" otherwise, "reason": "<one sentence>"}}

NEW ISSUE:
{query}

COPIED DECISION:
{decision}

RETRIEVED CONTEXT:
{context or "none"}"""
