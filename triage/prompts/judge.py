"""Prompts for the LLM-as-judge (binary verdicts)."""
from __future__ import annotations

METRICS = ["answer_relevancy", "context_relevance", "groundedness"]

_SCORE_DESCRIPTIONS = {
    "answer_relevancy": "Does the decision directly address the new issue's problem?",
    "context_relevance": "Does the retrieved context actually support the decision?",
    "groundedness": (
        "Are the decision's substantive claims (suggested labels, triage route, "
        "affected modules, and any factual statements about the issue) supported by, "
        "and traceable to, the cited sources in the retrieved context? Next-step "
        "recommendations may be synthesized but must follow from that evidence."
    ),
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
