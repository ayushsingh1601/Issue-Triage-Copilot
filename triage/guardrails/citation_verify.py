"""Verify that every cited source exists before it reaches the final output."""
from __future__ import annotations

from triage.guardrails.schema import TriageDecision


def verify_decision(
    decision: TriageDecision,
    issue_ids: set[str],
    doc_ids: set[str],
) -> TriageDecision:
    valid = issue_ids | doc_ids
    cleaned = decision.model_copy(deep=True)
    cleaned.citations = [citation for citation in decision.citations if citation in valid]
    cleaned.similar_issues = [
        issue for issue in decision.similar_issues if issue.issue_id in issue_ids
    ]
    return cleaned


def invalid_citations(
    decision: TriageDecision,
    issue_ids: set[str],
    doc_ids: set[str],
) -> list[str]:
    valid = issue_ids | doc_ids
    return [citation for citation in decision.citations if citation not in valid]
