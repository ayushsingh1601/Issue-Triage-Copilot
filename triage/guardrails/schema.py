"""Output schema for triage decisions (shared by vanilla baseline and orchestrator)."""
from __future__ import annotations

from pydantic import BaseModel


class SimilarIssue(BaseModel):
    issue_id: str
    repo: str
    title: str
    reason: str


class TriageDecision(BaseModel):
    issue_id: str
    suggested_labels: list[str]
    triage_route: str
    next_steps: list[str]
    affected_modules: list[str]
    similar_issues: list[SimilarIssue]
    citations: list[str]


def format_decision(decision: TriageDecision | None) -> str:
    if decision is None:
        return "(no decision produced)"
    lines = [
        f"Suggested labels: {', '.join(decision.suggested_labels) or 'none'}",
        f"Triage route: {decision.triage_route}",
        "Next steps:",
        *(f"  - {step}" for step in decision.next_steps),
        f"Affected modules: {', '.join(decision.affected_modules) or 'none'}",
    ]
    if decision.similar_issues:
        lines.append("Similar resolved issues:")
        lines.extend(
            f"  - {item.issue_id}: {item.title} ({item.reason})"
            for item in decision.similar_issues
        )
    else:
        lines.append("Similar resolved issues: none")
    lines.append(f"Citations: {', '.join(decision.citations) or 'none'}")
    return "\n".join(lines)
