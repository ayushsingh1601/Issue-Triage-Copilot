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
