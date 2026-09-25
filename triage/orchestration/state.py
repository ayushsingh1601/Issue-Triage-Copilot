"""Shared graph state for the multi-agent triage flow."""
from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel
from triage.guardrails.schema import TriageDecision


class TriageState(BaseModel):
    issue: str = ""
    issue_id: str = ""
    classification: dict[str, Any] = {}
    historical_evidence: list[Any] = []
    runbook_steps: list[Any] = []
    decision: TriageDecision | None = None
    confidence: float = 0.0
    needs_human: bool = False
    citations: Annotated[list[str], operator.add] = []
