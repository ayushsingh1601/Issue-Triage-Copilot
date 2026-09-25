"""Prompts for the orchestrator's final decision."""
from __future__ import annotations

import json
from typing import Any

from triage.orchestration.state import TriageState

SCHEMA_DESCRIPTION = """{
  "issue_id": "string",
  "suggested_labels": ["string"],
  "triage_route": "string",
  "next_steps": ["string"],
  "affected_modules": ["string"],
  "similar_issues": [
    {"issue_id": "string", "repo": "string", "title": "string", "reason": "string"}
  ],
  "citations": ["string"]
}"""


def build_orchestrator_prompt(
    state: TriageState,
    resolution_patterns: dict[str, Any] | None = None,
) -> str:
    historical = json.dumps(state.historical_evidence, indent=2) or "none"
    runbook = json.dumps(state.runbook_steps, indent=2) or "none"
    patterns = json.dumps(resolution_patterns, indent=2) if resolution_patterns else "none"
    return f"""You are the orchestrator of an issue-triage copilot. Merge the specialist
evidence below into a final decision JSON.

NEW ISSUE (id: {state.issue_id}):
{state.issue}

CLASSIFICATION: {state.classification}

SIMILAR RESOLVED ISSUES (from historical evidence):
{historical}

PROCESS RUNBOOK STEPS (from process evidence):
{runbook}

RESOLUTION PATTERNS:
{patterns}

Respond with JSON only, exactly this schema:
{SCHEMA_DESCRIPTION}

Rules:
- suggested_labels: concrete labels appropriate for this repo.
- triage_route: route for the issue type with any needed notes.
- next_steps: concrete, actionable steps, each grounded in a citation.
- affected_modules: likely affected modules, grounded in evidence.
- similar_issues: pick 2-4 from SIMILAR RESOLVED ISSUES, using their exact issue_id.
- citations: every issue as repo#number and every doc section by its source id from RUNBOOK STEPS.
Only cite sources present in the evidence above."""
