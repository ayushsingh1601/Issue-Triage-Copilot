"""Stub graph nodes; real implementations land in later steps."""
from __future__ import annotations

from triage.orchestration.state import TriageState


def plan_node(state: TriageState) -> dict:
    return {"classification": {"type": "", "confidence": 0.0}}


def historical_node(state: TriageState) -> dict:
    return {"historical_evidence": []}


def process_node(state: TriageState) -> dict:
    return {"runbook_steps": []}


def decide_node(state: TriageState) -> dict:
    return {"decision": None, "needs_human": False, "confidence": 0.0}


def human_in_loop_node(state: TriageState) -> dict:
    return {"needs_human": True}
